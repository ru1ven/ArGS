#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
import os
import sys
from datetime import datetime
import numpy as np
import random

import torch.nn as nn
import lpips
import cv2
import yaml
from easydict import EasyDict
from kornia.geometry import rotation_matrix_to_angle_axis
from pytorch3d.io import load_obj
from libyana.meshutils import meshio
from skimage.metrics import structural_similarity as compute_ssim

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

_YCB_CLASSES = {
        1: '002_master_chef_can',
        2: '003_cracker_box',
        3: '004_sugar_box',
        4: '005_tomato_soup_can',
        5: '006_mustard_bottle',
        6: '007_tuna_fish_can',
        7: '008_pudding_box',
        8: '009_gelatin_box',
        9: '010_potted_meat_can',
        10: '011_banana',
        11: '019_pitcher_base',
        12: '021_bleach_cleanser',
        13: '024_bowl',
        14: '025_mug',
        15: '035_power_drill',
        16: '036_wood_block',
        17: '037_scissors',
        18: '040_large_marker',
        19: '051_large_clamp',
        20: '052_extra_large_clamp',
        21: '061_foam_brick',
    }


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) format.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: small value to avoid zero division.

    Return:
        the rotation in quaternion with shape :math:`(*, 4)`.

    Example:
        >>> input = tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond() -> torch.Tensor:
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def quaternion_multiply(r, s):
    r0, r1, r2, r3 = r.unbind(-1)
    s0, s1, s2, s3 = s.unbind(-1)
    t0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3
    t1 = r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2
    t2 = r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1
    t3 = r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0
    t = torch.stack([t0, t1, t2, t3], dim=-1)
    return t

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    if r.shape[-1] == 4:
        # quaternion to matrix
        R = build_rotation(r)
    else:
        R = r

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def fix_random(seed):
    if seed >= 0:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

# evaluation metrics
class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()

    def forward(self, inputs, targets):
        psnr = self.psnr(inputs, targets)
        ssim = self.ssim(inputs, targets)
        lpips_ = self.lpips(inputs, targets)
        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_,
        }

class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets) ** 2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        device = inputs.device
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        if valid_mask is not None:
            valid_mask = valid_mask.cpu().numpy()
            x, y, w, h = cv2.boundingRect(valid_mask.astype(np.uint8))
            img_pred = inputs[y:y + h, x:x + w]
            img_gt = targets[y:y + h, x:x + w]
        else:
            img_pred = inputs
            img_gt = targets

        # compute ssim
        ssim = compute_ssim(img_pred, img_gt, channel_axis=0)
        ssim = torch.tensor(ssim, device=device)
        return ssim


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        self.loss_fn_vgg.eval()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        if valid_mask is not None:
            x, y, w, h = cv2.boundingRect(valid_mask.cpu().numpy().astype(np.uint8))
            img_pred = inputs[:, y:y + h, x:x + w]
            img_gt = targets[:, y:y + h, x:x + w]
        else:
            img_pred = inputs
            img_gt = targets

        score = self.loss_fn_vgg(img_pred, img_gt, normalize=True)
        return score.flatten()

class PSEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.cuda()
        self.eval()

    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        rgb = rgb.unsqueeze(0)
        rgb_gt = rgb_gt.unsqueeze(0)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }

def points3DToImg(joint_xyz, para):
    joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
    joint_uvd[:, :, 0] = (
            joint_xyz[:, :, 0] * para[:, 0].unsqueeze(1) / (joint_xyz[:, :, 2] + 1e-8) + para[:, 2].unsqueeze(
        1))
    joint_uvd[:, :, 1] = (
            joint_xyz[:, :, 1] * para[:, 1].unsqueeze(1) / (joint_xyz[:, :, 2]) + para[:, 3].unsqueeze(
        1))
    joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
    return joint_uvd


def cal_pose_error(camera, body_model):

    # body_gt = body_model(global_orient=camera.hand_param_gt[:, :3],
    #                                       hand_pose=camera.hand_param_gt[:, 3:48], betas=camera.hand_param_gt[:, 48:58],
    #                                       transl=camera.hand_param_gt[:, 58:])
    #
    # Jtr_posed_gt = body_gt['joints'].detach().cpu().numpy()

    Jtr_posed_gt = camera.joints_gt.view(21, 3).detach().cpu().numpy()
    body = body_model(global_orient=camera.hand_param[:, :3],
                             hand_pose=camera.hand_param[:, 3:48], betas=camera.hand_param[:, 48:58],
                             transl=camera.hand_param[:, 58:])

    Jtr_posed = body['joints'].detach().cpu().numpy()[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    #print(np.linalg.norm(Jtr_posed_gt - Jtr_posed, axis=-1).mean())
    return np.linalg.norm(Jtr_posed_gt - Jtr_posed, axis=-1).mean()


def relative_pose_error(camera):
    # angle error between 2 vectors
    R, t, R_gt, t_gt, = rotation_matrix_to_angle_axis(camera.obj_rots).detach().cpu().numpy(), camera.obj_trans.detach().cpu().numpy(),\
                        rotation_matrix_to_angle_axis(camera.obj_rots_gt).detach().cpu().numpy(), camera.obj_trans_gt.detach().cpu().numpy()
    t_err = np.linalg.norm(t.reshape(-1,3) - t_gt.reshape(-1,3), axis=-1)
    return t_err

def compute_obj_metrics_ycb(obj_rot_gt,obj_trans_gt,obj_rot,obj_trans,obj_label):
    templates = prepare_model_template(
        obj_root='/home/cyc/pycharm/lxy/HOISDF/dataset/DexYCB/simple_models/')

    template_meshes = torch.stack(
        [templates[_YCB_CLASSES[int(obj_label)]]["verts"].clone().detach()]
    ).cuda()
    sample_nums = 1

    target_meshes = (
            torch.bmm(
                template_meshes,
                obj_rot_gt.cuda().view(sample_nums, 3, 3).permute(0, 2, 1),
            )
            + obj_trans_gt.view(sample_nums, 1, 3).cuda()
    )
    pred_meshes = (
            torch.bmm(
                template_meshes,
                obj_rot.view(sample_nums, 3, 3).cuda().permute(0, 2, 1).float(),
            )
            + obj_trans.view(sample_nums, 1, 3).cuda().float()
    )



    B, N, _ = pred_meshes.shape
    add_gt = target_meshes.unsqueeze(1).repeat(1, N, 1, 1)
    add_pred = pred_meshes.unsqueeze(2).repeat(1, 1, N, 1)
    dis = torch.norm(add_gt - add_pred, dim=-1)
    add_bias = torch.mean(torch.min(dis, dim=2)[0], dim=1)
    add_bias = add_bias.detach().cpu()

    corner_indexes = torch.tensor(
        [[0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1]]
    ).cuda()
    target_mm = torch.stack(
        [torch.min(target_meshes, dim=1)[0], torch.max(target_meshes, dim=1)[0]], dim=2
    )
    target_bboxes = torch.stack(
        [
            target_mm[:, 0, corner_indexes[0]],
            target_mm[:, 1, corner_indexes[1]],
            target_mm[:, 2, corner_indexes[2]],
        ],
        dim=2,
    )
    pred_mm = torch.stack(
        [torch.min(pred_meshes, dim=1)[0], torch.max(pred_meshes, dim=1)[0]], dim=2
    )
    pred_bboxes = torch.stack(
        [
            pred_mm[:, 0, corner_indexes[0]],
            pred_mm[:, 1, corner_indexes[1]],
            pred_mm[:, 2, corner_indexes[2]],
        ],
        dim=2,
    )

    MCE_error = (
        (pred_bboxes - target_bboxes.float()).norm(2, -1).mean(-1).detach().cpu()
    )

    return add_bias, MCE_error


def prepare_model_template(obj_root):
    templates = {}  # faces order depends on the os.listdir


    for obj in sorted(os.listdir(obj_root)):
        path = os.path.join(obj_root, obj, "textured_simple_2000.obj")

        with open(path) as m_f:
            mesh = meshio.fast_load_obj(m_f)[0]
            if mesh["vertices"].shape[0] != 1000 or "meshlab" in obj_root:
                verts, faces, aux = load_obj(path)
                assert verts.shape[0] == 1000
                templates[obj] = {
                        "verts": verts,
                        "face": faces.verts_idx.long(),
                    }

            else:
                templates[obj] = {
                        "verts": torch.Tensor(mesh["vertices"]),
                        "face": torch.Tensor(mesh["faces"]).long(),
                    }

    return templates

def get_jtr(body):
    Jtrs = body['Jtr_a_pose']

    v_shaped = body['v_shaped']
    v_shaped = v_shaped.detach()

    center = torch.mean(v_shaped, dim=1).unsqueeze(1)
    minimal_shape_centered = v_shaped - center
    cano_max = minimal_shape_centered.max()
    cano_min = minimal_shape_centered.min()
    padding = (cano_max - cano_min) * 0.05

    # compute pose condition
    Jtrs = Jtrs - center
    Jtrs = (Jtrs - cano_min + padding) / (cano_max - cano_min) / 1.1
    Jtrs -= 0.5
    Jtrs *= 2.
    Jtrs = Jtrs.contiguous()
    return Jtrs


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config


def tensor_to_numpy_image(tensor):
    """
    将 torch.Tensor 图像（1xHxW 或 3xHxW）转为 swanlab.Image 可接受的 numpy 图像
    支持 float32 / float64（会归一化并转 uint8）
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

        # 去 batch 维度
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # (C,H,W)

        if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
            # CHW -> HWC or HW
            if tensor.shape[0] == 1:
                img = tensor.squeeze(0)  # (H, W)
            else:
                img = tensor.permute(1, 2, 0)  # (H, W, 3)
        elif tensor.ndim == 2:
            img = tensor  # 已经是 (H, W)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        # 转 numpy + 类型处理
        img = img.numpy()
        if img.dtype in [np.float32, np.float64]:
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

        return img
    else:
        raise TypeError("Input must be a torch.Tensor")


def update_frame_weights(delta_history):
    scores = []
    frame_indices = sorted(delta_history.keys())

    #num_frames = len(delta_history)
    for frame_idx in frame_indices:
        delta_xyz, delta_scale, delta_rot = delta_history[frame_idx]
        delta_xyz_norm = delta_xyz.norm(dim=1)  # [N]
        delta_scale_norm = delta_scale.norm(dim=1)  # [N]
        delta_rot_norm = delta_rot.norm(dim=1)  # [N]

        # 每帧的整体变化量
        xyz_mean = delta_xyz_norm.mean()
        scale_mean = delta_scale_norm.mean()
        rot_mean = delta_rot_norm.mean()

        score = xyz_mean + 0.1 * scale_mean + 0.2 * rot_mean
        scores.append(score.item())

    # === 2. 根据变化度更新权重 ===
    scores = torch.tensor(scores)
    frame_weights = (scores + 1e-6) / (scores.sum() + 1e-6)
    return frame_weights


def save_deltas(delta_history, xyz, filename, thrs=[0.1,  0.2, 0.3, 0.5], norm=True):
    # 累加每帧 delta
    total_delta = None

    # 遍历每帧
    frames = sorted(delta_history.keys())
    frame_deltas = []
    for f in frames:
        delta_xyz = delta_history[f].detach()
        if norm:
            delta_xyz -= delta_history[frames[0]].detach()
        frame_deltas.append(delta_xyz.detach().cpu())
    # 帧平均
    avg_delta = torch.stack(frame_deltas, dim=0).mean(dim=0)  # [N,3]
    if total_delta is None:
        total_delta = avg_delta
    else:
        total_delta = total_delta + avg_delta

    # frame_indices = sorted(delta_history.keys())
    # # 假设所有帧点云数量相同
    # N = delta_history[frame_indices[0]][0].shape[0]
    #
    # # 累积每点xyz移动量
    # delta_sum = torch.zeros(N, 3)
    # for idx in frame_indices:
    #     delta_xyz, _, _ = delta_history[idx]
    #     delta_sum += delta_xyz
    #
    # # 平均每点移动量
    # delta_avg = delta_sum / len(frame_indices)

    delta_norm = total_delta.norm(dim=1)  # [N]

    # 归一化到0~1
    if norm:
        delta_norm = delta_norm / (delta_norm.max() + 1e-6)

    # 红蓝映射: 蓝(小) -> 红(大)
    colors = torch.zeros(len(delta_norm), 3)
    colors[:, 0] = delta_norm       # R
    colors[:, 2] = 1.0 - delta_norm # B

    # 点位置可以取最后一帧的delta累加或者已有点位置

    colors = colors.detach().cpu().numpy()

    with open(filename, 'w') as f:
        for p, c in zip(xyz, colors):
            f.write('v {} {} {} {} {} {}\n'.format(p[0], p[1], p[2], c[0], c[1], c[2]))
        # 没有面信息，只保存点云
    print(f"Saved {filename} with vertex colors.")

    for thr in thrs:

        # 红蓝映射：超过阈值为红色，其余为蓝色
        colors = torch.zeros(len(delta_norm), 3)
        mask_red = delta_norm >= thr
        colors[mask_red, 0] = 1.0  # R
        colors[mask_red, 1] = 0.0
        colors[mask_red, 2] = 0.0
        colors[~mask_red, 0] = 0.0
        colors[~mask_red, 1] = 0.0
        colors[~mask_red, 2] = 1.0  # B
        colors = colors.detach().cpu().numpy()


        with open(f"{filename}_th{thr}.obj", 'w') as f:
            for p, c in zip(xyz, colors):
                f.write('v {} {} {} {} {} {}\n'.format(p[0], p[1], p[2], c[0], c[1], c[2]))
        print(f"Saved {filename} with threshold={thr}")

    return delta_norm



def positional_encoding(x, m=6):
    """
    x: [..., 3]  输入点 (可以是 [N, 3] 批量)
    m: int       每维使用的频率数量
    return: [..., 3 + 6m]
    """
    freqs = 2 ** torch.arange(m, dtype=torch.float32, device=x.device) * math.pi
    # x[..., None]: [N, 3, 1]
    # freqs[None, None, :]: [1, 1, m]
    angles = x[..., None] * freqs[None, None, :]  # [N, 3, m]

    sin_part = torch.sin(angles)  # [N, 3, m]
    cos_part = torch.cos(angles)  # [N, 3, m]

    # 拼接 sin 和 cos
    encoded = torch.cat([sin_part, cos_part], dim=-1)  # [N, 3, 2m]
    encoded = encoded.reshape(*x.shape[:-1], 3 * 2 * m)  # [N, 6m]

    return torch.cat([x, encoded], dim=-1)  # [N, 3 + 6m]

# import pyvista as pv
# def visualize_axis_pointcloud(xyz, pivot, axis, save_path="axis_pcl.png", point_size=10):
#     """
#     xyz: (N,3) numpy array 点云
#     pivot: (3,) numpy array
#     axis: (3,) numpy array, 单位向量
#     """
#     xyz = np.asarray(xyz)
#     pivot = np.asarray(pivot)
#     axis = np.asarray(axis) / np.linalg.norm(axis)
#
#     plotter = pv.Plotter(off_screen=True)
#
#     # 点云，灰白色，体积感
#     plotter.add_points(xyz, color='lightgrey', point_size=point_size)
#
#     # pivot 蓝球
#     plotter.add_mesh(pv.Sphere(radius=0.02, center=pivot), color='blue')
#
#     # 旋转轴：红色圆柱 + 箭头
#     axis_len = 0.5 * np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
#     cyl = pv.Cylinder(center=pivot + axis*axis_len*0.45, direction=axis, radius=0.01, height=axis_len*0.9)
#     plotter.add_mesh(cyl, color='red')
#     arrow = pv.Arrow(start=pivot, direction=axis, scale=axis_len*0.1)
#     plotter.add_mesh(arrow, color='red')
#
#     # 相机自适应
#     plotter.set_focus(pivot)
#     plotter.set_viewup([0,0,1])
#     plotter.show(auto_close=False)  # 必须渲染一次
#     plotter.screenshot(save_path)
#     plotter.close()
#     print(f"✅ 图片已保存到 {save_path}")


# import open3d as o3d
# def visualize_axis_pointcloud(xyz_norm, pivot_norm, axis, filename="axis_pcl.png"):
#     """
#     xyz_norm: (N,3) torch tensor 或 numpy array, 点云
#     pivot_norm: (3,) torch tensor 或 numpy array, 旋转中心
#     axis: (3,) torch tensor 或 numpy array, 旋转轴
#     filename: 保存图片文件名
#     """
#
#     # 转 numpy
#     if isinstance(xyz_norm, torch.Tensor):
#         xyz = xyz_norm.detach().cpu().numpy()
#     else:
#         xyz = np.array(xyz_norm)
#
#     if isinstance(pivot_norm, torch.Tensor):
#         pivot = pivot_norm.detach().cpu().numpy()
#     else:
#         pivot = np.array(pivot_norm)
#
#     if isinstance(axis, torch.Tensor):
#         axis = axis.detach().cpu().numpy()
#     else:
#         axis = np.array(axis)
#
#     render = o3d.visualization.rendering.OffscreenRenderer(1920, 1280)
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = "defaultUnlit"
#
#     # 归一化旋转轴
#     axis = axis / np.linalg.norm(axis)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     mat_point = o3d.visualization.rendering.MaterialRecord()
#     mat_point.shader = "defaultUnlit"  # 不受光照影响
#     mat_point.base_color = [0.8, 0.8, 0.8, 1.0]
#     render.scene.add_geometry("pcd", pcd, mat_point)
#
#     # pivot
#     pivot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#     pivot_sphere.translate(pivot)
#     pivot_sphere.compute_vertex_normals()
#     mat_pivot = o3d.visualization.rendering.MaterialRecord()
#     mat_pivot.shader = "defaultUnlit"
#     mat_pivot.base_color = [0, 0, 1, 1]
#     render.scene.add_geometry("pivot", pivot_sphere, mat_pivot)
#
#     # ==============================
#     # 旋转轴
#     # ==============================
#     # axis_len = 3
#     # axis_end = pivot + axis * axis_len
#     # axis_line = o3d.geometry.LineSet(
#     #     points=o3d.utility.Vector3dVector(np.vstack([pivot, axis_end])),
#     #     lines=o3d.utility.Vector2iVector([[0, 1]])
#     # )
#     # axis_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
#     def create_axis_mesh(pivot, axis, length=1.5, radius=0.015, arrow_len=0.1):
#         """
#         用圆柱 + 锥体表示旋转轴
#         """
#         axis = axis / np.linalg.norm(axis)
#         # 1. 圆柱体
#         cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length - arrow_len)
#         cyl.paint_uniform_color([1, 0, 0])
#
#         # 调整方向
#         from scipy.spatial.transform import Rotation as R
#         z_axis = np.array([0, 0, 1])
#         rot_vec = np.cross(z_axis, axis)
#         if np.linalg.norm(rot_vec) < 1e-6:
#             rot_mat = np.eye(3)
#         else:
#             angle = np.arccos(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
#             rot_mat = R.from_rotvec(rot_vec / np.linalg.norm(rot_vec) * angle).as_matrix()
#         cyl.rotate(rot_mat, center=np.zeros(3))
#         cyl.translate(pivot + axis * (arrow_len / 2))
#
#         # 2. 箭头锥体
#         cone = o3d.geometry.TriangleMesh.create_cone(radius=radius * 2, height=arrow_len)
#         cone.paint_uniform_color([1, 0, 0])
#         cone.rotate(rot_mat, center=np.zeros(3))
#         cone.translate(pivot + axis * (length - arrow_len / 2))
#
#         return [cyl, cone]
#     axis_mesh_list = create_axis_mesh(pivot, axis, length=1.5, radius=0.02, arrow_len=0.15)
#
#     # ==============================
#     # 使用离屏渲染器
#     # ==============================
#     # OffscreenRenderer
#
#     #render.scene.add_geometry("axis", axis_line, mat)
#     for m in axis_mesh_list:
#         render.scene.add_geometry("axis", m, mat)
#
#
#     # 相机自动设置
#     center = xyz.mean(axis=0)
#
#     eye = center + np.array([1.5, 1.5, 1.5])  # 拉远相机位置
#     up = np.array([0, 0, 1])
#     render.setup_camera(60.0, center, eye, up)
#
#     # 渲染并保存
#     img = render.render_to_image()
#     o3d.io.write_image(filename, img)
#     print(f"✅ 图片已保存到 {os.path.abspath(filename)}")



import numpy as np
import os


def visualize_axis_pointcloud(
        xyz_norm,
        pivot,
        axis,
        file_name="pointcloud_axis.obj",
        axis_len_ratio=0.5,
        axis_points=30,
        pivot_points=1,
        colors=None
):
    """
    使用点表示点云 + 旋转轴 + pivot，并保存为 OBJ，带颜色

    Parameters:
    - xyz_norm: (N,3) numpy 或 torch tensor，点云
    - pivot: (3,) numpy 或 torch tensor，旋转中心
    - axis: (3,) numpy 或 torch tensor，单位向量
    - axis_len_ratio: 旋转轴长度占点云对角线比例
    - axis_points: 旋转轴上点数量
    - pivot_points: pivot 点数量
    - colors: dict, 可选，指定颜色 { 'pointcloud':[r,g,b], 'axis':[r,g,b], 'pivot':[r,g,b] }
    """

    # 默认颜色
    if colors is None:
        colors = {
            'pointcloud': [0.8, 0.8, 0.8],
            'axis': [1.0, 0.0, 0.0],
            'pivot': [0.0, 0.0, 1.0]
        }

    # 转 numpy
    if isinstance(xyz_norm, torch.Tensor):
        xyz = xyz_norm.detach().cpu().numpy()
    else:
        xyz = np.asarray(xyz_norm)
    if isinstance(pivot, torch.Tensor):
        pivot = pivot.detach().cpu().numpy()
    else:
        pivot = np.asarray(pivot)
    if isinstance(axis, torch.Tensor):
        axis = axis.detach().cpu().numpy()
    else:
        axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    diag_len = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))

    # 生成旋转轴上的点
    axis_len = diag_len * axis_len_ratio
    axis_line_points = np.array([pivot + axis * axis_len * t for t in np.linspace(0, 1, axis_points)])

    # pivot 点
    pivot_points_array = np.tile(pivot.reshape(1, 3), (pivot_points, 1))

    # 写 OBJ
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        # 点云
        for v in xyz:
            f.write(
                f"v {v[0]} {v[1]} {v[2]} {colors['pointcloud'][0]} {colors['pointcloud'][1]} {colors['pointcloud'][2]}\n")
        # 轴
        for v in axis_line_points:
            f.write(f"v {v[0]} {v[1]} {v[2]} {colors['axis'][0]} {colors['axis'][1]} {colors['axis'][2]}\n")
        # pivot
        for v in pivot_points_array:
            f.write(f"v {v[0]} {v[1]} {v[2]} {colors['pivot'][0]} {colors['pivot'][1]} {colors['pivot'][2]}\n")

    total_points = xyz.shape[0] + axis_line_points.shape[0] + pivot_points_array.shape[0]
    print(f"✅ 已生成点云 OBJ: {file_name}, 总点数: {total_points}")
    return file_name


def save_pointcloud(xyz, filename):
    """保存为 .ply 点云文件"""
    xyz = xyz.detach().cpu().numpy()
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
