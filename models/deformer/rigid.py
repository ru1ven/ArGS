import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d.ops as ops
import trimesh
import igl

from common.vis_utils import save_bone_contributions_with_joints
from utils.general_utils import build_rotation, rotation_matrix_to_quaternion, visualize_axis_pointcloud
from models.network_utils import get_skinning_mlp, HashGrid, VanillaCondMLP
from scipy.spatial.transform import Rotation as R

from utils.graphics_utils import axis_angle_to_matrix
from utils.loss_utils import gumbel_sigmoid, anti_penetration_loss, build_occupancy_grid, occupancy_loss
from utils.network_utils import Pointnet2_Ssg, PointNet2


class RigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera):
        raise NotImplementedError

    def regularization(self):
        return NotImplementedError

class Identity(RigidDeform):
    """ Identity mapping for single frame reconstruction """
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera):
        return gaussians

    def regularization(self):
        return {}

class SMPLNN(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_verts = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights = torch.from_numpy(metadata["skinning_weights"]).float().cuda()

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def forward(self, gaussians, iteration, camera):
        bone_transforms = camera.bone_transforms

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        pts_W = self.query_weights(xyz)
        T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_pts, 4, 4).float()

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self):
        return {}

def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return F.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_point, n_dim = x.shape

    prob_all = torch.ones(n_point, 24, device=x.device)
    # softmax_x = F.softmax(x, dim=-1)
    sigmoid_x = sigmoid(x).float()

    prob_all[:, [1, 2, 3]] = sigmoid_x[:, [0]] * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = 1 - sigmoid_x[:, [0]]

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid_x[:, [4, 5, 6]])
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid_x[:, [4, 5, 6]])

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid_x[:, [7, 8, 9]])
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid_x[:, [7, 8, 9]])

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid_x[:, [10, 11]])
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid_x[:, [10, 11]])

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid_x[:, [24]] * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid_x[:, [24]])

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid_x[:, [15]])
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid_x[:, [15]])

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid_x[:, [16, 17]])
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid_x[:, [16, 17]])

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid_x[:, [18, 19]])
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid_x[:, [18, 19]])

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid_x[:, [20, 21]])
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid_x[:, [20, 21]])

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid_x[:, [22, 23]])
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid_x[:, [22, 23]])

    # prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

class SkinningField(RigidDeform):
    def __init__(self, cfg, metadata, hand_side):
        super().__init__(cfg)
        self.smpl_verts = metadata["smpl_verts"]
        self.skinning_weights = metadata["skinning_weights"]
        self.aabb = metadata["aabb"]
        self.faces = metadata['faces']
        self.cano_mesh = metadata["cano_mesh"]
        self.hand_side = hand_side

        self.distill = cfg.distill
        d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
        self.resolution = (d, h, w)
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()

        self.lbs_network = get_skinning_mlp(3, cfg.d_out, cfg.skinning_network)


    def precompute(self, pose_model, recompute_skinning=True):
        if recompute_skinning or not hasattr(self, "lbs_voxel_final"):
            d, h, w = self.resolution

            lbs_voxel_final = self.lbs_network(self.grid[0]).float()
            lbs_voxel_final = self.cfg.soft_blend * lbs_voxel_final

            lbs_voxel_final = self.softmax(lbs_voxel_final)

            self.lbs_voxel_final = lbs_voxel_final.permute(1, 0).reshape(1, 24, d, h, w)

    def get_forward_transform(self, xyz, tfs, pose_model):
        if self.distill:
            self.precompute(pose_model,recompute_skinning=self.training)
            fwd_grid = torch.einsum("bcdhw,bcxy->bxydhw", self.lbs_voxel_final, tfs[None])
            fwd_grid = fwd_grid.reshape(1, -1, *self.resolution)
            T_fwd = F.grid_sample(fwd_grid, xyz.reshape(1, 1, 1, -1, 3), padding_mode='border')
            T_fwd = T_fwd.reshape(4, 4, -1).permute(2, 0, 1)
        else:
            pts_W = self.lbs_network(xyz)
            pts_W = self.softmax(pts_W)
            T_fwd = torch.matmul(pts_W, tfs.view(-1, 16)).view(-1, 4, 4).float()
        return T_fwd

    def sample_skinning_loss(self):
        points_skinning, face_idx = self.cano_mesh.sample(self.cfg.n_reg_pts, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            self.smpl_verts[self.faces[face_idx, 0], :],
            self.smpl_verts[self.faces[face_idx, 1], :],
            self.smpl_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)

        points_skinning = torch.from_numpy(points_skinning).cuda().float()
        pts_W = torch.from_numpy(pts_W).cuda().float()
        return points_skinning, pts_W

    def softmax(self, logit):
        if logit.shape[-1] == 17:
            w = hierarchical_softmax(logit)
        elif logit.shape[-1] == 16:
            w = F.softmax(logit, dim=-1)
        else:
            raise ValueError
        return w

    def get_skinning_loss(self, pose_model):
        pts_skinning, sampled_weights = self.sample_skinning_loss()
        pts_skinning = self.aabb.normalize(pts_skinning, sym=True)

        if self.distill:
            pred_weights = F.grid_sample(self.lbs_voxel_final, pts_skinning.reshape(1, 1, 1, -1, 3), padding_mode='border')
            pred_weights = pred_weights.reshape(16, -1).permute(1, 0)
        else:
            pred_weights = self.lbs_network(pts_skinning)
            pred_weights = self.softmax(pred_weights)
        skinning_loss = torch.nn.functional.mse_loss(
            pred_weights, sampled_weights, reduction='none').sum(-1).mean()
        # breakpoint()

        return skinning_loss


    def forward(self, gaussians, iteration, camera, pose_model):
        if self.hand_side == 'left':
            tfs = camera.bone_transforms_l
        elif self.hand_side == 'right':
            tfs = camera.bone_transforms_r

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        T_fwd = self.get_forward_transform(xyz_norm, tfs, pose_model)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        #deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self, pose_model):
        loss_skinning = self.get_skinning_loss(pose_model)
        return {
            'loss_skinning': loss_skinning
        }

class ObjDeform(RigidDeform):
    def __init__(self, cfg, metadata, hand_side):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera, pose_model, delay=False):
        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        rot = camera.obj_rots

        trans = camera.obj_trans

        T_fwd = torch.eye(4).to(rot.device)
        T_fwd[:3,:3] = rot.float().to(rot.device)

        T_fwd[:3, 3] = trans

        T_fwd = T_fwd.repeat(n_pts,1,1)
        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz,homo_coord],dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        #deformed_gaussians._rotation = rotation_bar
        #deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians


class ObjBoneDeform(RigidDeform):
    def __init__(self, cfg, metadata, hand_side):
        super().__init__(cfg)
        self.cfg = cfg
        self.obj_rigid_deform = ObjDeform(cfg, metadata, hand_side)
        self.num_parts = 2
        self.aabb = metadata[list(metadata.keys())[0]]["obj_aabb"]
        self.assignment_mlp = VanillaCondMLP(48,0,1,cfg.skinning_network)

        self.hashgrid = HashGrid(cfg.hashgrid)

        d_cond = 0
        self.latent_dim = 64
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata[list(metadata.keys())[0]]['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)
        self.bone_mlp = VanillaCondMLP(256, d_cond, 16, cfg.mlp)
        self.init_weights()
        self.delta_history = {}

    def init_weights(self):
        # self.assignment_mlp[-1].bias.data[:] = 0  # 全部初始化为 -2
        # self.assignment_mlp[-1].bias.data[0] = 0

        last_layer = getattr(self.bone_mlp, f'lin{self.bone_mlp.num_layers - 2}')
        last_layer.weight.data.zero_()
        last_layer.bias.data.copy_(torch.eye(4).view(-1))


    def forward(self, gaussians, iteration, camera, pose_model, delay=False, save_dir=None):
        # if self.training and iteration <= self.cfg.get("bone_delay", 0):
        #     return self.obj_rigid_deform(gaussians, iteration, camera, pose_model)
        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        xyz_norm = self.aabb.normalize(xyz, sym=True)

        #rela_trans_weight = self.assignment_mlp(xyz_norm) # N*1

        # predict bone tfs
        hash_feature = self.hashgrid(xyz_norm).float()

        #pcl_feature = self.pointnet(xyz_norm.unsqueeze(0).permute(0, 2, 1)).squeeze(0)
        pc_feature = gaussians.non_rigid_feature.mean(dim=0).view(1,256)

        frame_idx = camera.frame_id
        if frame_idx not in self.frame_dict:
            latent_idx = len(self.frame_dict) - 1
        else:
            latent_idx = self.frame_dict[frame_idx]
        latent_idx = torch.Tensor([latent_idx]).long().to(pc_feature.device)
        latent_code = self.latent(latent_idx)
        latent_code = latent_code.expand(pc_feature.shape[0], -1)  # 扩展到 [B, M, C]

        tfs_rela = self.bone_mlp(pc_feature, cond=latent_code)

        if delay:
            # pts_adj 预测每点对 delta_weight 的微调
            pts_adj = self.assignment_mlp(hash_feature)  # 输出 N x num_bone
            #pts_adj = torch.tanh(pts_adj)  # 限制调整范围 [-1,1]

            # 融合初始 delta 先验
            # pts_W = self.per_delta[frame_idx] * (1 + pts_adj)
            # pts_W = pts_W / pts_W.sum(dim=-1, keepdim=True)  # 归一化

            # 将 delta 转换成 logit 形式作为初始偏置
            per_delta = self.per_delta.cuda().unsqueeze(1)

            delta_logits = torch.log(per_delta + 1e-8) - torch.log(
                1 - per_delta + 1e-8)  # logit(delta)，0→-inf，1→+inf

            # 加上可学习偏移量
            logits = delta_logits + pts_adj
            #print(logits.shape)
            # sigmoid 生成最终权重
            pts_W = gumbel_sigmoid(logits)  # [N, num_bone]，范围在0~1


            # pts_W = F.softmax(self.assignment_mlp(hash_feature), dim=-1)
            T_fwd = torch.matmul(pts_W, tfs_rela.view(-1, 16)).view(-1, 4, 4).float()

            deformed_gaussians = gaussians.clone()
            deformed_gaussians.set_fwd_transform(T_fwd)

            homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
            x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
            x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
            deformed_gaussians._xyz = x_bar

            rotation_hat = build_rotation(gaussians._rotation)
            rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
            setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
            #deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

            # deformed_gaussians._rotation = rotation_bar
            # debug bone & skin weight
            if not self.training and iteration % 200 == 0:
                save_bone_contributions_with_joints(
                    xyz=x_bar,  # [N, 3]
                    pts_W=pts_W,  # [N, B]
                    joint_pos=tfs_rela.view(-1, 4, 4)[:, :3, 3],  # [B, 3]
                    out_dir="/mnt/sda2/lxy/ARGS_results/bones_debug_refine/iter_{}".format(iteration)
                )

            return self.obj_rigid_deform(deformed_gaussians, iteration, camera, pose_model), None
        # rigid

        ##if iteration <= 15000:
        deformed_gaussians = gaussians.clone()
        return self.obj_rigid_deform(deformed_gaussians, iteration, camera, pose_model), None
        #else:
        # bone
        # # === 时序一致性正则 ===
        frame_idx = camera.frame_id
        #
        # self.delta_history[frame_idx] = (tfs_rela.detach())
        # if frame_idx > 0 and (frame_idx - 1) in self.delta_history:
        #
        #     tfs_rela_prev = self.delta_history[frame_idx - 1]
        #
        #     loss_temporal_relstr = torch.norm(tfs_rela - tfs_rela_prev, p=2, dim=1).mean()
        #
        #
        # else:
        #     # 第一帧没有上一帧，不计算
        #     loss_temporal_relstr = 0.
        #
        # loss_reg={
        #     'tempo_bone': loss_temporal_relstr,
        # }

        # predict skin
        pts_W = F.softmax(self.assignment_mlp(hash_feature), dim=-1)


        T_fwd = torch.matmul(pts_W, tfs_rela.view(-1, 16)).view(-1, 4, 4).float()

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd)

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz,homo_coord],dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        self.delta_history[frame_idx] = (x_bar.detach() - xyz.detach()).detach().cpu()

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        #deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)
        #deformed_gaussians._rotation = rotation_bar

        # debug bone & skin weight
        if not self.training and iteration % 200 == 0:

            save_bone_contributions_with_joints(
                xyz=x_bar,  # [N, 3]
                pts_W=pts_W,  # [N, B]
                joint_pos=tfs_rela.view(-1, 4, 4)[:, :3, 3],  # [B, 3]
                out_dir="/mnt/sda2/lxy/ARGS_results/bones_debug/iter_{}".format(iteration)
            )
        return self.obj_rigid_deform(deformed_gaussians, iteration, camera, pose_model), None
        #return deformed_gaussians


class ObjRevoluteDeform(RigidDeform):
    def __init__(self, cfg, metadata, hand_side):
        super().__init__(cfg)
        self.cfg = cfg
        self.obj_rigid_deform = ObjDeform(cfg, metadata, hand_side)
        self.num_parts = 2
        self.aabb = metadata[list(metadata.keys())[0]]["obj_aabb"]

        self.hashgrid = HashGrid(cfg.hashgrid)

        d_cond = 0
        self.latent_dim = 64
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata[list(metadata.keys())[0]]['frame_dict']
            self.latent_angle = nn.Embedding(len(self.frame_dict), self.latent_dim)


        self.pivot_mlp = VanillaCondMLP(48, 0, 1, cfg.skinning_network)
        self.axis_mlp = VanillaCondMLP(48, 0, 3, cfg.skinning_network)
        #self.angle_mlp = VanillaCondMLP(d_cond, 0, 2, cfg.skinning_network)
        self.angle_mlp = VanillaCondMLP(256, d_cond, 2, cfg.mlp)

        self.init_weights()
        self.delta_history = {}
        self.angle_history = {}


    def init_weights(self):

        last_layer_name = "lin" + str(self.angle_mlp.num_layers - 2)
        last_layer = getattr(self.angle_mlp, last_layer_name)
        with torch.no_grad():
            last_layer.bias.copy_(torch.tensor([0.0, 1.0], dtype=last_layer.bias.dtype))
            last_layer.weight.zero_()  # 可选

    def forward(self, gaussians, canonical_gs, iteration, camera, pose_model, rigid_iter, delay=False, save_dir=None):
        if delay:
            return self.obj_rigid_deform(gaussians, iteration, camera, pose_model), {}

            # teacher（原始gaussians，只做forward，梯度不回传）
            #with torch.no_grad():
            gaussians_student = canonical_gs.clone()

            teacher_xyz = gaussians._xyz.detach().clone()
            teacher_rotation = gaussians._rotation.detach().clone()
            teacher_scaling = gaussians._scaling.detach().clone()
            teacher_dynamic = gaussians.get_raw_dynamic.detach()

            student_xyz = canonical_gs._xyz.detach().clone()
            student_rotation = canonical_gs._rotation.detach().clone()
            student_scaling = canonical_gs._scaling.detach().clone()

            xyz_norm = self.aabb.normalize(student_xyz, sym=True)
            hash_feature = self.hashgrid(xyz_norm).float()

            # 将 delta 转换成 logit 形式作为初始偏置
            is_movable = gumbel_sigmoid(teacher_dynamic, hard=True)

            # 学习归一化的pivot
            pivot_scores = self.pivot_mlp(hash_feature)
            weights = F.softmax(pivot_scores.squeeze(-1), dim=0)  # (N,)
            pivot = torch.sum(weights[:, None] * student_xyz, dim=0)  # (3,)
            # pivot_norm = torch.sum(weights[:, None] * xyz_norm, dim=0)  # (3,)
            # pivot = self.aabb.unnormalize(pivot_norm, sym=True)

            # pivot_norm 激活为[-1, 1]
            # pivot_norm = torch.tanh(pivot_norm)
            # pivot = self.aabb.unnormalize(pivot_norm, sym=True)

            axis = self.axis_mlp(hash_feature.mean(dim=0).unsqueeze(0)).squeeze(0)
            axis = axis / axis.norm(dim=-1, keepdim=True)

            # angle = self.angle_mlp(pc_feature, cond=latent_code).squeeze(0)
            # angle = torch.tanh(angle)*math.pi

            pc_feature = gaussians.non_rigid_feature.detach().clone().mean(dim=0).view(1, 256)

            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pc_feature.device)
            latent_code = self.latent_angle(latent_idx)
            latent_code = latent_code.expand(pc_feature.shape[0], -1)  # 扩展到 [B, M, C]

            angle_2d = self.angle_mlp(pc_feature, cond=latent_code).squeeze(0)
            angle_2d = angle_2d / (angle_2d.norm() + 1e-8)  # 归一化
            angle = torch.atan2(angle_2d[0], angle_2d[1])  # 转弧度


            if iteration % 100 == 0:
                print(angle)
                print(axis)
                print(pivot)

            R = axis_angle_to_matrix(axis, angle)

            xyz_bar = (R @ (student_xyz - pivot).T).T + pivot
            rotation_hat = build_rotation(student_rotation)
            rotation_bar = torch.matmul(R, rotation_hat)
            deformed_xyz = (1 - is_movable) * student_xyz+ is_movable * xyz_bar
            basic_rotation_bar = build_rotation(student_rotation.clone())
            dynamic_rotation_bar = (1 - is_movable.view(-1, 1, 1)) * basic_rotation_bar + is_movable.view(-1, 1,
                                                                                                          1) * rotation_bar

            if iteration % 1000 == 0:
                save_pcl_path = os.path.join(save_dir, 'movable', 'iteration_{}'.format(iteration))

                visualize_axis_pointcloud(student_xyz, pivot.squeeze(0), axis,
                                          file_name=os.path.join(save_pcl_path, 'axis_pcl.obj'))
                visualize_axis_pointcloud(teacher_xyz, pivot.squeeze(0),
                                          axis,
                                          file_name=os.path.join(save_pcl_path, 'axis_nr_pcl.obj'))
                visualize_axis_pointcloud(deformed_xyz, pivot.squeeze(0),
                                          axis,
                                          file_name=os.path.join(save_pcl_path, 'axis_deformed_pcl.obj'))

            loss_rigid = {}


            # if iteration > 3000:
            #     loss_rigid = {'revolute_obj': F.smooth_l1_loss(self.aabb.normalize(deformed_xyz, sym=True),
            #                                                        self.aabb.normalize(teacher_xyz, sym=True),
            #                                                        beta=0.05)}

            return self.obj_rigid_deform(gaussians, iteration, camera, pose_model), loss_rigid

        # teacher（原始gaussians，只做forward，梯度不回传）
        #teacher_dynamic = gaussians.get_raw_dynamic.detach()
        teacher_dynamic = gaussians.get_raw_dynamic
        student_xyz = canonical_gs._xyz
        student_rotation = canonical_gs._rotation
        student_scaling = canonical_gs._scaling

        xyz_norm = self.aabb.normalize(student_xyz, sym=True)
        hash_feature = self.hashgrid(xyz_norm).float()

        # 将 delta 转换成 logit 形式作为初始偏置
        is_movable = gumbel_sigmoid(teacher_dynamic, hard=True)

        # 学习归一化的pivot
        pivot_scores = self.pivot_mlp(hash_feature)
        weights = F.softmax(pivot_scores.squeeze(-1), dim=0)  # (N,)
        pivot = torch.sum(weights[:, None] * student_xyz, dim=0)  # (3,)
        #pivot_norm = torch.sum(weights[:, None] * xyz_norm, dim=0)  # (3,)
        #pivot = self.aabb.unnormalize(pivot_norm, sym=True)

        # pivot_norm 激活为[-1, 1]
        # pivot_norm = torch.tanh(pivot_norm)
        # pivot = self.aabb.unnormalize(pivot_norm, sym=True)

        axis = self.axis_mlp(hash_feature.mean(dim=0).unsqueeze(0)).squeeze(0)
        axis = axis / axis.norm(dim=-1, keepdim=True)

        # angle = self.angle_mlp(pc_feature, cond=latent_code).squeeze(0)
        # angle = torch.tanh(angle)*math.pi

        pc_feature = gaussians.non_rigid_feature.mean(dim=0).view(1, 256)
        frame_idx = camera.frame_id
        if frame_idx not in self.frame_dict:
            latent_idx = len(self.frame_dict) - 1
        else:
            latent_idx = self.frame_dict[frame_idx]
        latent_idx = torch.Tensor([latent_idx]).long().to(pc_feature.device)
        latent_code = self.latent_angle(latent_idx)
        latent_code = latent_code.expand(pc_feature.shape[0], -1)  # 扩展到 [B, M, C]

        # angle_2d = self.angle_mlp(latent_code).squeeze(0)
        # loss_norm = 0.1 * (angle_2d.pow(2).sum() - 1.0).pow(2)
        # angle_2d = angle_2d / (angle_2d.norm() + 1e-8)  # 归一化
        # angle = torch.atan2(angle_2d[0], angle_2d[1])  # 转弧度

        #逐帧
        angle_2d = self.angle_mlp(pc_feature, cond=latent_code).squeeze(0)
        loss_norm = 0.1 * (angle_2d.pow(2).sum() - 1.0).pow(2)
        angle_2d = angle_2d / (angle_2d.norm() + 1e-8)  # 归一化
        angle = torch.atan2(angle_2d[0], angle_2d[1])  # 转弧度
        #
        self.angle_history[int(latent_idx)] = angle.detach()
        # velocity loss
        if (int(latent_idx) - 1) in self.angle_history:
            vel = angle - self.angle_history[int(latent_idx) - 1]
            loss_vel = (vel ** 2).mean()

            # acceleration loss
            if (int(latent_idx) - 2) in self.angle_history:
                acc = angle - 2 * self.angle_history[int(latent_idx) - 1] + self.angle_history[int(latent_idx) - 2]
                loss_acc = (acc ** 2).mean()
            else:
                loss_acc = torch.tensor(0., device=angle.device)
        else:

            loss_vel = torch.tensor(0., device=angle.device)
            loss_acc = torch.tensor(0., device=angle.device)

        loss_temporal = loss_vel + 0.1 * loss_acc
        # if iteration >= rigid_iter+2000:
        #loss_norm += loss_temporal


        if iteration % 5 == 0:
            print(angle)
            print(axis)
            print(pivot)
        R = axis_angle_to_matrix(axis, angle)

        xyz_bar = (R @ (student_xyz - pivot).T).T + pivot
        rotation_hat = build_rotation(student_rotation)
        rotation_bar = torch.matmul(R, rotation_hat)
        deformed_xyz = (1 - is_movable) * student_xyz.detach().clone() + is_movable * xyz_bar
        basic_rotation_bar = build_rotation(student_rotation.clone())
        dynamic_rotation_bar = (1 - is_movable.view(-1,1,1)) * basic_rotation_bar + is_movable.view(-1,1,1) * rotation_bar

        if iteration % 1000 == 0:
            save_pcl_path = os.path.join(save_dir, 'movable', 'iteration_{}'.format(iteration))

            visualize_axis_pointcloud(student_xyz, pivot.squeeze(0), axis,
                                      file_name=os.path.join(save_pcl_path, 'axis_pcl.obj'))

            visualize_axis_pointcloud(deformed_xyz, pivot.squeeze(0),
                                      axis,
                                      file_name=os.path.join(save_pcl_path, 'axis_deformed_pcl.obj'))

        #loss_rigid = {}
        loss_rigid = {'revolute_obj': loss_norm}

        canonical_gs._xyz = deformed_xyz
        setattr(canonical_gs, 'rotation_precomp', dynamic_rotation_bar)
        return self.obj_rigid_deform(canonical_gs, iteration, camera, pose_model), loss_rigid



def get_rigid_deform(cfg, metadata, hand_side):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "smpl_nn": SMPLNN,
        "skinning_field": SkinningField,
        "obj_deform": ObjRevoluteDeform,
        #"obj_deform": ObjDeform,
    }
    return model_dict[name](cfg, metadata, hand_side)