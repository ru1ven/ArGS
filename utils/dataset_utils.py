import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from common import data_utils
from utils.graphics_utils import BasicPointCloud
from plyfile import PlyData, PlyElement

import cv2

def downsample(fnames, split):
    if "small" not in split and "mini" not in split and "tiny" not in split:
        return fnames
    import random

    random.seed(1)
    assert (
        random.randint(0, 100) == 17
    ), "Same seed but different results; Subsampling might be different."

    num_samples = len(fnames)
    curr_keys = random.sample(fnames, num_samples)
    return curr_keys

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def transform_2d_for_speedup(
    speedup,
    is_egocam,
    _joints2d_r,
    _joints2d_l,
    _kp2d_b,
    _kp2d_t,
    _bbox2d_b,
    _bbox2d_t,
    _bbox_crop,
    ego_image_scale,
):
    joints2d_r = np.copy(_joints2d_r)
    joints2d_l = np.copy(_joints2d_l)
    kp2d_b = np.copy(_kp2d_b)
    kp2d_t = np.copy(_kp2d_t)
    bbox2d_b = np.copy(_bbox2d_b)
    bbox2d_t = np.copy(_bbox2d_t)
    bbox_crop = np.array(_bbox_crop)
    # bbox is normalized in scale

    if speedup:
        if is_egocam:
            joints2d_r[:, :2] *= ego_image_scale
            joints2d_l[:, :2] *= ego_image_scale
            kp2d_b[:, :2] *= ego_image_scale
            kp2d_t[:, :2] *= ego_image_scale
            bbox2d_b[:, :2] *= ego_image_scale
            bbox2d_t[:, :2] *= ego_image_scale

            bbox_crop = [num * ego_image_scale for num in bbox_crop]
        else:
            # change to new coord system
            joints2d_r = data_utils.transform_kp2d(joints2d_r, bbox_crop)
            joints2d_l = data_utils.transform_kp2d(joints2d_l, bbox_crop)
            kp2d_b = data_utils.transform_kp2d(kp2d_b, bbox_crop)
            kp2d_t = data_utils.transform_kp2d(kp2d_t, bbox_crop)
            bbox2d_b = data_utils.transform_kp2d(bbox2d_b, bbox_crop)
            bbox2d_t = data_utils.transform_kp2d(bbox2d_t, bbox_crop)

            bbox_crop[0] = 500
            bbox_crop[1] = 500
            bbox_crop[2] = 1000 / (1.5 * 200)

    # bbox is normalized in scale
    return (
        joints2d_r,
        joints2d_l,
        kp2d_b,
        kp2d_t,
        bbox2d_b,
        bbox2d_t,
        bbox_crop,
    )


def transform_bbox_for_speedup(
    speedup,
    is_egocam,
    _bbox_crop,
    ego_image_scale,
):
    bbox_crop = np.array(_bbox_crop)
    # bbox is normalized in scale

    if speedup:
        if is_egocam:
            bbox_crop = [num * ego_image_scale for num in bbox_crop]
        else:
            # change to new coord system
            bbox_crop[0] = 500
            bbox_crop[1] = 500
            bbox_crop[2] = 1000 / (1.5 * 200)

    # bbox is normalized in scale
    return bbox_crop


def update_K_after_bbox_crop_resize(K, bbox, cap_dim=1000, crop_ratio=1.5):
    """
    根据 bbox (cx, cy, scale) 和 crop + resize 操作，更新内参 K。

    Args:
        K: 原始相机内参，3x3 numpy array
        bbox: (cx, cy, scale)，其中 scale 是基于 200 得到的 bbox 大小因子
        cap_dim: 最终图像尺寸（裁剪后 resize 到的尺寸）
        crop_ratio: 裁剪区域相对于 bbox 的放缩比例，默认是 1.5，表示上下左右各加 25%

    Returns:
        K_new: 裁剪 + 缩放后的新内参
    """
    cx, cy, scale = bbox
    s = 200.0 * scale
    crop_size = crop_ratio * s  # 原图中裁剪的实际像素大小

    # Step 1: 计算 crop 左上角位置
    crop_x0 = cx - crop_size / 2.0
    crop_y0 = cy - crop_size / 2.0

    # Step 2: 缩放比例（将 crop_size 映射为 cap_dim）
    resize_scale = cap_dim / crop_size

    # Step 3: 更新内参
    fx_new = K[0, 0] * resize_scale
    fy_new = K[1, 1] * resize_scale
    cx_new = (K[0, 2] - crop_x0) * resize_scale
    cy_new = (K[1, 2] - crop_y0) * resize_scale

    # K_new = np.array([
    #     [fx_new, 0.0, cx_new],
    #     [0.0, fy_new, cy_new],
    #     [0.0, 0.0, 1.0]
    # ], dtype=np.float32)
    K_flip = np.array([
        [fx_new, 0, cap_dim-cx_new],  # 水平翻转
        [0, fy_new, cap_dim-cy_new],  # 垂直翻转
        [0, 0, 1]
    ])

    return K_flip

from scipy.spatial.transform import Rotation as R

def apply_w2c_pose_numpy(rot, trans, w2c):
    """
    将物体的旋转和平移从世界坐标转换到相机坐标下。

    :param rot: (3,) Rodrigues 向量（世界系）
    :param trans: (3,) 平移向量（世界系）
    :param w2c: (4, 4) 齐次变换矩阵，表示 world2cam
    :return: rot_cam (3,), trans_cam (3,)
    """

    # 1. 世界系旋转向量 -> 旋转矩阵
    R_obj = R.from_rotvec(rot).as_matrix()  # (3,3)

    # 2. 相机变换矩阵分解
    R_wc = w2c[:3, :3]  # (3,3)
    T_wc = w2c[:3, 3]  # (3,)

    # 3. 旋转变换：物体旋转变换到相机系
    R_obj_cam = R_wc @ R_obj

    # 4. 平移变换
    trans_cam = R_wc @ trans + T_wc  # (3,)

    # 5. 转换回 Rodrigues 向量
    rot_cam = R.from_matrix(R_obj_cam).as_rotvec()

    return rot_cam, trans_cam


def apply_w2c_pose_with_center(rot, trans, center, w2c):
    """
    考虑姿态中心的 rot/trans 坐标变换，适用于 SMPL/SMPL-X/MANO。

    rot, trans: (3,) 世界系
    center: (3,) 姿态中心点坐标，世界系
    w2c: (4,4) 齐次矩阵，world to cam

    返回 rot_cam, trans_cam，供 SMPL-X 使用（相机坐标下）
    """
    # 世界旋转
    R_world = R.from_rotvec(rot).as_matrix()

    # world2cam 旋转和平移
    R_wc = w2c[:3, :3]
    T_wc = w2c[:3, 3]

    # 新旋转
    R_cam = R_wc @ R_world

    # 新平移，参考姿态中心点补偿
    trans_world = R_world @ center + trans  # rot后center的位置
    trans_cam = R_wc @ trans_world + T_wc - R_cam @ center

    # rot back to axis-angle
    rot_cam = R.from_matrix(R_cam).as_rotvec()

    return rot_cam, trans_cam


def pad_jts2d(jts):
    num_jts = jts.shape[0]
    jts_pad = np.ones((num_jts, 3))
    jts_pad[:, :2] = jts
    return jts_pad

def get_valid(data_2d, data_cam, vidx, view_idx, imgname):
    assert (
        vidx < data_2d["joints.right"].shape[0]
    ), "The requested vidx does not exist in annotation"
    is_valid = data_cam["is_valid"][vidx, view_idx]
    right_valid = data_cam["right_valid"][vidx, view_idx]
    left_valid = data_cam["left_valid"][vidx, view_idx]
    return vidx, is_valid, right_valid, left_valid


# add ZJUMoCAP dataloader
def get_02v_bone_transforms(Jtr,):
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)