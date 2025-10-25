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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix_offaxis

from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
from pytorch3d.renderer import look_at_view_transform


# class Camera:
#     def __init__(self, camera=None, **kwargs):
#         if camera is not None:
#             self.data = camera.data.copy()
#             return
#
#         self.data = kwargs
#         self.data['trans'] = np.array([0.0, 0.0, 0.0])
#         self.data['scale'] = 1.0
#
#         self.data['original_image'] = self.image.clamp(0.0, 1.0).unsqueeze(0).to(self.data_device)
#         self.data['image_width'] = self.original_image.shape[2]
#         self.data['image_height'] = self.original_image.shape[1]
#         self.data['original_mask'] = self.mask.float().unsqueeze(0).to(self.data_device)
#         self.data['obj_image'] = self.obj_image.clamp(0.0, 1.0).unsqueeze(0).to(self.data_device)
#         self.data['obj_mask'] = self.obj_mask.float().unsqueeze(0).to(self.data_device)
#         self.data['full_image'] = self.full_image.clamp(0.0, 1.0).unsqueeze(0).to(self.data_device)
#         self.data['full_mask'] = self.full_mask.float().unsqueeze(0).to(self.data_device)
#
#         self.data['img_ROI'] = self.img_ROI.clamp(0.0, 1.0).unsqueeze(0).to(self.data_device)
#         self.data['trans_img2roi'] = torch.tensor(self.trans_img2roi).unsqueeze(0).to(self.data_device)
#
#         self.data['zfar'] = 100.0
#         self.data['znear'] = 0.01
#
#         self.data['world_view_transform'] = torch.tensor(
#             getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).unsqueeze(0).to(self.data_device)
#         self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
#                                                              fovY=self.FoVy).transpose(0, 1).unsqueeze(0).to(self.data_device)
#         self.data['full_proj_transform'] = (
#             self.world_view_transform.bmm(self.projection_matrix))
#         self.data['camera_center'] = self.world_view_transform.squeeze(0).inverse()[3, :3].unsqueeze(0).to(self.data_device)
#
#         self.data['rots'] = self.rots.unsqueeze(0).to(self.data_device)
#         self.data['Jtrs'] = self.Jtrs.unsqueeze(0).to(self.data_device)
#         self.data['bone_transforms'] = self.bone_transforms.unsqueeze(0).to(self.data_device)
#
#         self.data['obj_rots'] = self.obj_rots.unsqueeze(0).to(self.data_device)
#         self.data['obj_trans'] = self.obj_trans.unsqueeze(0).to(self.data_device)
#
#         self.data['R'] = torch.tensor(self.R).unsqueeze(0).to(self.data_device)
#         self.data['T'] = torch.tensor(self.T).unsqueeze(0).to(self.data_device)
#         self.data['K'] = torch.tensor(self.K).unsqueeze(0).to(self.data_device)
#
#         self.data['obj_rots_gt'] = self.obj_rots_gt.unsqueeze(0).to(self.data_device)
#         self.data['obj_trans_gt'] = self.obj_trans_gt.unsqueeze(0).to(self.data_device)
#         self.data['hand_param'] = self.hand_param.unsqueeze(0).to(self.data_device)
#         self.data['hand_param_gt'] = self.hand_param_gt.unsqueeze(0).to(self.data_device)
#         self.data['joints_gt'] = self.joints_gt.unsqueeze(0).to(self.data_device)
#         self.data['hand_root'] = self.hand_root.unsqueeze(0).to(self.data_device)
#
#     def __getattr__(self, item):
#         return self.data[item]
#
#     def update(self, **kwargs):
#         self.data.update(kwargs)
#
#     def copy(self):
#         new_cam = Camera(camera=self)
#         return new_cam
#
#     def merge(self, cam):
#         self.data['frame_id'] = cam.frame_id
#         self.data['rots'] = cam.rots.detach()
#         self.data['Jtrs'] = cam.Jtrs.detach()
#         self.data['bone_transforms'] = cam.bone_transforms.detach()
class Camera:
    def __init__(self, camera=None, **kwargs):
        if camera is not None:
            self.data = camera.data.copy()
            return

        self.data = kwargs
        self.data['trans'] = np.array([0.0, 0.0, 0.0])
        self.data['scale'] = 1.0

        self.data['original_image'] = self.image.clamp(0.0, 1.0).to(self.data_device)
        self.data['image_width'] = self.original_image.shape[2]
        self.data['image_height'] = self.original_image.shape[1]
        self.data['original_mask'] = self.mask.float().to(self.data_device)
        self.data['obj_image'] = self.obj_image.clamp(0.0, 1.0).to(self.data_device)
        self.data['obj_mask'] = self.obj_mask.float().to(self.data_device)
        self.data['mask_static'] = self.mask_static.float().to(self.data_device)
        self.data['mask_dynamic'] = self.mask_dynamic.float().to(self.data_device)
        self.data['full_image'] = self.full_image.clamp(0.0, 1.0).to(self.data_device)
        # self.data['full_image_ori'] = self.full_image_ori.clamp(0.0, 1.0).to(self.data_device)
        self.data['full_mask'] = self.full_mask.float().to(self.data_device)
        self.data['bbox'] = self.bbox.float().float().to(self.data_device)

        self.data['img_ROI'] = self.img_ROI.clamp(0.0, 1.0).to(self.data_device)
        # self.data['ori_img'] = self.ori_img.clamp(0.0, 1.0).to(self.data_device)
        self.data['trans_img2roi'] = torch.tensor(self.trans_img2roi).to(self.data_device)

        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01

        self.data['world_view_transform'] = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()

        #print(self.data['world_view_transform'])
        if  self.offaxis:
            self.data['projection_matrix'] = getProjectionMatrix_offaxis(self.znear, self.zfar, self.K, self.original_image.shape[2],
                                                                         self.original_image.shape[1]).transpose(0, 1).cuda()
        else:
            self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                                 fovY=self.FoVy).transpose(0, 1).cuda()
        self.data['full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.data['camera_center'] = self.world_view_transform.inverse()[:3, 3]

        self.data['rots_r'] = self.rots_r.to(self.data_device)
        self.data['Jtrs_r'] = self.Jtrs_r.to(self.data_device)
        self.data['rots_l'] = self.rots_l.to(self.data_device)
        self.data['Jtrs_l'] = self.Jtrs_l.to(self.data_device)
        self.data['Jtrs_r_3d'] = self.Jtrs_r_3d.to(self.data_device)
        self.data['Jtrs_l_3d'] = self.Jtrs_l_3d.to(self.data_device)
        self.data['bone_transforms_r'] = self.bone_transforms_r.to(self.data_device)
        self.data['bone_transforms_l'] = self.bone_transforms_l.to(self.data_device)

        self.data['obj_rots'] = self.obj_rots.to(self.data_device)
        self.data['obj_trans'] = self.obj_trans.to(self.data_device)

        self.data['R'] = torch.tensor(self.R).to(self.data_device)
        self.data['T'] = torch.tensor(self.T).to(self.data_device)
        self.data['K'] = torch.tensor(self.K).to(self.data_device)

        # self.data['obj_rots_gt'] = self.obj_rots_gt.to(self.data_device)
        # self.data['obj_trans_gt'] = self.obj_trans_gt.to(self.data_device)
        # self.data['hand_param'] = self.hand_param.to(self.data_device)
        # self.data['hand_param_gt'] = self.hand_param_gt.to(self.data_device)
        # self.data['joints_gt'] = self.joints_gt.to(self.data_device)
        # self.data['hand_root'] = self.hand_root.to(self.data_device)

    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = Camera(camera=self)
        return new_cam

    def merge(self, cam):
        self.data['frame_id'] = cam.frame_id
        self.data['rots'] = cam.rots.detach()
        self.data['Jtrs'] = cam.Jtrs.detach()
        self.data['bone_transforms'] = cam.bone_transforms.detach()



class QueryCamera:
    def __init__(self, camera=None, **kwargs):
        if camera is not None:
            self.data = camera.data.copy()
            return

        self.data = kwargs
        self.data['trans'] = np.array([0.0, 0.0, 0.0])
        self.data['scale'] = 1.0
        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01
        self.data['world_view_transform'] = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                             fovY=self.FoVy).transpose(0, 1).cuda()
        self.data['full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).cuda()
        self.data['camera_center'] = self.world_view_transform.inverse()[3, :3].cuda()
        self.data['R'] = torch.tensor(self.R).cuda()
        self.data['T'] = torch.tensor(self.T).cuda()
        self.data['K'] = torch.tensor(self.K).cuda()


    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = QueryCamera(camera=self)
        return new_cam


class QueryCamerasLoader:
    def __init__(self, coord_min, coord_max, cam_num=64):
        self.cam_num = cam_num
        self._cam = []
        self.K = np.array([[8000, 0, 500], [0, 8000, 500], [0, 0, 1]])
        self.h, self.w = 1000, 1000
        focal_length_x = self.K[0][0]
        focal_length_y = self.K[1][1]
        self.FoVy = focal2fov(focal_length_y, self.h)
        self.FoVx = focal2fov(focal_length_x, self.w)
        extrinsics = self.get_camera_extrinsics(coord_min, coord_max, cam_num)
        for R, T in extrinsics:
            camera = QueryCamera(K=self.K, R=np.array(R), T=np.array(T), focal_x=focal_length_x, focal_y=focal_length_y,
                                 FoVx=self.FoVx, FoVy=self.FoVy,image_height=self.h, image_width=self.w)
            self._cam.append(camera)

    @property
    def get_cam(self):
        return self._cam

    def generate_fibonacci_sphere(self, n_points, radius, center_tensor):
        """生成均匀分布在球面上的点"""
        device = center_tensor.device
        # 黄金比
        phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2  # golden ratio
        # 使用torch张量生成点
        z = torch.linspace(-1, 1, n_points, device=device)  # 映射到 [-1, 1]
        theta = 2 * torch.pi * torch.arange(n_points, device=device) / phi
        # 计算x, y, z
        x = torch.sqrt(1 - z ** 2) * torch.cos(theta)
        y = torch.sqrt(1 - z ** 2) * torch.sin(theta)
        # 将坐标移到指定的中心并缩放到给定的半径
        x = x * radius + center_tensor[0]
        y = y * radius + center_tensor[1]
        z = z * radius + center_tensor[2]
        # 将x, y, z 合并为一个点，并将所有点添加到列表中
        points = torch.stack((x, y, z), dim=-1)
        return points

    def compute_camera_extrinsics(self, camera_pos, bbox_center):
        """计算每个相机的外参，包括旋转矩阵和平移向量，支持CUDA加速，但输出为list和numpy格式"""

        rotation_matrix, translation_vector = look_at_view_transform(eye=camera_pos.unsqueeze(0), at=bbox_center.unsqueeze(0))

        # 转换为numpy并返回
        rotation_matrix = rotation_matrix.squeeze(0).cpu().numpy()  # 将tensor转换为numpy数组
        translation_vector = translation_vector.squeeze(0).cpu().numpy()  # 转换为numpy数组

        return rotation_matrix, translation_vector

    def get_radius(self, r, extension=1.1):
        """计算最小半径"""
        # BBox 尺寸加上扩展系数
        r = r * extension

        # 距离计算：取水平和垂直的最大约束
        d_h = r / np.sin(self.FoVx / 2)  # 根据水平视角计算距离
        d_v = r / np.sin(self.FoVy / 2)  # 根据垂直视角计算距离

        return max(d_h, d_v)

    def get_camera_extrinsics(self, c_min, c_max, n_cameras=64, extension=1.25):
        """
        输入 BBox 的最小点、最大点、相机数量和扩展系数，生成所有相机的外参
        """
        # 计算 BBox 的中心和尺寸
        bbox_center = (c_min + c_max) / 2
        r = torch.norm(c_max - c_min) / 2.0
        print(c_max)
        print(c_min)
        print(r)
        

        # 计算最小安全距离
        radius = self.get_radius(r, extension)
        print(radius)

        # 生成球面上均匀分布的相机位置
        camera_positions = self.generate_fibonacci_sphere(n_cameras, radius, bbox_center)

        # 计算每个相机的外参
        extrinsics = []
        path = '/home/cyc/pycharm/lxy/3DGS/debug/mesh/camera.obj'
        with open(path, 'w') as fp:
            for cam_pos in camera_positions:
                v = cam_pos.detach().cpu().numpy()
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

                R, t = self.compute_camera_extrinsics(cam_pos, bbox_center)
                extrinsics.append((R, t))
        
        # for cam_pos in camera_positions:
        #     v = cam_pos.detach().cpu().numpy()

        #     R, t = self.compute_camera_extrinsics(cam_pos, bbox_center)
        #     extrinsics.append((R, t))
        return extrinsics