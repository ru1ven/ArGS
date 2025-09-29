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


class Camera_multi_batch:
    def __init__(self, camera=None, **kwargs):
        if camera is not None:
            self.data = camera.data.copy()
            return

        self.data = kwargs
        self.data['trans'] = np.array([0.0, 0.0, 0.0])
        self.data['scale'] = 1.0

        self.data['original_image'] = self.image.clamp(0.0, 1.0)
        self.data['image_width'] = self.original_image.shape[2]
        self.data['image_height'] = self.original_image.shape[1]
        self.data['original_mask'] = self.mask.float()
        self.data['obj_image'] = self.obj_image.clamp(0.0, 1.0)
        self.data['obj_mask'] = self.obj_mask.float()
        self.data['full_image'] = self.full_image.clamp(0.0, 1.0)
        # self.data['full_image_ori'] = self.full_image_ori.clamp(0.0, 1.0)
        self.data['full_mask'] = self.full_mask.float()
        self.data['bbox'] = self.bbox.float()

        self.data['img_ROI'] = self.img_ROI.clamp(0.0, 1.0)
        self.data['trans_img2roi'] = torch.tensor(self.trans_img2roi)

        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01

        self.data['world_view_transform'] = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                             fovY=self.FoVy).transpose(0, 1)
        self.data['full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.data['camera_center'] = self.world_view_transform.inverse()[3, :3]

        self.data['rots'] = self.rots
        self.data['Jtrs'] = self.Jtrs
        self.data['bone_transforms'] = self.bone_transforms

        self.data['obj_rots'] = self.obj_rots
        self.data['obj_trans'] = self.obj_trans

        self.data['R'] = torch.tensor(self.R)
        self.data['T'] = torch.tensor(self.T)
        self.data['K'] = torch.tensor(self.K)

        self.data['obj_rots_gt'] = self.obj_rots_gt
        self.data['obj_trans_gt'] = self.obj_trans_gt
        self.data['hand_param'] = self.hand_param
        self.data['hand_param_gt'] = self.hand_param_gt
        self.data['joints_gt'] = self.joints_gt
        self.data['hand_root'] = self.hand_root

        # self.data['Jtrs_gt'] = self.Jtrs_gt.to(self.data_device)

    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = Camera_multi_batch(camera=self)
        return new_cam

    def merge(self, cam):
        self.data['frame_id'] = cam.frame_id
        self.data['rots'] = cam.rots.detach()
        self.data['Jtrs'] = cam.Jtrs.detach()
        self.data['bone_transforms'] = cam.bone_transforms.detach()


class Viewpoint_data:
    def __init__(self, camera=None, **kwargs):

        self.data = kwargs

        for key, value in camera.items():
            if isinstance(value, torch.Tensor):
                self.data[key] = value.squeeze(0).clone()
            else:
                self.data[key] = value

        self.data['hand_param'] = self.data['hand_param'].unsqueeze(0)
        self.data['hand_param_gt'] = self.data['hand_param_gt'].unsqueeze(0)
        self.data['subject_id'] = int(self.data['subject_id'])
        self.data['obj_id'] = int(self.data['obj_id'])
        self.data['image_name'] = str(self.data['image_name'])

        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01

    def copy(self):
        new_cam = Camera(camera=self)
        return new_cam

    def __getattr__(self, item):
        return self.data[item]
