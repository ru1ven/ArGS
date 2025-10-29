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
import numpy as np

from utils.camera_utils import get_frustum_mask
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, rotation_matrix_to_quaternion
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation

import trimesh
import igl

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.dynamic_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, cfg):
        self.cfg = cfg

        # two modes: SH coefficient or feature
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._dynamic = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def clone(self):
        cloned = GaussianModel(self.cfg)

        properties = ["active_sh_degree",
                      "non_rigid_feature",
                      ]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = ["_xyz",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity",
                      "_dynamic"]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        return cloned

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._dynamic,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer else None,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._dynamic,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        #_,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    def copy_state_from(self, other: "GaussianModel"):
        """
        将另一个 GaussianModel 的状态复制到当前对象（就地拷贝，保留原有 nn.Parameter 对象与 optimizer 状态）。
        使用场景：render 得到一个临时 GaussianModel（无梯度），想把它的值写回正在训练的模型，但不破坏 optimizer。
        要求：other 中对应 tensor 的 shape 必须与当前对象一致（或能广播/转换为一致）。
        """
        with torch.no_grad():
            # 要按需同步的参数名列表（按你的类定义）
            param_names = [
                "_xyz",
                "_features_dc",
                "_features_rest",
                "_scaling",
                "_rotation",
                "_opacity",
                "_dynamic"
            ]

            # 1) 先同步那些是 nn.Parameter 的字段：就地写入 .data 保持参数对象不变
            for name in param_names:
                if hasattr(self, name) and hasattr(other, name):
                    src = getattr(other, name)
                    dst = getattr(self, name)

                    # 如果目标是 nn.Parameter，尽量用 data.copy_ 保留对象与 optimizer state
                    if isinstance(dst, torch.nn.Parameter):
                        # 把 src 转到 dst device / dtype 并拷贝数据
                        src_t = src.detach().to(dst.device).type_as(dst)
                        if src_t.shape != dst.data.shape:
                            raise RuntimeError(f"Shape mismatch for {name}: src {src_t.shape} vs dst {dst.data.shape}")
                        dst.data.copy_(src_t)
                    else:
                        # dst 不是 Parameter，则直接替换（保持同名属性）
                        setattr(self, name, src.detach().clone().to(dst.device if hasattr(dst, "device") else "cuda"))

            # # 2) 同步其他非参数缓冲 / 状态
            # other_buffers = [
            #     "max_radii2D",
            #     "xyz_gradient_accum",
            #     "denom",
            #     "percent_dense",
            #     "spatial_lr_scale",
            #     "active_sh_degree",
            # ]
            # for name in other_buffers:
            #     if hasattr(other, name):
            #         val = getattr(other, name)
            #         # 如果 self 有该属性并且是 tensor，就 in-place copy，否则直接赋值 clone
            #         if hasattr(self, name):
            #             dst = getattr(self, name)
            #             if isinstance(dst, torch.Tensor) and isinstance(val, torch.Tensor):
            #                 dst_device = dst.device if dst.is_cuda else torch.device("cpu")
            #                 val_t = val.detach().to(dst_device).type_as(dst)
            #                 if val_t.shape == dst.shape:
            #                     dst.copy_(val_t)
            #                 else:
            #                     # shapes differ: replace attribute
            #                     setattr(self, name, val_t.clone())
            #             else:
            #                 # 非 tensor 或 shape 不匹配，直接替换
            #                 setattr(self, name, val.detach().clone() if isinstance(val, torch.Tensor) else val)
            #         else:
            #             # self 没有该属性，直接赋值
            #             setattr(self, name, val.detach().clone() if isinstance(val, torch.Tensor) else val)

            # 3) 可选：如果你希望保持 rotation_precomp 等字段
            if hasattr(other, "rotation_precomp"):
                self.rotation_precomp = other.rotation_precomp.detach().clone() if isinstance(other.rotation_precomp,
                                                                                              torch.Tensor) else other.rotation_precomp

        # end with torch.no_grad()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_dynamic(self):
        return self.dynamic_activation(self._dynamic)

    @property
    def get_raw_dynamic(self):
        return self._dynamic
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_precomp(self):
        if hasattr(self, 'rotation_precomp'):
            return self.rotation_activation(rotation_matrix_to_quaternion(self.rotation_precomp))
        #print('no rotation_precomp')
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        if hasattr(self, 'rotation_precomp'):
            return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation_precomp)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self):
        # opacity classification loss
        opacity = self.get_opacity
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def get_view2gaussian(self, viewmatrix):
        r = self._rotation
        norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        rots = R
        xyz = self.get_xyz
        N = xyz.shape[0]
        G2W = torch.zeros((N, 4, 4), device='cuda')
        G2W[:, :3, :3] = rots  # TODO check if we need to transpose here
        G2W[:, :3, 3] = xyz
        G2W[:, 3, 3] = 1.0

        viewmatrix = viewmatrix.transpose(0, 1)
        G2V = viewmatrix @ G2W

        R = G2V[:, :3, :3]
        t = G2V[:, :3, 3]

        t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
        V2G = torch.zeros((N, 4, 4), device='cuda')
        V2G[:, :3, :3] = R.transpose(1, 2)
        V2G[:, :3, 3] = t2
        V2G[:, 3, 3] = 1.0

        # transpose view2gaussian to match glm in CUDA code
        V2G = V2G.transpose(2, 1).contiguous()

        # precompute results to reduce computation and IO
        #scales = self.get_scaling_with_3D_filter
        scales = self.get_scaling
        S_inv_square = 1.0 / (scales ** 2)
        R = V2G[:, :3, :3].transpose(1, 2)
        t2 = V2G[:, 3:, :3]

        C = torch.sum((t2 ** 2) * S_inv_square[:, None, :], dim=2)
        S_inv_square_R = S_inv_square[:, :, None] * R
        B = t2 @ S_inv_square_R
        Sigma = R.transpose(1, 2) @ S_inv_square_R
        merged = torch.cat([Sigma[:, :, 0], Sigma[:, 1:, 1], Sigma[:, 2:, 2], B.squeeze(), C], dim=1)

        return merged

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale=1.,lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        self.lr_scale = lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        if self.use_sh:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_color.shape[0], 1, self.feature_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        dynamic = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._dynamic = nn.Parameter(dynamic.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        feature_ratio = 20.0 if self.use_sh else 1.0
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * self.lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * self.lr_scale, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / feature_ratio * self.lr_scale, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * self.lr_scale, "name": "opacity"},
            {'params': [self._dynamic], 'lr': training_args.dynamic_lr * self.lr_scale, "name": "dynamic"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * self.lr_scale, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale * self.lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale * self.lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def refine_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        feature_ratio = 20.0 if self.use_sh else 1.0
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * self.lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr * self.lr_scale, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / feature_ratio * self.lr_scale, "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr * self.lr_scale, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * self.lr_scale, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * self.lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * self.lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, save_dynamic=True):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        if save_dynamic:
            l.append('dynamic')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path, save_dynamic=True, clean=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        if save_dynamic:
            dynamic = self._dynamic.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        
        if hasattr(self, 'rotation_precomp'):
            rotation = Rotation.from_matrix(self.rotation_precomp.detach().cpu().numpy())
            quaternion = rotation.as_quat()  # 四元数形式 (x, y, z, w)
            # 转换四元数顺序: (x, y, z, w) -> (w, x, y, z)
            rotation = np.array([q[[3, 0, 1, 2]] for q in quaternion])
        else:
            rotation = self._rotation.detach().cpu().numpy()

        if clean:
            # ---- Step 1: 过滤低透明度点 ----
            valid_mask = opacities[:, 0] >= 0.01
            removed_count = np.count_nonzero(~valid_mask)
            # print(f"🧹 已筛除低透明度点数: {removed_count} (opacity < 0.01)")

            # 过滤后的数据
            xyz = xyz[valid_mask]
            normals = normals[valid_mask]
            f_dc = f_dc[valid_mask]
            f_rest = f_rest[valid_mask]
            opacities = opacities[valid_mask]
            if save_dynamic:
                dynamic = dynamic[valid_mask]
            scale = scale[valid_mask]
            rotation = rotation[valid_mask]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(save_dynamic)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if save_dynamic:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, dynamic, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

   
    def save_parted_ply(self, path, clean=True):
        os.makedirs(path, exist_ok=True)

            # ---- 数据提取 ----
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        dynamic = self._dynamic.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # conv = self.get_covariance().detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(True)]
        if clean:
            # ---- Step 1: 过滤低透明度点 ----
            valid_mask = opacities[:, 0] >= 0.01
            removed_count = np.count_nonzero(~valid_mask)
            # print(f"🧹 已筛除低透明度点数: {removed_count} (opacity < 0.01)")

            # 过滤后的数据
            xyz = xyz[valid_mask]
            normals = normals[valid_mask]
            f_dc = f_dc[valid_mask]
            f_rest = f_rest[valid_mask]
            opacities = opacities[valid_mask]
            dynamic = dynamic[valid_mask]
            scale = scale[valid_mask]
            rotation = rotation[valid_mask]

        # ---- Step 2: 按 dynamic 分两部分 ----
        mask_part0 = dynamic[:, 0] <= 0.5
        mask_part1 = ~mask_part0
        masks = [mask_part0, mask_part1]

        # ---- Step 3: 分别保存 ----
        for i, mask in enumerate(masks):
            attributes = np.concatenate((
                xyz[mask],
                normals[mask],
                f_dc[mask],
                f_rest[mask],
                opacities[mask],
                dynamic[mask],
                scale[mask],
                rotation[mask]
            ), axis=1)

            elements = np.empty(attributes.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, attributes))

            el = PlyElement.describe(elements, 'vertex')
        
        
            PlyData([el]).write(os.path.join(path, f'part_{i}.ply'))

    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        dynamics = np.asarray(plydata.elements[0]["dynamic"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        non_rigid_feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("non_rigid_feature")]
        non_rigid_feature_names = sorted(non_rigid_feature_names, key=lambda x: int(x.split('_')[-1]))
        non_rigid_feature = np.zeros((xyz.shape[0], len(non_rigid_feature_names)))
        for idx, attr_name in enumerate(non_rigid_feature_names):
            non_rigid_feature[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._dynamic = nn.Parameter(torch.tensor(dynamics, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.non_rigid_feature = nn.Parameter(torch.tensor(non_rigid_feature, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._dynamic = optimizable_tensors["dynamic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_dynamic, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "dynamic" : new_dynamic,
        "scaling": new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._dynamic = optimizable_tensors["dynamic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_dynamic = self._dynamic[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_dynamic, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_dynamic = self._dynamic[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_dynamic, new_scaling, new_rotation)

    def densify_and_prune(self, opt, scene, max_screen_size):
        extent = scene.cameras_extent

        max_grad = opt.densify_grad_threshold
        min_opacity = opt.opacity_threshold

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        #print('low_opacity:{}%'.format(torch.sum(prune_mask).item()/prune_mask.numel()))
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        #print('prune_mask:{}%'.format(torch.sum(prune_mask).item() / prune_mask.numel()))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    @torch.no_grad()
    def get_tetra_points(self, views, near: float = 0.02, far: float = 1e6):
        M = trimesh.creation.box()
        M.vertices *= 2

        rots = build_rotation(self.get_rotation_precomp)
        xyz = self.get_xyz
        scale = self.get_scaling * 3.  # TODO test
        # filter points with small opacity for bicycle scene
        # opacity = self.get_opacity_with_3D_filter
        # mask = (opacity > 0.1).squeeze(-1)
        # xyz = xyz[mask]
        # scale = scale[mask]
        # rots = rots[mask]

        vertices = M.vertices.T
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)

        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)

        # Mask out vertices outside of context views
        vertex_mask = get_frustum_mask(vertices, views, near, far)

        # path = '/home/cyc/pycharm/lxy/3DGS/debug/mesh/tetra_points.obj'
        # with open(path, 'w') as fp:
        #     for v in xyz.detach().cpu().numpy():
        #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        #
        #return xyz, scale

        path = '/home/cyc/pycharm/lxy/3DGS/debug/tetra_points.obj'
        with open(path, 'w') as fp:
            for v in vertices[vertex_mask].detach().cpu().numpy():
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        return vertices[vertex_mask], vertices_scale[vertex_mask]