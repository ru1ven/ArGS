import cv2
import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
import torchvision
from omegaconf import OmegaConf

from models.resnet import ResNet, Residual, BasicBlock
from torch.nn import functional as F


def homoify(points):
    """
    Convert a batch of points to homogeneous coordinates.
    Args:
        points: e.g. (B, N, 3) or (N, 3)
    Returns:
        homoified points: e.g., (B, N, 4)
    """
    points_dim = points.shape[:-1] + (1,)
    ones = points.new_ones(points_dim)

    return torch.cat([points, ones], dim=-1)


def dehomoify(points):
    """
    Convert a batch of homogeneous points to cartesian coordinates.
    Args:
        homogeneous points: (B, N, 4/3) or (N, 4/3)
    Returns:
        cartesian points: (B, N, 3/2)
    """
    return points[..., :-1] / points[..., -1:]


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    if multires == 0:
        return lambda x: x, input_dims
    assert multires > 0

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


class HannwEmbedder:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        # get hann window weights
        if self.cfg.full_band_iter <= 0 or self.cfg.kick_in_iter >= self.cfg.full_band_iter:
            alpha = torch.tensor(N_freqs, dtype=torch.float32)
        else:
            kick_in_iter = torch.tensor(self.cfg.kick_in_iter,
                                        dtype=torch.float32)
            t = torch.clamp(self.kwargs['iter_val'] - kick_in_iter, min=0.)
            N = self.cfg.full_band_iter - kick_in_iter
            m = N_freqs
            alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            # print("freq_idx: ", freq_idx, "weight: ", w, "iteration: ", self.kwargs['iter_val'])
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_hannw_embedder(cfg, multires, iter_val, ):
    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
        'iter_val': iter_val
    }

    embedder_obj = HannwEmbedder(cfg, **embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class HierarchicalPoseEncoder(nn.Module):
    '''Hierarchical encoder from LEAP.'''

    def __init__(self, num_joints=24, rel_joints=False, dim_per_joint=6, out_dim=-1, **kwargs):
        super().__init__()

        self.num_joints = num_joints
        self.rel_joints = rel_joints
        # self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
        #     9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)
        self.ktree_parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10,
                                       11, 0, 13, 14], dtype=np.int32)

        self.layer_0 = nn.Linear(9 * num_joints + 3 * num_joints, dim_per_joint)
        dim_feat = 13 + dim_per_joint

        layers = []
        for idx in range(num_joints):
            layer = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim_per_joint))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if out_dim <= 0:
            self.out_layer = nn.Identity()
            self.n_output_dims = num_joints * dim_per_joint
        else:
            self.out_layer = nn.Linear(num_joints * dim_per_joint, out_dim)
            self.n_output_dims = out_dim

    def forward(self, rots, Jtrs, skinning_weight=None):
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat([rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1)
        global_feat = self.layer_0(global_feat)
        # global_feat = (self.layer_0.weight@global_feat[0]+self.layer_0.bias)[None]
        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = out[parent]
                bone_l = torch.norm(Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :], dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(out, dim=-1)
        out = self.out_layer(out)
        return out


class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(VanillaCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            embed_fn, input_ch = get_embedder(config.multires, input_dims=dim_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.last_layer_init = config.get('last_layer_init', False)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if self.last_layer_init and l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0., std=1e-5)
                torch.nn.init.constant_(lin.bias, val=0.)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, coords, cond=None):
        if cond is not None:
            if cond.shape[0] == 1:
                cond = cond.expand(coords.shape[0], -1)
        # if cond is not None:
        #     cond = cond.expand(coords.shape[0], -1)


        if self.embed_fn is not None:
            coords_embedded = self.embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

def points3DToImg( joint_xyz, para):
    joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
    if len(joint_uvd.shape) == 2:
        joint_uvd[ :, 0] = (joint_xyz[ :, 0] * para[0, 0] / (joint_xyz[ :, 2]+1e-8) + para[0, 2])
        joint_uvd[ :, 1] = (1 * joint_xyz[ :, 1] * para[1, 1] / (joint_xyz[ :, 2]) + para[1, 2])
        joint_uvd[ :, 2] = joint_xyz[ :, 2]
    elif len(joint_uvd.shape) == 3:
        joint_uvd[:, :, 0] = (joint_xyz[:,:, 0] * para[:,0, 0].unsqueeze(1) / (joint_xyz[:,:, 2] + 1e-8) + para[:,0, 2].unsqueeze(1))
        joint_uvd[:,:, 1] = (1 * joint_xyz[:,:, 1] * para[:,1, 1].unsqueeze(1) / (joint_xyz[:,:, 2]) + para[:,1, 2].unsqueeze(1))
        joint_uvd[:,:, 2] = joint_xyz[:,:, 2]
    else:
        raise NotImplementedError
    return joint_uvd

def pointsImgTo3D(point_uvd, para, flip=None):

    point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
    point_xyz[ :, 0] = (point_uvd[ :, 0] - para[0, 2]) * point_uvd[ :, 2] / para[0, 0]
    point_xyz[ :, 1] = 1 * (point_uvd[ :, 1] -para[1, 2]) * point_uvd[ :, 2] / para[1, 1]
    point_xyz[ :, 2] = point_uvd[ :, 2]
    return point_xyz


def pixel_align(camera, input_xyz_points, num_points_per_scene, feature_maps, full_proj_transform, img_w, img_h,trans_img2roi,type,roi_size):
    input_points = input_xyz_points.clone()
    input_points = input_points.reshape((-1, num_points_per_scene, 3))
    full_proj_transform = full_proj_transform.unsqueeze(0)
    trans_img2roi = trans_img2roi.unsqueeze(0)
    # print(feature_maps.shape)#b 256 8 8
    # print(full_proj_transform.shape)#b 4 4
    #print(input_xyz_points)

    batch_size = input_points.shape[0]
    # xyz = input_points * 2 / cfg.recon_scale + hand_center_3d.unsqueeze(1)
    homo_xyz = homoify(input_points)
    homo_xyz_2d = torch.matmul(full_proj_transform, homo_xyz.transpose(1, 2)).transpose(1, 2)
    xyz_2d = (homo_xyz_2d[:, :, :2] / homo_xyz_2d[:, :, [3]])
    #print(xyz_2d.mean())

    cam_coord = torch.mm(camera.R, input_points[0].transpose(1, 0)).transpose(1, 0) + camera.T.reshape(1, 3)
    xyz_2d = points3DToImg(cam_coord,camera.K)[:, :2].unsqueeze(0)
    # print("w:",xyz_2d[:,0].max())
    # print("w:", xyz_2d[:, 0].min())
    # print("h:",xyz_2d[:, 1].max())
    # print("h:", xyz_2d[:, 1].min())

    # # 定义图像尺寸
    # w, h = img_w, img_h  # 你可以根据需要调整图像尺寸
    # # 创建黑色背景图
    # background = np.zeros((h, w, 3), dtype=np.uint8)
    # # 假设你已经有了坐标列表coords，每个坐标是一个元组 (x, y)
    # coords = xyz_2d.squeeze(0).detach().cpu().numpy()  # 替换为你的实际坐标列表
    # # 将坐标列表中的像素设置为白色
    # for coord in coords:
    #     x, y = int(coord[0]), int(coord[1])
    #     # 确保坐标在图像范围内
    #     if 0 <= x < w and 0 <= y < h:
    #         if type == 'hand':
    #             background[y, x] = (255, 0, 255)  # 设置为白色
    #         else:
    #             background[y, x] = (255, 255, 255)  # 设置为白色
    #     # else:
    #     #     print("!")
    #
    # # 保存结果图像
    # cv2.imwrite("/home/cyc/pycharm/lxy/gs/ho_gs/debug/img_ho_0_project_{}.png".format(type), background)

    # u_2d = xyz_2d[:,:, 0] / img_w * 2 - 1
    # v_2d = xyz_2d[:,:, 1] / img_h * 2 - 1
    # uv_2d = torch.cat([u_2d, v_2d], dim=-1)
    ones = torch.ones((batch_size,num_points_per_scene, 1), dtype=torch.float32).to(input_points.device)
    uv_homogeneous = torch.cat([xyz_2d, ones], dim=-1)  # (N, 3)
    uv_2d_roi = torch.matmul(trans_img2roi, uv_homogeneous.transpose(1, 2)).transpose(1, 2)[:,:,:2].unsqueeze(2) #b n 1 2

    # # 定义图像尺寸
    # w, h = 128, 128  # 你可以根据需要调整图像尺寸
    # # 创建黑色背景图
    # background = np.zeros((h, w, 3), dtype=np.uint8)
    # # 假设你已经有了坐标列表coords，每个坐标是一个元组 (x, y)
    # coords = uv_2d_roi.squeeze(2).squeeze(0).detach().cpu().numpy()  # 替换为你的实际坐标列表
    # # 将坐标列表中的像素设置为白色
    # for coord in coords:
    #     x, y = int(coord[0]), int(coord[1])
    #     # 确保坐标在图像范围内
    #     if 0 <= x < w and 0 <= y < h:
    #         if type == 'hand':
    #             background[y, x] = (255, 0, 255)  # 设置为白色
    #         else:
    #             background[y, x] = (255, 255, 255)  # 设置为白色
    #     # else:
    #     #     print("!")
    #
    # # 保存结果图像
    # cv2.imwrite("/home/cyc/pycharm/lxy/gs/ho_gs/debug/img_ROI_0_project{}.png".format(type), background)

    # print(uv_2d_roi.min())
    # print(uv_2d_roi.max())
    uv_2d_roi = uv_2d_roi / roi_size * 2 - 1

    sample_feat = torch.nn.functional.grid_sample(feature_maps, uv_2d_roi, align_corners=True)[:, :, :, 0].transpose(1, 2)
    sample_color = torch.nn.functional.grid_sample(camera.img_ROI.unsqueeze(0), uv_2d_roi, align_corners=True)[:, :, :,
                   0].transpose(1, 2)
    uv_2d_roi = uv_2d_roi.squeeze(2).reshape((-1, 2))
    sample_feat = sample_feat.reshape((uv_2d_roi.shape[0], -1))
    sample_color = sample_color.reshape((uv_2d_roi.shape[0], -1))
    validity = (uv_2d_roi[:, 0] >= -1.0) & (uv_2d_roi[:, 0] <= 1.0) & (uv_2d_roi[:, 1] >= -1.0) & (uv_2d_roi[:, 1] <= 1.0)
    validity = validity.unsqueeze(1)

    # if cfg.with_add_feats:
    #     depth_feat = input_xyz_points.reshape((-1, 3))[:, [-1]]
    #     view_dir_feat = F.normalize(input_xyz_points.reshape((-1, 3)), p=2, dim=1)
    #     sample_feat = torch.cat([sample_feat, depth_feat, view_dir_feat], axis=1)

    return sample_feat, sample_color
    #return feature_maps.mean(3).mean(2), None


def get_projected_uvd(camera, input_xyz_points, num_points_per_scene, trans_img2roi,roi_size):
    input_points = input_xyz_points.clone()
    input_points = input_points.reshape((-1, num_points_per_scene, 3))
    trans_img2roi = trans_img2roi.unsqueeze(0)
    batch_size = input_points.shape[0]
    cam_coord = torch.mm(camera.R, input_points[0].transpose(1, 0)).transpose(1, 0) + camera.T.reshape(1, 3)
    xyz_2d = points3DToImg(cam_coord,camera.K)[:, :2].unsqueeze(0)
    ones = torch.ones((batch_size,num_points_per_scene, 1), dtype=torch.float32).to(input_points.device)
    uv_homogeneous = torch.cat([xyz_2d, ones], dim=-1)  # (N, 3)
    uv_2d_roi = torch.matmul(trans_img2roi, uv_homogeneous.transpose(1, 2)).transpose(1, 2)[:,:,:2] #b n 2

    # # 定义图像尺寸
    # w, h = 224, 224  # 你可以根据需要调整图像尺寸
    # # 创建黑色背景图
    # #background = np.zeros((h, w, 3), dtype=np.uint8)
    # background = camera.img_ROI.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255
    # #print(background.shape)
    # # 假设你已经有了坐标列表coords，每个坐标是一个元组 (x, y)
    # coords = uv_2d_roi.squeeze(2).squeeze(0).detach().cpu().numpy()  # 替换为你的实际坐标列表
    # # 将坐标列表中的像素设置为白色
    # for coord in coords:
    #     x, y = int(coord[0]), int(coord[1])
    #     # 确保坐标在图像范围内
    #     if 0 <= x < w and 0 <= y < h:
    #
    #         background[y, x] = (255, 255, 255)  # 设置为白色
    #     # else:
    #     #     print("!")
    #
    # # 保存结果图像
    # cv2.imwrite("/home/cyc/pycharm/lxy/gs/ho_gs/debug/img_ROI_0_project.png", background)

    uv_2d_norm = uv_2d_roi / roi_size * 2 - 1
    depth = cam_coord.reshape((batch_size, -1, 3))[:, :, [-1]]
    return torch.cat([uv_2d_norm, depth], dim=-1).squeeze(0)


def get_unprojected_xyz(camera, input_uvd_points, num_points_per_scene, trans_img2roi, roi_size):
    input_uvd_points = input_uvd_points.reshape((-1, num_points_per_scene, 3))
    batch_size = input_uvd_points.shape[0]
    uv_2d_norm = input_uvd_points[:,:,:2]
    depth = input_uvd_points[:,:,[2]]
    uv_2d_roi = (uv_2d_norm+1)/2*roi_size

    ones = torch.ones((batch_size, num_points_per_scene, 1), dtype=torch.float32).to(input_uvd_points.device)
    uv_homogeneous_roi = torch.cat([uv_2d_roi, ones], dim=-1)
    uv_2d_img = torch.matmul(torch.inverse(trans_img2roi), uv_homogeneous_roi.transpose(1, 2)).transpose(1, 2)[:,:,:2] #b n 2

    uvd_img = torch.cat([uv_2d_img, depth], dim=-1).squeeze(0)
    cam_coord = pointsImgTo3D(uvd_img, camera.K)


    # # 定义图像尺寸
    # w, h = 640, 480  # 你可以根据需要调整图像尺寸
    # # 创建黑色背景图
    # #background = np.zeros((h, w, 3), dtype=np.uint8)
    # background = camera.full_image.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255
    # #print(background.shape)
    # # 假设你已经有了坐标列表coords，每个坐标是一个元组 (x, y)
    # coords = uv_2d_img.squeeze(2).squeeze(0).detach().cpu().numpy()  # 替换为你的实际坐标列表
    # # 将坐标列表中的像素设置为白色
    # for coord in coords:
    #     x, y = int(coord[0]), int(coord[1])
    #     # 确保坐标在图像范围内
    #     if 0 <= x < w and 0 <= y < h:
    #
    #         background[y, x] = (255, 255, 255)  # 设置为白色
    #     # else:
    #     #     print("!")
    #
    # # 保存结果图像
    # cv2.imwrite("/home/cyc/pycharm/lxy/gs/ho_gs/debug/img_0_unproject.png", background)

    R_inv = camera.R.transpose(0, 1)  # 由于 R 是正交矩阵，R 的逆等于它的转置
    # 计算世界坐标系中的点 xyz_points
    xyz_points = torch.mm(cam_coord - camera.T.reshape(1, 3), R_inv)

    return xyz_points


def get_skinning_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaCondMLP(n_input_dims, 0, n_output_dims, config)
    else:
        raise ValueError

    return network


class HannwCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(HannwCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            _, input_ch = get_hannw_embedder(config.embedder, config.multires, 0)
            dims[0] = input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if l in config.cond_in:
                # Conditional input layer initialization
                torch.nn.init.constant_(lin.weight[:, -dim_cond:], 0.0)
            torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, coords, iteration, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.config.multires > 0:
            embed_fn, _ = get_hannw_embedder(self.config.embedder, self.config.multires, iteration)
            coords_embedded = embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)

# from scipy.spatial.transform import Rotation
# def rotation_matrix_to_quaternion(R_matrix):
#     # 将旋转矩阵转换为四元数
#     quaternions = []
#     for i in range(R_matrix.shape[0]):
#         r = Rotation.from_matrix(R_matrix[i])
#         q = r.as_quat()
#         quaternions.append(q)
#     quaternions = np.array(quaternions)
#     return quaternions

class HashGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        xL = config.get('max_resolution', -1)
        if xL > 0:
            L = config.n_levels
            x0 = config.base_resolution
            config.per_level_scale = float(np.exp(np.log(xL / x0) / (L - 1)))
        self.encoding = tcnn.Encoding(3, config_to_primitive(config))
        self.n_output_dims = self.encoding.n_output_dims
        self.n_input_dims = self.encoding.n_input_dims

    def forward(self, x):
        x = (x + 1.) * 0.5  # [-1, 1] => [0, 1]

        return self.encoding(x)


class OfficialResNetUnet(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128, out_dim_list=[3 * 21, 21, 21]):
        super(OfficialResNetUnet, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = 18
        block, layers = BasicBlock, [2, 2, 2, 2]
        self.backbone = ResNet(block, layers)
        self.skip_layer4 = Residual(256 * block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512 * block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512 + 256), 256)

        self.skip_layer3 = Residual(128 * block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256 + 128), 128)

        self.skip_layer2 = Residual(64 * block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128 + 64), deconv_dim)

        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain:
            if layers_num == 18:
                print('load weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 50:
                print('load weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
