import torch
import torch.nn as nn

from utils.sh_utils import eval_sh, eval_sh_bases, augm_rots
from utils.general_utils import build_rotation
from models.network_utils import VanillaCondMLP

class ColorPrecompute(nn.Module):
    def __init__(self, cfg, metadata,metadata_obj,ho_type):
        super().__init__()
        self.cfg = cfg
        self.non_rigid_dim = cfg.get('non_rigid_dim', 0)
        self.metadata = metadata

    def forward(self, gaussians, camera):
        raise NotImplementedError


class SH(ColorPrecompute):
    def __init__(self, cfg, metadata, metadata_obj, ho_type):
        super().__init__(cfg, metadata, metadata_obj, ho_type)

    def precompute_color_multi_batch(self, xyz_posed, fwd_transform, f_sh, non_rigid_feature, camera):
        B, N, _ = xyz_posed.shape
        # print(f_sh.view(B, N, (self.cfg.sh_degree + 1) ** 2, 3).transpose(2, 3).shape)
        shs_view = f_sh.reshape(B, N, (self.cfg.sh_degree + 1) ** 2, 3).transpose(2, 3).view(-1, 3, (
                    self.cfg.sh_degree + 1) ** 2)

        dir_pp = (xyz_posed - camera['camera_center'].unsqueeze(1).repeat(1, N, 1))
        if self.cfg.cano_view_dir:
            T_fwd = fwd_transform
            R_bwd = T_fwd[:, :, :3, :3].transpose(2, 3)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1).view(B * N, -1)

        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
        sh2rgb = eval_sh(self.cfg.sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        return colors_precomp

    def forward(self, gaussians, camera):
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        if self.cfg.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = self.cfg.get('view_noise', 0.)
            if self.training and view_noise_scale > 0.:
                view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                          dtype=torch.float32,
                                          device=dir_pp.device).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, view_noise)

        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        return colors_precomp


class SH2RGB(ColorPrecompute):
    def __init__(self, cfg, metadata, metadata_obj,ho_type):
        super().__init__(cfg, metadata, metadata_obj,ho_type)
        #self.alpha = nn.Parameter(torch.tensor(0.))
        color_dim = 3
        #self.color_fc = nn.Linear(self.non_rigid_dim, color_dim)
        # self.mlp = VanillaCondMLP(color_dim, self.non_rigid_dim, color_dim, cfg.mlp)
        # self.color_activation = nn.Sigmoid()
        #self.color_weight = nn.Linear(self.non_rigid_dim+color_dim, 1)
        self.latent_dim = 64
        self.frame_dict = metadata['frame_dict']
        if self.latent_dim > 0:
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)
            self.color_fc = VanillaCondMLP(self.non_rigid_dim, self.latent_dim, color_dim, cfg.mlp)
        else:
            self.color_fc = VanillaCondMLP(self.non_rigid_dim, 0, color_dim, cfg.mlp)

        self.color_weight = nn.Sequential(
            nn.Linear(self.non_rigid_dim+color_dim, self.non_rigid_dim// 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.non_rigid_dim// 4, 1),
            #nn.Sigmoid()
        )

    # def precompute_color_multi_batch(self, xyz_posed, fwd_transform, f_sh, non_rigid_feature, camera):
    #     B, N, _ = xyz_posed.shape
    #     #print(f_sh.view(B, N, (self.cfg.sh_degree + 1) ** 2, 3).transpose(2, 3).shape)
    #     shs_view = f_sh.reshape(B, N, (self.cfg.sh_degree + 1) ** 2, 3).transpose(2, 3).view(-1, 3, (self.cfg.sh_degree + 1) ** 2)
    #
    #     dir_pp = (xyz_posed - camera['camera_center'].unsqueeze(1).repeat(1,N, 1))
    #     if self.cfg.cano_view_dir:
    #         T_fwd = fwd_transform
    #         R_bwd = T_fwd[:,:, :3, :3].transpose(2, 3)
    #         dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1).view(B*N, -1)
    #         # view_noise_scale = self.cfg.get('view_noise', 0.)
    #         # if self.training and view_noise_scale > 0.:
    #         #     view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
    #         #                               dtype=torch.float32,
    #         #                               device=dir_pp.device).transpose(0, 1)
    #         #     dir_pp = torch.matmul(dir_pp, view_noise)
    #
    #     dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
    #     sh2rgb = eval_sh(self.cfg.sh_degree, shs_view, dir_pp_normalized)
    #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #
    #
    #     # colors_precomp = (1-self.alpha)*colors_precomp + self.alpha *torch.sigmoid(gaussians.non_rigid_feature)
    #
    #     non_rigid_feature = non_rigid_feature.view(B*N, -1)
    #     alpha = torch.sigmoid(self.color_weight(torch.cat([colors_precomp, non_rigid_feature], dim=-1)))
    #
    #     colors_precomp = (1 - alpha) * colors_precomp + alpha * torch.sigmoid(
    #         self.color_fc(non_rigid_feature))
    #
    #     return colors_precomp

        
    def forward(self, gaussians, camera):
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        if self.cfg.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = self.cfg.get('view_noise', 0.)
            if self.training and view_noise_scale > 0.:
                view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                          dtype=torch.float32,
                                          device=dir_pp.device).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, view_noise)

        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        if self.non_rigid_dim > 0:
            assert hasattr(gaussians, "non_rigid_feature")
            if self.latent_dim > 0:
                frame_idx = camera.frame_id
                if frame_idx not in self.frame_dict:
                    latent_idx = len(self.frame_dict) - 1
                else:
                    latent_idx = self.frame_dict[frame_idx]
                latent_idx = torch.Tensor([latent_idx]).long().to(gaussians.non_rigid_feature.device)
                latent_code = self.latent(latent_idx)
                latent_code = latent_code.expand(gaussians.non_rigid_feature.shape[0], -1)
            #colors_precomp = (1-self.alpha)*colors_precomp + self.alpha *torch.sigmoid(gaussians.non_rigid_feature)
            alpha = torch.sigmoid(self.color_weight(torch.cat([colors_precomp,gaussians.non_rigid_feature],dim=-1)))
            colors_precomp = (1 - alpha) * colors_precomp + alpha * \
                             torch.sigmoid(self.color_fc(gaussians.non_rigid_feature, cond=latent_code))

        return colors_precomp



class ColorMLP(ColorPrecompute):
    def __init__(self, cfg, metadata, metadata_obj,ho_type):
        super().__init__(cfg, metadata, metadata_obj,ho_type)
        d_in = cfg.feature_dim

        self.use_xyz = cfg.get('use_xyz', False)
        self.use_cov = cfg.get('use_cov', False)
        self.use_normal = cfg.get('use_normal', False)
        self.sh_degree = cfg.get('sh_degree', 0)
        self.cano_view_dir = cfg.get('cano_view_dir', False)
        self.non_rigid_dim = cfg.get('non_rigid_dim', 0)
        # if ho_type == 'obj':
        #     self.non_rigid_dim = 0

        self.latent_dim = 64

        if self.use_xyz:
            d_in += 3
        if self.use_cov:
            d_in += 6 # only upper triangle suffice
        if self.use_normal:
            d_in += 3 # quasi-normal by smallest eigenvector...
        if self.sh_degree > 0:
            d_in += (self.sh_degree + 1) ** 2 - 1
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        if self.non_rigid_dim > 0:
            d_in += self.non_rigid_dim
        if self.latent_dim > 0:
            d_in += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = 3
        self.mlp = VanillaCondMLP(d_in, 0, d_out, cfg.mlp)
        self.color_activation = nn.Sigmoid()

    def compose_input(self, gaussians, camera, type):
        features = gaussians.get_features.squeeze(-1)
        n_points = features.shape[0]
        if self.use_xyz:
            if type == 'hand':
                aabb = self.metadata["aabb"]
                xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
                features = torch.cat([features, xyz_norm], dim=1)
            else:
                aabb = self.metadata_obj[camera.obj_id]["obj_aabb"]
                xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
                features = torch.cat([features, xyz_norm], dim=1)
        if self.use_cov:
            cov = gaussians.get_covariance()
            features = torch.cat([features, cov], dim=1)
        if self.use_normal:
            scale = gaussians._scaling
            rot = build_rotation(gaussians._rotation)
            normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
            features = torch.cat([features, normal], dim=1)
        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            if self.cano_view_dir:
                T_fwd = gaussians.fwd_transform
                R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = self.cfg.get('view_noise', 0.)
                if self.training and view_noise_scale > 0.:
                    view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                              dtype=torch.float32,
                                              device=dir_pp.device).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, view_noise)
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
            dir_embed = self.sh_embed(dir_pp_normalized)
            features = torch.cat([features, dir_embed], dim=1)
        if self.non_rigid_dim > 0:
            assert hasattr(gaussians, "non_rigid_feature")
            features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(features.shape[0], -1)
            features = torch.cat([features, latent_code], dim=1)

        return features


    def forward(self, gaussians, camera, type='hand'):
        inp = self.compose_input(gaussians, camera, type)
        output = self.mlp(inp)
        color = self.color_activation(output)
        return color


def get_texture(cfg, metadata,metadata_obj,ho_type):
    name = cfg.name
    model_dict = {
        "sh_only": SH,
        "sh2RGB": SH2RGB,
        "mlp": ColorMLP,
    }
    return model_dict[name](cfg, metadata,metadata_obj,ho_type)