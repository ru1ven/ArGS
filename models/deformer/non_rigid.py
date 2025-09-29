
import torch
import torch.nn as nn
import pytorch3d.transforms as tf

from models.network_utils import (VanillaCondMLP,
                                  HashGrid, pixel_align)

from right_hand_model import MANO
import pytorch3d.ops as ops
from utils.pointnet_utils import index_points
import torch.nn.functional as F


class NonRigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, img_feature, iteration, camera, compute_loss=True):
        raise NotImplementedError

    def apply_non_rigid_trans(self, gaussians, refined_gaussians, deltas):
        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        refined_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            refined_gaussians._scaling = gaussians._scaling + delta_scale
        # elif scale_offset == 'exp':
        #     refined_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        # elif scale_offset == 'zero':
        #     delta_scale = torch.zeros_like(delta_scale)
        #     refined_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')

        if rot_offset == 'mult':
            q1 = delta_rot
            q1[:,0] = 1.  # [1,0,0,0] represents identity rotation
            # q1[0] = 1.  #
            delta_rot = delta_rot[:,1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            refined_gaussians._rotation = tf.quaternion_multiply(q1, q2)

        else:
            raise ValueError

        return refined_gaussians



class Non_Rigid(NonRigidDeform):
    def __init__(self, cfg, metadata, metadata_obj, ho_type):
        super().__init__(cfg)

        d_cond = 256
    
        self.latent_dim = 0
        self.frame_dict = metadata['frame_dict']
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim
        self.ho_type = ho_type
        self.metadata = metadata
        self.metadata_obj = metadata_obj

        self.hashgrid = HashGrid(cfg.hashgrid)
        self.clip_fc = nn.Linear(768 * 2, d_cond)

        # for a pre rigid trans
        if ho_type != 'obj':
            self.smpl_verts = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
            self.skinning_weights = torch.from_numpy(metadata["skinning_weights"]).float().cuda()
        self.h, self.w = 1000, 1000
        self.roi_size = cfg.get('roi_size', 224)

        self.delta_mlp = VanillaCondMLP(103, d_cond, d_out, cfg.mlp)

        self.L2Loss = nn.SmoothL1Loss(reduction="mean").cuda()
        self.delta_history = {}

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def get_jtr(self, body):
        Jtrs = body['Jtr_a_pose']

        v_shaped = body['v_shaped']
        v_shaped = v_shaped.detach()
        center = torch.mean(v_shaped, dim=1)
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



    def forward(self, gaussians, img_feature, iteration, camera, compute_loss=True, delay=False, prev_data=None):
        loss_reg = {}

        refined_gaussians = gaussians.clone()

        setattr(refined_gaussians, "non_rigid_feature",
                torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
        updated_camera = camera.copy()
        # if delay:
        #     # for 3dgs-avartar
        #     return updated_camera, None, refined_gaussians, loss_reg, None, None, None, None
        #
        # else:
        pixel_feat = img_feature.last_hidden_state
        pose_feat_global = img_feature.pooler_output

        if self.ho_type in ['left', 'right']:
            aabb = self.metadata['aabb'].cuda()
        else:
            aabb = self.metadata_obj[camera.obj_id]['obj_aabb'].cuda()

        xyz = gaussians.get_xyz
        xyz_norm = aabb.normalize(xyz, sym=True)
        # deformed_gaussians = gaussians.clone()

        f_sh = gaussians.get_features.reshape(xyz.shape[0], -1)
        f_opacity = gaussians.get_opacity
        f_cov = gaussians.get_covariance()

       
        pc_feature_hashgrid = self.hashgrid(xyz_norm).float()
        pc_feature = torch.cat([pc_feature_hashgrid, f_cov, f_sh, f_opacity], dim=-1)
        
        # get pixel_aligned points
        xyz_points = gaussians.get_xyz.clone()
        xyz_canonical = xyz_points.clone()

        # a pre visual-driven rigid trans, to reduce the difficulty of obj non-rigid trans

        if self.ho_type == 'obj':
            # if self.training:
            #     camera, loss_reg = self.pose_correct_obj(camera, pose_feat_global, pose_model)
            n_pts = xyz_points.shape[0]
            rot = camera.obj_rots
            trans = camera.obj_trans

            T_fwd = torch.eye(4).to(rot.device)
            T_fwd[:3, :3] = rot.float().to(rot.device)
            T_fwd[:3, 3] = trans
            T_fwd = T_fwd.repeat(n_pts, 1, 1)

            homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz_points.device)
            x_hat_homo = torch.cat([xyz_points, homo_coord], dim=-1).view(n_pts, 4, 1)
            xyz_point_trans = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        else:
            # if self.training:
            #     camera, loss_reg = self.pose_correct_hand(camera, pose_feat_global, pose_model)
            if self.ho_type == 'left':
                bone_transforms = camera.bone_transforms_l
            elif self.ho_type == 'right':
                bone_transforms = camera.bone_transforms_r
            n_pts = xyz_points.shape[0]
            pts_W = self.query_weights(xyz_points)
            T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_pts, 4, 4).float()

            homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz_points.device)
            x_hat_homo = torch.cat([xyz_points, homo_coord], dim=-1).view(n_pts, 4, 1)
            xyz_point_trans = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]

        pixel_feat = pixel_feat.permute(0, 2, 1)
        pixel_feat = pixel_feat[:,:,1:].reshape(-1,768,7,7)

        pose_feat_pixel, roi_color_pixel = pixel_align(camera, xyz_point_trans, xyz_point_trans.shape[0], pixel_feat,
                                         camera.full_proj_transform, self.w, self.h, camera.trans_img2roi,
                                         self.ho_type,self.roi_size)



        pose_feat_global_N = pose_feat_global.expand(xyz_point_trans.shape[0], -1)
        pose_feat = self.clip_fc(torch.cat([pose_feat_global_N, pose_feat_pixel], dim=-1))

        frame_idx = camera.frame_id
        latent_idx = self.frame_dict[frame_idx]
        if self.latent_dim > 0:
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        deltas = self.delta_mlp(pc_feature, cond=pose_feat)

        if self.ho_type == 'obj':
            movable_prob = refined_gaussians.get_dynamic
            deltas = deltas * movable_prob

        else:
            movable_prob = None

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]
        delta_rot = delta_rot[:, 1:]

        if self.feature_dim > 0:
            setattr(refined_gaussians, "non_rigid_feature", deltas[:, 10:])



        canonical_gs = refined_gaussians.clone()

        # if not delay and (self.ho_type == 'obj' and int(latent_idx) != 0):
        #     refined_gaussians = self.apply_non_rigid_trans(gaussians, refined_gaussians, deltas)
        # elif latent_idx==0:
        #     print('first frame')
        if not delay:
            refined_gaussians = self.apply_non_rigid_trans(gaussians, refined_gaussians, deltas)

        if compute_loss and self.ho_type == 'obj':
        #if compute_loss and self.ho_type == 'obj':
            loss_nr_reg_loss = 0.
            if not delay:
                # regularization
                loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
                loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
                loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            else:
                loss_xyz=0
                loss_scale=0
                loss_rot=0
            if iteration >1200 and not delay:
            # if iteration > 1200:
                loss_nr_reg_loss = 0.5 * torch.mean(movable_prob * (1 - movable_prob)) + 0.1 * torch.mean(movable_prob)

            # # === 时序一致性正则 ===
            frame_idx = camera.frame_id
            self.delta_history[frame_idx] = delta_xyz.detach().cpu()

            #     loss_temporal_scale = torch.zeros_like(loss_scale)
            #     loss_temporal_rot = torch.zeros_like(loss_rot)

            loss_reg.update({
                'nr_xyz_{}'.format(self.ho_type): loss_xyz,
                'nr_scale_{}'.format(self.ho_type): loss_scale,
                'nr_rot_{}'.format(self.ho_type): loss_rot,
                'nr_reg_{}'.format(self.ho_type): loss_nr_reg_loss
                # 'nr_tempo_scale_{}'.format(self.ho_type): loss_temporal_scale,
                # 'nr_tempo_rot_{}'.format(self.ho_type): loss_temporal_rot
            })


        return updated_camera, movable_prob, refined_gaussians, canonical_gs, loss_reg


def get_non_rigid_deform(cfg, metadata, metadata_obj, type='hand'):
    name = cfg.name
    model_dict = {

        #"3dgs_avatar": Non_Rigid_3dgs_avatar,
        "hashgrid": Non_Rigid,
        #"non_rigid_feat_ony": Non_Rigid,
        #"hashgrid": Non_Rigid_wo_visual
    }
    return model_dict[name](cfg, metadata, metadata_obj, type)