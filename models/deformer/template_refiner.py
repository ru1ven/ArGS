import torch
import torch.nn as nn
import torchvision

from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform, NonRigidDeform
from models.network_utils import OfficialResNetUnet, VanillaCondMLP
from models.resnet import ResNet, BasicBlock
import pytorch3d.transforms as tf


class F_img (nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
        pretrain_weight = torchvision.models.resnet18(pretrained=True)
        self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)


    def forward(self, gaussians, camera, iteration):
        img_ROI = camera.img_ROI


class MLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.cfg = cfg
        #self.backbone = OfficialResNetUnet()
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
        pretrain_weight = torchvision.models.resnet18(pretrained=True)
        self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
        d_cond = 256
        self.globel_fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=d_cond, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d_cond),
            nn.ReLU(inplace=True))

        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_in = 3
        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)
        self.aabb = metadata['aabb']

        self.delay = cfg.get('delay', 0)


    def forward(self, gaussians, camera, iteration, compute_loss=True):
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature", torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        img_ROI = camera.img_ROI.unsqueeze(0)

        _, c1, c2, c3, pose_feat = self.backbone(img_ROI)
        #print(pose_feat.shape)
        pose_feat = self.globel_fc(pose_feat)
        pose_feat = pose_feat.mean(3).mean(2)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:,0] = 1. # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:,1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])


        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                're_xyz': loss_xyz,
                're_scale': loss_scale,
                're_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg



def get_template_refiner(cfg, metadata):

    name = cfg.name
    model_dict = {
        "image_feature": MLP,
    }
    return model_dict[name](cfg, metadata)
