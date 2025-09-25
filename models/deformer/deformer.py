import torch.nn as nn

from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform

class Deformer(nn.Module):
    def __init__(self, cfg, metadata, metadata_obj, hand_side='hand'):
        super().__init__()
        self.cfg = cfg
        if hand_side == 'obj':
            self.rigid = get_rigid_deform(cfg.rigid, metadata_obj, hand_side)
        else:
            self.rigid = get_rigid_deform(cfg.rigid, metadata, hand_side)

        if hasattr(self.cfg, 'non_rigid'):
            self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata,metadata_obj, hand_side)
        else:
            print('non_rigid = None')
            self.non_rigid = None

    # def forward(self, gaussians, img_feat, camera, iteration, pose_model, compute_loss=True, delay=False):
    #     loss_reg = {}
    #     if delay:
    #         camera, refined_gaussians, loss_non_rigid = self.non_rigid(gaussians, img_feat, iteration, camera,pose_model, compute_loss, delay)
    #         deformed_gaussians = self.rigid(refined_gaussians, iteration, camera, pose_model)
    #         loss_reg.update(loss_non_rigid)
    #         return camera, deformed_gaussians, loss_reg
    #
    #     else:
    #         camera, posed_gaussians, refined_gaussians, loss_non_rigid, pc_feature, xyz_norm, pose_feat, roi_color_pixel = self.non_rigid(
    #             gaussians, img_feat, iteration, camera,
    #             pose_model, compute_loss, delay)
    #         # deformed_gaussians = self.rigid(deformed_gaussians, iteration, camera)
    #         loss_reg.update(loss_non_rigid)
    #         return camera, posed_gaussians, refined_gaussians, loss_reg, pc_feature, xyz_norm, pose_feat, roi_color_pixel


def get_deformer(cfg, metadata,metadata_obj, hand_side):
    return Deformer(cfg, metadata,metadata_obj, hand_side)

def get_deformer_obj(cfg, metadata, metadata_obj):
    #del cfg.non_rigid
    cfg.rigid.name = 'obj_deform'
    return Deformer(cfg, metadata,metadata_obj,'obj')
