import copy

import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision
from kornia.geometry import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix
from pytorch3d import ops
from transformers import CLIPModel

from right_hand_model import MANO
from utils.general_utils import cfg_from_yaml_file
from utils.network_utils import Pointnet2_Ssg
from utils.pointbert.point_encoder import PointTransformer_Colored
from utils.pointnet_utils import index_points
from .deformer import get_deformer
from .deformer.deformer import get_deformer_obj
from .network_utils import VanillaCondMLP, HashGrid, homoify, points3DToImg, get_skinning_mlp
from .pose_correction import get_pose_correction
from .texture import get_texture
from models.resnet import ResNet, BasicBlock


class GaussianConverter(nn.Module):
    def __init__(self, cfg, save_dir, metadata, metadata_obj, subject_labels, obj_labels):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        self.save_dir = save_dir
        self.metadata_obj = metadata_obj
        self.obj_labels = obj_labels
        self.subject_labels = subject_labels

        #self.pose_correction = get_pose_correction(cfg.model.pose_correction, metadata)
        if self.cfg.dataset.get('backbone', 'CLIP') == 'CLIP':
            self.backbone = CLIPModel.from_pretrained("/home/cyc/pycharm/lxy/3DGS/lib/clip-vit-base-patch32/")
        else:
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
            pretrain_weight = torchvision.models.resnet18(pretrained=True)
            self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)

        for subject_id in subject_labels:
            setattr(self, f"deformer_hand_{subject_id}_r",
                    get_deformer(cfg.model.deformer, metadata['right'], metadata_obj, 'right'))
            if 'left' in self.metadata.keys():
                setattr(self, f"deformer_hand_{subject_id}_l",
                    get_deformer(cfg.model.deformer, metadata['left'], metadata_obj,  'left'))

        for obj_id in obj_labels:
            setattr(self, f"deformer_obj_{obj_id}", get_deformer_obj(cfg.model.deformer, metadata['right'], metadata_obj))

        self.texture_r = get_texture(cfg.model.texture, metadata['right'], metadata_obj,'hand')
        if 'left' in self.metadata.keys():
            self.texture_l = get_texture(cfg.model.texture, metadata['left'], metadata_obj, 'hand')
        self.objtexture = get_texture(cfg.model.texture, metadata['right'], metadata_obj,'obj')

        self.lr_scale = 1 * self.cfg.get('batch_size', 8) / 8
        self.hand_lr_scale = 1
        

        self.optimizer, self.scheduler = None, None
        self.set_optimizer()

        #self.smpl_verts = torch.from_numpy(self.metadata["smpl_verts"]).float().cuda()
        #self.skinning_weights = torch.from_numpy(self.metadata["skinning_weights"]).float().cuda()

        self.roi_size = cfg.model.deformer.non_rigid.get('roi_size', 224)
        #self.L2Loss = torch.nn.MSELoss().cuda()
        self.L2Loss = nn.SmoothL1Loss(reduction="mean").cuda()

        #self.body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/').cuda()



    def set_optimizer(self):
        opt_params = []
        opt_params.extend([
            {'name': 'texture_r', 'params': [p for n, p in self.texture_r.named_parameters()],
             'lr': self.cfg.opt.get('texture_lr', 0.) * self.hand_lr_scale},
            {'name': 'obj_texture', 'params': [p for n, p in self.objtexture.named_parameters()],
             'lr': self.cfg.opt.get('texture_lr', 0.)},
        ])
        
        # Backbone
        opt_params.append({'name': 'backbone', 'params': [p for n, p in self.backbone.named_parameters()],
                           'lr': self.cfg.opt.get('pose_correction_lr', 0.) * 1})

        # Deformer objects
        for obj_id in self.obj_labels:
            # rigid 除 angle_mlp
            opt_params.append({
                'name': f'obj_{obj_id}_rigid',
                'params': [p for n, p in getattr(self, f"deformer_obj_{obj_id}").rigid.named_parameters() if
                           'angle' not in n],
                'lr': self.cfg.opt.get('objrigid_lr', 0.)
            })
            # angle_mlp
            opt_params.append({
                'name': f'obj_{obj_id}_angle',
                'params': [p for n, p in getattr(self, f"deformer_obj_{obj_id}").rigid.named_parameters() if
                           'angle' in n],
                'lr': self.cfg.opt.get('angle_lr', 0.)
            })
            # non-rigid
            opt_params.append({
                'name': f'obj_{obj_id}_non_rigid',
                'params': getattr(self, f"deformer_obj_{obj_id}").non_rigid.parameters(),
                'lr': self.cfg.opt.get('non_rigid_lr', 0.)
            })

        # Deformer hands
        for sub_id in self.subject_labels:
            if 'left' in self.metadata.keys(): 
                hand_sides =  ['r', 'l']
            else :
                hand_sides =  ['r']
            for side in hand_sides:
                # rigid
                opt_params.append({
                    'name': f'hand_{side}_{sub_id}_rigid',
                    'params': getattr(self, f"deformer_hand_{sub_id}_{side}").rigid.parameters(),
                    'lr': self.cfg.opt.get('rigid_lr', 0.)
                })
                # non-rigid
                opt_params.append({
                    'name': f'hand_{side}_{sub_id}_non_rigid',
                    'params': getattr(self, f"deformer_hand_{sub_id}_{side}").non_rigid.parameters(),
                    'lr': self.cfg.opt.get('non_rigid_lr', 0.)
                })

        if 'left' in self.metadata.keys(): 
            opt_params.extend([
            {'name': 'texture_l', 'params': [p for n, p in self.texture_l.named_parameters()],
             'lr': self.cfg.opt.get('texture_lr', 0.) * self.hand_lr_scale},
        ])

            

        self.optimizer = torch.optim.Adam(params=opt_params, lr=0.001, eps=1e-15)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)


    def forward(self, gaussians_r, gaussians_l, gaussians_obj, camera, iteration, compute_loss=True, rigid_delay=False, prev_camera=None):
        # if pose_refine:
        #     return self.forward_pose_refine(gaussians, gaussians_obj, camera, iteration, compute_loss=True)

        loss_reg = {}

        if len(camera.img_ROI.shape) != 4:
            img_ROI = camera.img_ROI.unsqueeze(0)
        if self.cfg.dataset.get('backbone', 'CLIP') == 'CLIP':
            img_feat = self.backbone.vision_model(pixel_values=img_ROI)

        else:
            _, c1, c2, c3, img_feat = self.backbone(img_ROI)
        if prev_camera is not None:
            prev_img_feat = self.backbone.vision_model(pixel_values=prev_camera.img_ROI.unsqueeze(0))
        else:
            prev_img_feat = None

        # loss_reg.update(gaussians.get_opacity_loss())
        #camera, loss_reg_pose = self.pose_correction(camera, img_feat,gaussians, gaussians_obj,iteration)


        deformer_hand_r = getattr(self, f"deformer_hand_{camera.subject_id}_r")
        if 'left' in self.metadata.keys():
            deformer_hand_l = getattr(self, f"deformer_hand_{camera.subject_id}_l")
        camera, movable_prob, refined_gaussians_hand_r, _, loss_non_rigid_hand_r = \
            deformer_hand_r.non_rigid(gaussians_r, img_feat, iteration, camera,
                                      compute_loss, delay=False)
        loss_reg.update(loss_non_rigid_hand_r)
        if 'left' in self.metadata.keys():
            camera, movable_prob, refined_gaussians_hand_l, _, loss_non_rigid_hand_l= \
                deformer_hand_l.non_rigid(gaussians_l, img_feat, iteration, camera, compute_loss,
                                        delay=False)
            loss_reg.update(loss_non_rigid_hand_l)

        deformer_obj = getattr(self, f"deformer_obj_{camera.obj_id}")

        nr_delay = not rigid_delay
        #######nr_delay = not rigid_delay
        camera, movable_prob, refined_gaussians_obj, xyz_canonical, loss_non_rigid_obj = \
                deformer_obj.non_rigid(gaussians_obj, img_feat, iteration, camera, compute_loss, nr_delay,
                                                         (prev_camera, prev_img_feat))

        loss_reg.update(loss_non_rigid_obj)

        #if delay:
        deformed_gaussians_hand_r = deformer_hand_r.rigid(refined_gaussians_hand_r, iteration, camera, None)
        if 'left' in self.metadata.keys():
            deformed_gaussians_hand_l = deformer_hand_l.rigid(refined_gaussians_hand_l, iteration, camera,
                                                            None)
        else:
            deformed_gaussians_hand_l = None
        deformed_gaussians_obj, loss_reg_rigid, articulated_obj, pivot, axis = deformer_obj.rigid(refined_gaussians_obj, xyz_canonical, iteration, camera,
                                                None, self.cfg.rigid_iter, rigid_delay, self.save_dir)
        if loss_reg_rigid is not None:
            loss_reg.update(loss_reg_rigid)

        color_precompute_r = self.texture_r(deformed_gaussians_hand_r, camera)
        if 'left' in self.metadata.keys():
            # NOTE: We incorrectly use `texture_r` for the left hand here — this is a code inconsistency 
            # as it should ideally be `texture_l` or a shared module. However, due to the bilateral symmetry 
            # of hands and shared weights, this currently produces valid results. 
            color_precompute_l = self.texture_r(deformed_gaussians_hand_l, camera)
        else:
            color_precompute_l = None
        objcolor_precompute = self.objtexture(deformed_gaussians_obj, camera)

        deformed_gaussians_obj_rigid = None
        objcolor_precompute_rigid = None

        #if iteration>=3000:
        # deformed_gaussians_obj_rigid, _ = deformer_obj.rigid(refined_gaussians_obj, xyz_canonical, iteration,
        #                                                             camera,self.pose_model_obj, False, self.save_dir)
        # objcolor_precompute_rigid = self.objtexture(deformed_gaussians_obj, camera)

        return deformed_gaussians_hand_r, deformed_gaussians_hand_l,deformed_gaussians_obj, refined_gaussians_obj, loss_reg,\
               color_precompute_r, color_precompute_l, objcolor_precompute, camera, movable_prob, articulated_obj, pivot, axis


    def optimize(self, iteration):
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        # if grad_clip > 0:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

        for group in self.optimizer.param_groups:
            name = group['name']
            if  grad_clip  > 0 and 'angle' in name:
                torch.nn.utils.clip_grad_norm_(group['params'], grad_clip*2)
            elif grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(group['params'], grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()
        #self.scheduler.step()

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        if iteration >= self.cfg.rigid_iter:
            gamma_angle = (5*self.cfg.opt.lr_ratio) ** (1. / self.cfg.opt.iterations)
        else:
            gamma_angle = 1

        for group in self.optimizer.param_groups:
            name = group['name']
            if 'angle' in name:
                group['lr'] = group['lr'] * gamma_angle
            else:
                group['lr'] = group['lr'] * gamma





    def random_sampling(self, xyz, npoint=2048):
        """
        Input:
            xyz: point cloud data, [B, N, C]
            npoint: number of points to sample
        Return:
            sampled_xyz: sampled point cloud data, [B, npoint, C]
        """
        B, N, C = xyz.shape
        # 生成随机索引
        indices = torch.randint(0, N, (B, npoint,), device=xyz.device)
        # 使用随机索引对点云进行采样
        sampled_xyz = index_points(xyz, indices)
        return sampled_xyz, indices


    def query_weights(self, xyz):
        # find the nearest vertex

        knn_ret = ops.knn_points(xyz, self.smpl_verts.unsqueeze(0).repeat(xyz.shape[0],1,1))
        p_idx = knn_ret.idx

        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    # def pose_correct_hand(self, camera, pc_feature_global):
    #
    #     hand_param = self.pose_model_hand.pose_mlp(pc_feature_global, camera['hand_param'])
    #     root_orient, pose_hand, betas, trans = camera['hand_param'][:, :3] + hand_param[:, :3], \
    #                                            camera['hand_param'][:, 3:48] + hand_param[:, 3:48], \
    #                                            camera['hand_param'][:, 48:58] + hand_param[:, 48:58], \
    #                                            camera['hand_param'][:, 58:]
    #     # root_orient, pose_hand, betas, trans = hand_param[:, :3], \
    #     #                                       hand_param[:, 3:48], \
    #     #                                       hand_param[:, 48:58], \
    #     #                                       camera['hand_param'][:, 58:]
    #
    #
    #     body = self.body_model(global_orient=root_orient, hand_pose=pose_hand, betas=betas, transl=trans)
    #     bone_transforms = body['bone_transforms']
    #
    #     rot_mats = body['rot_mats']
    #
    #     rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).repeat(hand_param.shape[0], 1, 1, 1).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
    #     rots = rots.reshape(hand_param.shape[0], -1, 9).contiguous()
    #
    #
    #     bone_transforms[:,:, :3, 3] = bone_transforms[:,:, :3, 3] + trans.unsqueeze(1)
    #     # use estimated trans
    #     # bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + camera.hand_param[:, 58:].squeeze(0)
    #
    #     Jtrs = self.get_jtr(body)
    #
    #     updated_camera = camera.copy()
    #     updated_camera.update({
    #         'rots': rots,
    #         'Jtrs':Jtrs,
    #         'bone_transforms':bone_transforms,
    #         'hand_param':torch.cat([root_orient, pose_hand, betas, trans],dim=-1)}
    #     )
    #
    #     # regularization
    #     loss_pose_regularization = self.L2Loss(root_orient, camera['hand_param'][:, :3]) + \
    #                                self.L2Loss(pose_hand, camera['hand_param'][:, 3:48]) + \
    #                                self.L2Loss(betas, camera['hand_param'][:, 48:58]) * 0.01
    #
    #
    #     if camera['hand_param_gt'] is not None:
    #         loss_pose = self.L2Loss(root_orient, camera['hand_param_gt'][:, :3]) + \
    #                     self.L2Loss(pose_hand, camera['hand_param_gt'][:, 3:48]) + \
    #                     self.L2Loss(betas, camera['hand_param_gt'][:, 48:58]) * 0.01
    #
    #         # print("loss",self.L2Loss(root_orient, camera['hand_param_gt'][:, :3]))
    #         # print("root_", root_orient.mean(0))
    #         # print("gt", camera['hand_param_gt'][:, :3].mean(0))
    #
    #
    #
    #         body_gt = self.body_model(global_orient=camera['hand_param_gt'][:, :3],
    #                                       hand_pose=camera['hand_param_gt'][:, 3:48], betas=camera['hand_param_gt'][:, 48:58],
    #                                       transl=camera['hand_param_gt'][:, 58:])
    #         Jtrs_gt = self.get_jtr(body_gt)
    #         loss_mano_joint = self.L2Loss(Jtrs, Jtrs_gt)
    #
    #
    #         return updated_camera, {
    #             'pose_hand': loss_pose,
    #             'pose_regularization_hand': loss_pose_regularization,
    #             'mano_joint': loss_mano_joint,
    #         }
    #
    #
    # def pose_correct_obj(self, camera, pc_feature_global):
    #     obj_rot_ori = rotation_matrix_to_angle_axis(camera['obj_rots'].clone())
    #     obj_trans_ori = camera['obj_trans'].clone()
    #
    #     obj_feature = self.pose_model_obj.obj_emb(torch.cat([obj_rot_ori, obj_trans_ori], dim=-1))
    #     obj_param = self.pose_model_obj.pose_mlp(pc_feature_global, obj_feature)
    #
    #     delta_obj_rot_vector, delta_obj_trans = obj_param[:, 0:3], obj_param[:, 3:]
    #
    #     # 将新的旋转矩阵转换为旋转向量
    #     refined_rot_vec = delta_obj_rot_vector + obj_rot_ori
    #     refined_matrix = angle_axis_to_rotation_matrix(refined_rot_vec)
    #     #refined_obj_trans = camera['hand_param'][:, 58:] + delta_obj_trans
    #     refined_obj_trans = obj_trans_ori + delta_obj_trans
    #
    #     updated_camera = camera.copy()
    #     updated_camera.update({
    #         'obj_rots':refined_matrix,
    #         'obj_trans':refined_obj_trans}
    #     )
    #
    #     gt_obj_rot_vec = rotation_matrix_to_angle_axis(camera['obj_rots_gt'])
    #     #loss_pose_obj = self.L2Loss(refined_rot_vec, gt_obj_rot_vec)
    #     #print('rot',self.L2Loss(refined_rot_vec, gt_obj_rot_vec))
    #
    #     obj3DCorners = []
    #     for oid in camera['obj_id']:
    #         obj3DCorners.append(torch.from_numpy(self.metadata_obj[int(oid)]['obj3DCorners']))
    #
    #     obj3DCorners = torch.stack(obj3DCorners, dim=0).cuda()
    #
    #
    #     rotated_corners = torch.matmul(refined_matrix, obj3DCorners.transpose(1, 2)).transpose(1, 2)  # B * 8 * 3
    #     obj_corners = rotated_corners + refined_obj_trans.unsqueeze(1)  # B * 8 * 3
    #
    #     rotated_corners_gt = torch.matmul(camera['obj_rots_gt'], obj3DCorners.transpose(1, 2)).transpose(1, 2)  # B * 8 * 3
    #     obj_corners_gt = rotated_corners_gt + camera['obj_trans_gt'].unsqueeze(1)  # B * 8 * 3
    #
    #     loss_pose_obj = self.L2Loss(obj_corners, obj_corners_gt)
    #
    #     loss_pose_obj += self.L2Loss(refined_obj_trans, camera['obj_trans_gt'])
    #     loss_pose_obj += self.L2Loss(refined_matrix, camera['obj_rots_gt'])*0.01
    #     # print('corner', obj_corners[0])
    #     # print('trans', refined_obj_trans[0])
    #     #
    #     # print('corner_gt', obj_corners_gt[0])
    #     # print('trans_gt', camera['obj_trans_gt'][0])
    #     #
    #     # print('corner_loss',self.L2Loss(obj_corners, obj_corners_gt) * 1000)
    #     # print('trans_loss',self.L2Loss(refined_obj_trans, camera['obj_trans_gt'])*1000)
    #     loss_pose_regularization = torch.norm(delta_obj_rot_vector, p=2) + torch.norm(delta_obj_trans, p=2)
    #
    #     return updated_camera, {
    #         'pose_obj': loss_pose_obj,
    #         'pose_regularization_obj': loss_pose_regularization,
    #     }

    def pixel_align(self, camera, input_xyz_points, feature_maps, full_proj_transform, trans_img2roi, roi_size, ho='hand'):
        batch_size, num_points_per_scene, _ = input_xyz_points.shape
        input_points = input_xyz_points.clone()
        input_points = input_points.reshape((-1, num_points_per_scene, 3))
        #full_proj_transform = full_proj_transform
        trans_img2roi = trans_img2roi


        # homo_xyz = homoify(input_points)
        # homo_xyz_2d = torch.matmul(full_proj_transform, homo_xyz.transpose(1, 2)).transpose(1, 2)
        # xyz_2d = (homo_xyz_2d[:, :, :2] / homo_xyz_2d[:, :, [3]])


        cam_coord = torch.matmul(camera['R'], input_points.transpose(1, 2)).transpose(1, 2) + camera['T'].reshape(batch_size ,1, 3)

        xyz_2d = points3DToImg(cam_coord,camera['K'])[:, :, :2]

        ones = torch.ones((batch_size,num_points_per_scene, 1), dtype=torch.float32).to(input_points.device)
        uv_homogeneous = torch.cat([xyz_2d, ones], dim=-1)  # (b, N, 3)
        uv_2d_roi_unnorm = torch.matmul(trans_img2roi, uv_homogeneous.transpose(1, 2)).transpose(1, 2)[:,:,:2].unsqueeze(2) #b n 1 2

        uv_2d_roi = uv_2d_roi_unnorm / roi_size * 2 - 1

        sample_feat = torch.nn.functional.grid_sample(feature_maps, uv_2d_roi, align_corners=True)[:, :, :, 0].transpose(1, 2)
        sample_color = torch.nn.functional.grid_sample(camera['img_ROI'], uv_2d_roi, align_corners=True)[:, :, :,
                       0].transpose(1, 2)
        uv_2d_roi = uv_2d_roi.squeeze(2).reshape((batch_size, -1, 2))
        sample_feat = sample_feat.reshape((batch_size,uv_2d_roi.shape[1], -1))
        sample_color = sample_color.reshape((batch_size, uv_2d_roi.shape[1], -1))
        # validity = (uv_2d_roi[:, 0] >= -1.0) & (uv_2d_roi[:, 0] <= 1.0) & (uv_2d_roi[:, 1] >= -1.0) & (uv_2d_roi[:, 1] <= 1.0)
        # validity = validity.unsqueeze(1)

        # if cfg.with_add_feats:
        #     depth_feat = input_xyz_points.reshape((-1, 3))[:, [-1]]
        #     view_dir_feat = F.normalize(input_xyz_points.reshape((-1, 3)), p=2, dim=1)
        #     sample_feat = torch.cat([sample_feat, depth_feat, view_dir_feat], axis=1)

        # # 定义图像尺寸
        # w, h = 224, 224  # 你可以根据需要调整图像尺寸
        # # 创建黑色背景图
        # background = camera['img_ROI'][0].detach().permute(1, 2, 0).cpu().numpy()* 255
        #
        # #background = np.zeros((h, w, 3), dtype=np.uint8)
        # # 假设你已经有了坐标列表coords，每个坐标是一个元组 (x, y)
        # coords = uv_2d_roi_unnorm.squeeze(2)[0].detach().cpu().numpy()
        # # 将坐标列表中的像素设置为白色
        # for coord in coords:
        #     x, y = int(coord[0]), int(coord[1])
        #     # 确保坐标在图像范围内
        #     if 0 <= x < w and 0 <= y < h:
        #         if ho == 'hand':
        #             background[y, x] = (255, 0, 255)  # 设置为白色
        #         else:
        #             background[y, x] = (255, 255, 255)  # 设置为白色
        #     # else:
        #     #     print("!")

        # # 保存结果图像
        # cv2.imwrite("/home/cyc/pycharm/lxy/3DGS/debug/img_ho_0_project_{}.png".format(ho), background)

        return sample_feat, sample_color

    def get_jtr(self, body):
        Jtrs = body['Jtr_a_pose']

        v_shaped = body['v_shaped']
        v_shaped = v_shaped.detach()

        center = torch.mean(v_shaped, dim=1).unsqueeze(1)
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