import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation

import models
from right_hand_model import MANO
from .lbs import lbs
# from models.network_utils import get_mlp
from ..network_utils import VanillaCondMLP, get_projected_uvd
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix
import pytorch3d.ops as ops



def get_transforms_02v(Jtr):
    device = Jtr.device

    from scipy.spatial.transform import Rotation as R
    rot45p = torch.tensor(R.from_euler('z', 45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    rot45n = torch.tensor(R.from_euler('z', -45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(24, 1, 1)

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    R_02v_l = []
    t_02v_l = []
    chain = [1, 4, 7, 10]
    rot = rot45p
    for i, j_idx in enumerate(chain):
        R_02v_l.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_l[i-1]

        t_02v_l.append(t)

    R_02v_l = torch.stack(R_02v_l, dim=0)
    t_02v_l = torch.stack(t_02v_l, dim=0)
    t_02v_l = t_02v_l - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_l = F.pad(R_02v_l, (0, 0, 0, 1))  # 4 x 4 x 3
    t_02v_l = F.pad(t_02v_l, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_l, t_02v_l.unsqueeze(-1)], dim=-1)

    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    R_02v_r = []
    t_02v_r = []
    chain = [2, 5, 8, 11]
    rot = rot45n
    for i, j_idx in enumerate(chain):
        # bone_transforms_02v[j_idx, :3, :3] = rot
        R_02v_r.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_r[i-1]

        t_02v_r.append(t)

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    R_02v_r = torch.stack(R_02v_r, dim=0)
    t_02v_r = torch.stack(t_02v_r, dim=0)
    t_02v_r = t_02v_r - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_r = F.pad(R_02v_r, (0, 0, 0, 1))  # 4 x 3
    t_02v_r = F.pad(t_02v_r, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_r, t_02v_r.unsqueeze(-1)], dim=-1)

    return bone_transforms_02v

class NoPoseCorrection(nn.Module):
    def __init__(self, config, metadata=None):
        super(NoPoseCorrection, self).__init__()

    def forward(self, camera, iteration):
        return camera, {}

    def regularization(self, out):
        return {}

class PoseCorrection(nn.Module):
    def __init__(self, config, metadata=None):
        super(PoseCorrection, self).__init__()

        self.config = config
        self.metadata = metadata

        self.frame_dict = metadata['frame_dict']

        v_template = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/v_templates.npz')['rightHand']
        lbs_weights = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/skinning_weights_all.npz')['rightHand']
        posedirs = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/posedirs_all.npz')['rightHand']
        posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
        shapedirs = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/shapedirs_all.npz')['rightHand']
        J_regressor = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/J_regressors.npz')['rightHand']
        kintree_table = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/kintree_table.npy')

        self.register_buffer('v_template', torch.tensor(v_template, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32))
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights, dtype=torch.float32))
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

    def forward_smpl(self, betas, root_orient, pose_hand, trans):
        full_pose = torch.cat([root_orient,pose_hand], dim=-1)
        verts_posed, Jtrs_posed, Jtrs, bone_transforms, _, v_posed, v_shaped, rot_mats = lbs(betas=betas,
                                                                                             pose=full_pose,
                                                                                             v_template=self.v_template.clone(),
                                                                                             clothed_v_template=None,
                                                                                             shapedirs=self.shapedirs.clone(),
                                                                                             posedirs=self.posedirs.clone(),
                                                                                             J_regressor=self.J_regressor.clone(),
                                                                                             parents=self.kintree_table[
                                                                                                 0].long(),
                                                                                             lbs_weights=self.lbs_weights.clone(),
                                                                                             dtype=torch.float32)

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(1, -1, 9).contiguous()

        #bone_transforms_02v = get_transforms_02v(Jtrs.squeeze(0))

        #bone_transforms = torch.matmul(bone_transforms.squeeze(0), torch.inverse(bone_transforms_02v))
        bone_transforms = bone_transforms.squeeze(0)
        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + trans

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

        verts_posed = verts_posed + trans[None]

        return rots, Jtrs, bone_transforms, verts_posed, v_posed, Jtrs_posed

    def forward(self, camera, img_feat, pc_hand, pc_obj, iteration):
        frame = camera.frame_id
        if frame not in self.frame_dict:
            return camera, {}
        return self.pose_correct(camera, img_feat,pc_hand, pc_obj, iteration)

    def regularization(self, out):
        raise NotImplementedError

    def pose_correct(self, camera, img_feat, pc_hand, pc_obj,iteration):
        raise NotImplementedError

class DirectPoseOptimization(PoseCorrection):
    def __init__(self, config, metadata=None):
        super(DirectPoseOptimization, self).__init__(config, metadata)
        self.cfg = config

        root_orient = metadata['root_orient']
        pose_hand = metadata['pose_hand']
        trans = metadata['trans']
        betas = metadata['betas']
        frames = metadata['frames']

        self.frames = frames

        # use nn.Embedding
        root_orient = np.array(root_orient)
        pose_hand = np.array(pose_hand)
        trans = np.array(trans)
        self.root_orients = nn.Embedding.from_pretrained(torch.from_numpy(root_orient).float(), freeze=False)
        self.pose_hands = nn.Embedding.from_pretrained(torch.from_numpy(pose_hand).float(), freeze=False)
        self.trans = nn.Embedding.from_pretrained(torch.from_numpy(trans).float(), freeze=False)

        self.register_parameter('betas', nn.Parameter(torch.tensor(betas, dtype=torch.float32)))


    def pose_correct(self, camera, img_feat,pc_hand, pc_obj,iteration):
        if iteration < self.cfg.get('delay', 0):
            return camera, {}

        frame = camera.frame_id

        # use nn.Embedding
        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        betas = self.betas

        # compose rots, Jtrs, bone_transforms, posed_smpl_verts
        rots, Jtrs, bone_transforms, posed_smpl_verts, _, _ = self.forward_smpl(betas, root_orient, pose_hand, trans)

        rots_diff = camera.rots - rots
        updated_camera = camera.copy()
        updated_camera.update(
            rots=rots,
            Jtrs=Jtrs,
            bone_transforms=bone_transforms,
        )

        loss_pose = (rots_diff ** 2).mean()
        return updated_camera, {
            'pose': loss_pose,
        }

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    def export(self, frame):
        model_dict = {}

        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        betas = self.betas

        rots, Jtrs, bone_transforms, posed_smpl_verts, v_posed, Jtr_posed = self.forward_smpl(betas, root_orient,
                                                                                pose_hand, trans)
        model_dict.update({
            'minimal_shape': v_posed[0],
            'betas': betas,
            'Jtr_posed': Jtr_posed[0],
            'bone_transforms': bone_transforms,
            'trans': trans[0],
            'root_orient': root_orient[0],
            'pose_hand': pose_hand[0],
        })
        for k, v in model_dict.items():
            model_dict.update({k: v.detach().cpu().numpy()})
        return model_dict



class RelativePoseOptimization(PoseCorrection):
    def __init__(self, config, metadata=None):
        super(RelativePoseOptimization, self).__init__(config, metadata)
        self.cfg = config

        #self.pose_mlp = VanillaCondMLP(45+3+10+3, 768, 45+3+10+3, config.mlp)
        obj_d_cond = 128
        feature_d_cond = 128
        self.pose_mlp_hand = VanillaCondMLP(45 + 3 + 10 + 3 , feature_d_cond, 45 + 3 + 10 + 3, config.mlp)
        self.feature_fc = nn.Linear(768, feature_d_cond)
        self.obj_emb = nn.Linear(6, obj_d_cond)
        self.pose_mlp_obj = VanillaCondMLP(obj_d_cond, feature_d_cond, 6, config.mlp)
        self.L2Loss = torch.nn.MSELoss().cuda()
        self.body_model_pca = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/',use_pca=True,num_pca_comps=48,flat_hand_mean=False)#.cuda()

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

    def pose_correct(self, camera, img_feat, pc_hand, pc_obj,iteration):
        if iteration < self.cfg.get('delay', 0):
            return camera, {}
        if torch.is_tensor(img_feat):
            pose_feat_global = img_feat.mean(3).mean(2)
        else:
            pose_feat_global = img_feat.pooler_output
        obj_rot_ori = rotation_matrix_to_angle_axis(camera.obj_rots.clone())
        obj_trans_ori = camera.obj_trans.clone()
        pose_feat_global = self.feature_fc(pose_feat_global)
        hand_param = self.pose_mlp_hand(camera.hand_param, pose_feat_global)
        obj_feature = self.obj_emb(torch.cat([obj_rot_ori,obj_trans_ori], dim=-1))
        obj_param = self.pose_mlp_obj(obj_feature,pose_feat_global)

        root_orient,pose_hand,betas,trans = camera.hand_param[:,:3]+hand_param[:,:3],\
                                            camera.hand_param[:,3:48]+hand_param[:,3:48],\
                                            camera.hand_param[:,48:58]+hand_param[:,48:58],\
                                            camera.hand_param[:,58:]+hand_param[:,58:61]

        delta_obj_rot_vector,delta_obj_trans = obj_param[:,0:3],obj_param[:,3:]
        #delta_matrix = angle_axis_to_rotation_matrix(delta_obj_rot_vector)

        # 应用增量旋转
        #refined_matrix = torch.matmul(camera.obj_rots, delta_matrix)
        #refined_rot_vec = rotation_matrix_to_angle_axis(refined_matrix)

        # 将新的旋转矩阵转换为旋转向量

        refined_rot_vec = delta_obj_rot_vector+obj_rot_ori
        refined_matrix = angle_axis_to_rotation_matrix(refined_rot_vec)
        refined_obj_trans = camera.obj_trans+delta_obj_trans

        #root_orient, pose_hand, betas, trans = hand_param[:, :3], hand_param[:, 3:48],hand_param[:, 48:58], camera.hand_param[:,58:]+hand_param[:,58:]
        # compose rots, Jtrs, bone_transforms, posed_smpl_verts
        #rots, Jtrs, bone_transforms, posed_smpl_verts, _, _ = self.forward_smpl(betas, root_orient, pose_hand, trans)
        body = self.body_model_pca(global_orient=root_orient, hand_pose=pose_hand, betas=betas,transl=camera.hand_param[:, 58:])
        bone_transforms = body['bone_transforms']

        rot_mats = body['rot_mats']

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(1, -1, 9).contiguous()

        bone_transforms = bone_transforms.squeeze(0)
        #bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + trans
        # use estimated trans
        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + camera.hand_param[:, 58:].squeeze(0)

        Jtrs = self.get_jtr(body)

        updated_camera = camera.copy()
        updated_camera.update(
            rots=rots,
            Jtrs=Jtrs,
            bone_transforms=bone_transforms,
            obj_rots=refined_matrix,
            #obj_trans=refined_obj_trans
        )

        # regularization
        loss_pose_regularization = self.L2Loss(root_orient, camera.hand_param[:, :3]) + \
                                   self.L2Loss(pose_hand, camera.hand_param[:, 3:48]) + \
                                   self.L2Loss(betas, camera.hand_param[:, 48:58]) * 0.01
                                   #+self.L2Loss(trans, camera.hand_param[:, 58:])
        loss_pose_regularization += torch.norm(delta_obj_rot_vector, p=2) + torch.norm(delta_obj_trans, p=2)


        if camera.hand_param_gt is not None:
            loss_pose = self.L2Loss(root_orient, camera.hand_param_gt[:,:3])+\
                        self.L2Loss(pose_hand, camera.hand_param_gt[:, 3:48])+\
                        self.L2Loss(betas, camera.hand_param_gt[:,48:58])*0.01
                        #self.L2Loss(trans, camera.hand_param_gt[:,58:])

            body_gt = self.body_model_pca(global_orient=camera.hand_param_gt[:, :3],
                                          hand_pose=camera.hand_param_gt[:, 3:48], betas=camera.hand_param_gt[:, 48:58],
                                          transl=camera.hand_param_gt[:, 58:])
            Jtrs_gt = self.get_jtr(body_gt)
            loss_mano_joint = self.L2Loss(Jtrs, Jtrs_gt)


            gt_obj_rot_vec = rotation_matrix_to_angle_axis(camera.obj_rots_gt)
            loss_pose_obj =self.L2Loss(refined_rot_vec, gt_obj_rot_vec)
            #loss_pose += self.L2Loss(refined_obj_trans, camera.obj_trans_gt)


            return updated_camera, {
                'pose': loss_pose,
                'pose_obj': loss_pose_obj,
                'pose_regularization': loss_pose_regularization,
                'mano_joint':loss_mano_joint,
            }
        else:
            return updated_camera, {}

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    def export(self, frame):
        model_dict = {}

        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        betas = self.betas

        rots, Jtrs, bone_transforms, posed_smpl_verts, v_posed, Jtr_posed = self.forward_smpl(betas, root_orient,
                                                                                pose_hand, trans)
        model_dict.update({
            'minimal_shape': v_posed[0],
            'betas': betas,
            'Jtr_posed': Jtr_posed[0],
            'bone_transforms': bone_transforms,
            'trans': trans[0],
            'root_orient': root_orient[0],
            'pose_hand': pose_hand[0],
        })
        for k, v in model_dict.items():
            model_dict.update({k: v.detach().cpu().numpy()})
        return model_dict


class GlobalPoseOptimization(PoseCorrection):
    def __init__(self, config, metadata=None):
        super(GlobalPoseOptimization, self).__init__(config, metadata)
        self.cfg = config

        #self.pose_mlp = VanillaCondMLP(45+3+10+3, 768, 45+3+10+3, config.mlp)
        obj_d_cond = 128
        feature_d_cond = 128
        f_projected_cond = 128
        self.pose_mlp_hand = VanillaCondMLP(45 + 3 + 10 + 3 , feature_d_cond, 45 + 3 + 10 + 3, config.mlp)
        self.feature_fc = nn.Linear(768, feature_d_cond)
        self.obj_emb = nn.Linear(6, obj_d_cond)
        self.uvd_emb = nn.Linear(3, f_projected_cond)
        self.pose_mlp_obj = VanillaCondMLP(obj_d_cond, feature_d_cond, 6, config.mlp)
        self.L2Loss = torch.nn.MSELoss().cuda()
        self.body_model_pca = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/',use_pca=True,num_pca_comps=48,flat_hand_mean=False)#.cuda()

        self.roi_size = config.get('roi_size', 224)
        self.smpl_verts = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights = torch.from_numpy(metadata["skinning_weights"]).float().cuda()

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

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def pose_correct(self, camera,  img_feat, gaussians, gaussians_obj, iteration):
        if iteration < self.cfg.get('delay', 0):
            return camera, {}
        if torch.is_tensor(img_feat):
            pose_feat_global = img_feat.mean(3).mean(2)
        else:
            pose_feat_global = img_feat.pooler_output
        obj_rot_ori = rotation_matrix_to_angle_axis(camera.obj_rots.clone())
        obj_trans_ori = camera.obj_trans.clone()
        pose_feat_global = self.feature_fc(pose_feat_global)

        # # get pixel_aligned points
        # xyz_points_obj = gaussians_obj.get_xyz.clone()
        #
        # n_pts = xyz_points_obj.shape[0]
        # rot = camera.obj_rots
        # trans = camera.obj_trans
        # T_fwd = torch.eye(4).to(rot.device)
        # T_fwd[:3, :3] = rot.float().to(rot.device)
        # T_fwd[:3, 3] = trans
        # T_fwd = T_fwd.repeat(n_pts, 1, 1)
        #
        # homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz_points_obj.device)
        # x_hat_homo = torch.cat([xyz_points_obj, homo_coord], dim=-1).view(n_pts, 4, 1)
        # xyz_point_trans_obj = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        #
        # xyz_points = gaussians.get_xyz.clone()
        # bone_transforms = camera.bone_transforms
        # n_pts = xyz_points.shape[0]
        # pts_W = self.query_weights(xyz_points)
        # T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_pts, 4, 4).float()
        #
        # homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz_points.device)
        # x_hat_homo = torch.cat([xyz_points, homo_coord], dim=-1).view(n_pts, 4, 1)
        # xyz_point_trans_hand = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]

        # uvd_projected = get_projected_uvd(camera, torch.cat([xyz_point_trans_hand, xyz_point_trans_obj], dim=0),
        #                                           xyz_point_trans_hand.shape[0] + xyz_point_trans_obj.shape[0],
        #                                           camera.trans_img2roi, self.roi_size)
        # # mean
        # f_projected = self.uvd_emb(uvd_projected).mean(1)

        hand_param = self.pose_mlp_hand(camera.hand_param, pose_feat_global)
        obj_feature = self.obj_emb(torch.cat([obj_rot_ori,obj_trans_ori], dim=-1))
        obj_param = self.pose_mlp_obj(obj_feature, pose_feat_global)

        root_orient,pose_hand,betas,trans = camera.hand_param[:,:3]+hand_param[:,:3],\
                                            camera.hand_param[:,3:48]+hand_param[:,3:48],\
                                            camera.hand_param[:,48:58]+hand_param[:,48:58],\
                                            camera.hand_param[:,58:]+hand_param[:,58:61]

        delta_obj_rot_vector,delta_obj_trans = obj_param[:,0:3],obj_param[:,3:]

        # 将新的旋转矩阵转换为旋转向量

        refined_rot_vec = delta_obj_rot_vector+obj_rot_ori
        refined_matrix = angle_axis_to_rotation_matrix(refined_rot_vec)
        refined_obj_trans = camera.obj_trans+delta_obj_trans


        body = self.body_model_pca(global_orient=root_orient, hand_pose=pose_hand, betas=betas,transl=trans)
        bone_transforms = body['bone_transforms']

        rot_mats = body['rot_mats']

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(1, -1, 9).contiguous()

        bone_transforms = bone_transforms.squeeze(0)
        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + trans
        # use estimated trans
        #bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + camera.hand_param[:, 58:].squeeze(0)

        Jtrs = self.get_jtr(body)

        updated_camera = camera.copy()
        updated_camera.update(
            rots=rots,
            Jtrs=Jtrs,
            bone_transforms=bone_transforms,
            obj_rots=refined_matrix,
            obj_trans=refined_obj_trans
        )

        # regularization
        loss_pose_regularization = self.L2Loss(root_orient, camera.hand_param[:, :3]) + \
                                   self.L2Loss(pose_hand, camera.hand_param[:, 3:48]) + \
                                   self.L2Loss(betas, camera.hand_param[:, 48:58]) * 0.01+\
                                   self.L2Loss(trans, camera.hand_param[:, 58:])
        loss_pose_regularization += torch.norm(delta_obj_rot_vector, p=2) + torch.norm(delta_obj_trans, p=2)

        if camera.hand_param_gt is not None:
            loss_pose = self.L2Loss(root_orient, camera.hand_param_gt[:,:3])+\
                        self.L2Loss(pose_hand, camera.hand_param_gt[:, 3:48])+\
                        self.L2Loss(betas, camera.hand_param_gt[:,48:58])*0.01+\
                        self.L2Loss(trans, camera.hand_param_gt[:,58:])

            body_gt = self.body_model_pca(global_orient=camera.hand_param_gt[:, :3],
                                          hand_pose=camera.hand_param_gt[:, 3:48], betas=camera.hand_param_gt[:, 48:58],
                                          transl=camera.hand_param_gt[:, 58:])
            Jtrs_gt = self.get_jtr(body_gt)
            loss_mano_joint = self.L2Loss(Jtrs, Jtrs_gt)

            gt_obj_rot_vec = rotation_matrix_to_angle_axis(camera.obj_rots_gt)
            loss_pose_obj =self.L2Loss(refined_rot_vec, gt_obj_rot_vec)
            loss_pose_obj += self.L2Loss(refined_obj_trans, camera.obj_trans_gt)


            return updated_camera, {
                'pose': loss_pose,
                'pose_obj': loss_pose_obj,
                'pose_regularization': loss_pose_regularization,
                'mano_joint':loss_mano_joint,
            }

        else:
            return updated_camera, {}

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    def export(self, frame):
        model_dict = {}

        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        betas = self.betas

        rots, Jtrs, bone_transforms, posed_smpl_verts, v_posed, Jtr_posed = self.forward_smpl(betas, root_orient,
                                                                                pose_hand, trans)
        model_dict.update({
            'minimal_shape': v_posed[0],
            'betas': betas,
            'Jtr_posed': Jtr_posed[0],
            'bone_transforms': bone_transforms,
            'trans': trans[0],
            'root_orient': root_orient[0],
            'pose_hand': pose_hand[0],
        })
        for k, v in model_dict.items():
            model_dict.update({k: v.detach().cpu().numpy()})
        return model_dict


class HOPoseOptimization(PoseCorrection):
    def __init__(self, config, metadata=None):
        super(HOPoseOptimization, self).__init__(config, metadata)
        self.cfg = config

        #self.pose_mlp = VanillaCondMLP(45+3+10+3, 768, 45+3+10+3, config.mlp)
        obj_d_cond = 128
        feature_d_cond = 128
        f_projected_cond = 128
        self.pose_mlp_hand = VanillaCondMLP(45 + 3 + 10 + 3 , feature_d_cond, 45 + 3 + 10 + 3, config.mlp)
        self.feature_fc = nn.Linear(768, feature_d_cond)
        self.obj_emb = nn.Linear(6, obj_d_cond)
        self.uvd_emb = nn.Linear(3, f_projected_cond)
        self.pose_mlp_obj = VanillaCondMLP(obj_d_cond, feature_d_cond, 6, config.mlp)
        self.L2Loss = torch.nn.MSELoss().cuda()
        self.body_model_pca = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/',use_pca=True,num_pca_comps=48,flat_hand_mean=False)#.cuda()

        self.roi_size = config.get('roi_size', 224)
        self.smpl_verts = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights = torch.from_numpy(metadata["skinning_weights"]).float().cuda()

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

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def pose_correct(self, camera,  img_feat, gaussians, gaussians_obj, iteration):
        if iteration < self.cfg.get('delay', 0):
            return camera, {}
        if torch.is_tensor(img_feat):
            pose_feat_global = img_feat.mean(3).mean(2)
        else:
            pose_feat_global = img_feat.pooler_output
        obj_rot_ori = rotation_matrix_to_angle_axis(camera.obj_rots.clone())
        obj_trans_ori = camera.obj_trans.clone()
        pose_feat_global = self.feature_fc(pose_feat_global)

        # get pixel_aligned points
        # xyz_points_obj = gaussians_obj.get_xyz.clone()
        #
        # n_pts = xyz_points_obj.shape[0]
        # rot = camera.obj_rots
        # trans = camera.obj_trans
        # T_fwd = torch.eye(4).to(rot.device)
        # T_fwd[:3, :3] = rot.float().to(rot.device)
        # T_fwd[:3, 3] = trans
        # T_fwd = T_fwd.repeat(n_pts, 1, 1)
        #
        # homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz_points_obj.device)
        # x_hat_homo = torch.cat([xyz_points_obj, homo_coord], dim=-1).view(n_pts, 4, 1)
        # xyz_point_trans_obj = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]

        # xyz_points = gaussians.get_xyz.clone()
        # bone_transforms = camera.bone_transforms
        # n_pts = xyz_points.shape[0]
        # pts_W = self.query_weights(xyz_points)
        # T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_pts, 4, 4).float()
        #
        # homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz_points.device)
        # x_hat_homo = torch.cat([xyz_points, homo_coord], dim=-1).view(n_pts, 4, 1)
        # xyz_point_trans_hand = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]

        # uvd_projected = get_projected_uvd(camera, torch.cat([xyz_point_trans_hand, xyz_point_trans_obj], dim=0),
        #                                           xyz_point_trans_hand.shape[0] + xyz_point_trans_obj.shape[0],
        #                                           camera.trans_img2roi, self.roi_size)
        # mean
        #f_projected = self.uvd_emb(uvd_projected).mean(1)
        obj_trans_ho = obj_trans_ori-camera.hand_param[:,58:]
        hand_param = self.pose_mlp_hand(pose_feat_global, camera.hand_param)
        obj_feature = self.obj_emb(torch.cat([obj_rot_ori,obj_trans_ho],dim=-1))
        obj_param = self.pose_mlp_obj(obj_feature,pose_feat_global)

        root_orient,pose_hand,betas,trans = camera.hand_param[:,:3]+hand_param[:,:3],\
                                            camera.hand_param[:,3:48]+hand_param[:,3:48],\
                                            camera.hand_param[:,48:58]+hand_param[:,48:58],\
                                            camera.hand_param[:,58:]+hand_param[:,58:61]

        delta_obj_rot_vector,delta_obj_trans_ho = obj_param[:,0:3],obj_param[:,3:]
        # 将新的旋转矩阵转换为旋转向量

        refined_rot_vec = delta_obj_rot_vector+obj_rot_ori
        refined_matrix = angle_axis_to_rotation_matrix(refined_rot_vec)
        refined_obj_trans = camera.obj_trans+delta_obj_trans_ho


        body = self.body_model_pca(global_orient=root_orient, hand_pose=pose_hand, betas=betas,transl=camera.hand_param[:, 58:])
        bone_transforms = body['bone_transforms']

        rot_mats = body['rot_mats']

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(1, -1, 9).contiguous()

        bone_transforms = bone_transforms.squeeze(0)
        #bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + trans
        # use estimated trans
        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + camera.hand_param[:, 58:].squeeze(0)

        Jtrs = self.get_jtr(body)

        updated_camera = camera.copy()
        updated_camera.update(
            rots=rots,
            Jtrs=Jtrs,
            bone_transforms=bone_transforms,
            obj_rots=refined_matrix,
            obj_trans=refined_obj_trans
        )

        # regularization
        loss_pose_regularization = self.L2Loss(root_orient, camera.hand_param[:, :3]) + \
                                   self.L2Loss(pose_hand, camera.hand_param[:, 3:48]) + \
                                   self.L2Loss(betas, camera.hand_param[:, 48:58]) * 0.01
                                   #+self.L2Loss(trans, camera.hand_param[:, 58:])
        loss_pose_regularization += torch.norm(delta_obj_rot_vector, p=2) + torch.norm(delta_obj_trans_ho, p=2)

        if camera.hand_param_gt is not None:
            loss_pose = self.L2Loss(root_orient, camera.hand_param_gt[:,:3])+\
                        self.L2Loss(pose_hand, camera.hand_param_gt[:, 3:48])+\
                        self.L2Loss(betas, camera.hand_param_gt[:,48:58])*0.01
                        #self.L2Loss(trans, camera.hand_param_gt[:,58:])

            body_gt = self.body_model_pca(global_orient=camera.hand_param_gt[:, :3],
                                          hand_pose=camera.hand_param_gt[:, 3:48], betas=camera.hand_param_gt[:, 48:58],
                                          transl=camera.hand_param[:, 58:])
            Jtrs_gt = self.get_jtr(body_gt)
            loss_mano_joint = self.L2Loss(Jtrs, Jtrs_gt)

            gt_obj_rot_vec = rotation_matrix_to_angle_axis(camera.obj_rots_gt)
            loss_pose_obj =self.L2Loss(refined_rot_vec, gt_obj_rot_vec)
            loss_pose_obj += self.L2Loss(refined_obj_trans, camera.obj_trans_gt)


            return updated_camera, {
                'pose': loss_pose,
                'pose_obj': loss_pose_obj,
                'pose_regularization': loss_pose_regularization,
                'mano_joint':loss_mano_joint,
            }

        else:
            return updated_camera, {}

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    def export(self, frame):
        model_dict = {}

        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        betas = self.betas

        rots, Jtrs, bone_transforms, posed_smpl_verts, v_posed, Jtr_posed = self.forward_smpl(betas, root_orient,
                                                                                pose_hand, trans)
        model_dict.update({
            'minimal_shape': v_posed[0],
            'betas': betas,
            'Jtr_posed': Jtr_posed[0],
            'bone_transforms': bone_transforms,
            'trans': trans[0],
            'root_orient': root_orient[0],
            'pose_hand': pose_hand[0],
        })
        for k, v in model_dict.items():
            model_dict.update({k: v.detach().cpu().numpy()})
        return model_dict

def get_pose_correction(cfg, metadata):
    name = cfg.name
    model_dict = {
        "none": NoPoseCorrection,
        "direct": DirectPoseOptimization,
        "supervise": GlobalPoseOptimization,
    }
    return model_dict[name](cfg, metadata)