import numpy as np
import torch
import torch.nn as nn
import trimesh
from kornia.geometry import angle_axis_to_rotation_matrix

from manopth_utils.manopth.manolayer import ManoLayer
from right_hand_model import MANO
from utils.general_utils import get_jtr
from utils.loss_utils import compute_contact_loss
from utils.nets.config import cfg
from utils.nets.layer import MLP
from utils.nets.loss import ManoLoss, JointvoteLoss
from utils.nets.mano_head import ManoHead
from utils.nets.misc import get_mano_tgt_mask, get_mano_memory_mask
from utils.nets.transformer import Transformer, VoteTransformer
from utils.nets.transfusion_head import updatedDecoder


class KeypointTR(nn.Module):
    def __init__(self, config):
        super(KeypointTR, self).__init__()
        coord_change_mat = torch.tensor(
            [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32
        )
        self.mano_query_embed = nn.Embedding(cfg.mano_num_queries, cfg.hidden_dim)
        self.mano_layer = ManoLayer(
            ncomps=45,
            center_idx=0,
            flat_hand_mean=True,
            side="right",
            mano_root="/home/cyc/pycharm/lxy/HOISDF/tool/mano_models/",
            use_pca=False,
        )

        self.body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/')  # .cuda()

        self.mano_head = ManoHead(self.mano_layer, coord_change_mat=coord_change_mat)
        self.pose_fan_out = 3
        self.linear_pose = MLP(cfg.hidden_dim, cfg.hidden_dim, self.pose_fan_out, 3)

        self.linear_shape = MLP(cfg.hidden_dim, cfg.hidden_dim, 10, 3)

        self.linear_handvote = MLP(cfg.hidden_dim, cfg.hidden_dim, 20*3, 4)
        #self.linear_p2j = MLP(cfg.num_samp_hand, cfg.num_samp_hand // 4, 20, 4)
        self.linear_handcls = MLP(cfg.hidden_dim, cfg.hidden_dim, 20, 3)
        self.linear_objvote = MLP(cfg.hidden_dim, cfg.hidden_dim, 8 * 3, 4)
        self.linear_objcls = MLP(cfg.hidden_dim, cfg.hidden_dim, 8, 3)

        self.linear_obj_rel_trans = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)
        self.linear_obj_rot = MLP(cfg.hidden_dim, cfg.hidden_dim, 9, 3)

        self.hand_transformer = Transformer(
            d_model=cfg.hidden_dim,
            dropout=cfg.dropout,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            num_encoder_layers=cfg.enc_layers,
            num_decoder_layers=cfg.dec_layers,
            normalize_before=cfg.pre_norm,
            return_intermediate_dec=True,
        )

        self.obj_transformer = VoteTransformer(
            d_model=cfg.hidden_dim,
            dropout=cfg.dropout,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            num_encoder_layers=cfg.enc_layers // 2,
            normalize_before=cfg.pre_norm,
            return_intermediate_dec=True,
        )

        self.obj_rot_loss = nn.SmoothL1Loss(reduction="mean")
        self.obj_trans_loss = nn.SmoothL1Loss(reduction="mean")
        self.joints_vote_loss = JointvoteLoss()

        self.mano_loss = ManoLoss(
            lambda_verts3d=cfg.verts3d_weight,
            lambda_joints3d=cfg.joints3d_weight,
            lambda_manopose=cfg.manopose_weight,
            lambda_manoshape=cfg.manoshape_weight,
        )
        self.L2Loss = nn.SmoothL1Loss(reduction="mean").cuda()

        # self.crossTR_hand = updatedDecoder(joint_num=cfg.num_samp_hand,
        #                               hidden_channel=435,
        #                               num_heads=5,
        #                               ffn_channel=128,
        #                               dropout=0.1,
        #                               num_decoder_layers=4,
        #                               activation='relu')
        #
        # self.crossTR_obj = updatedDecoder(joint_num=cfg.num_samp_obj,
        #                                    hidden_channel=435,
        #                                    num_heads=5,
        #                                    ffn_channel=128,
        #                                    dropout=0.1,
        #                                    num_decoder_layers=4,
        #                                    activation='relu')

    def forward(self, camera, hand_points_posed, obj_points_posed, pixel_feat_h, gaussian_feat_h, pixel_feat_o, gaussian_feat_o, metadata_obj):

        hand_transformer_in = torch.cat([hand_points_posed, gaussian_feat_h, pixel_feat_h], dim=2)

        obj_transformer_in = torch.cat([obj_points_posed, gaussian_feat_o ,pixel_feat_o], dim=2)

        hand_transformer_in = hand_transformer_in.permute(1, 0, 2).contiguous()
        obj_transformer_in = obj_transformer_in.permute(1, 0, 2).contiguous()

        hand_positions = torch.zeros_like(hand_transformer_in).to(
            hand_transformer_in.device
        )
        obj_positions = torch.zeros_like(obj_transformer_in).to(
            obj_transformer_in.device
        )
        tgt_mask = get_mano_tgt_mask().to(hand_transformer_in.device)
        memory_mask = get_mano_memory_mask().to(hand_transformer_in.device)


        hand_transformer_out, memory, hand_encoder_out, attn_wts = (
            self.hand_transformer(
                src=hand_transformer_in,
                mask=None,
                pos_embed=hand_positions,
                src_mask=None,
                query_embed=self.mano_query_embed.weight,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
        )

        obj_memory, obj_encoder_out = self.obj_transformer(
            src=obj_transformer_in, mask=None, pos_embed=obj_positions, src_mask=None
        )


        obj_rot = self.linear_obj_rot(
            obj_encoder_out[:, : cfg.num_samp_obj]
        )  # 6 x N x 3
        obj_trans = self.linear_obj_rel_trans(obj_encoder_out[:, : cfg.num_samp_obj])

        L, N ,B, _ = obj_rot.shape
        obj_rot = obj_rot.view(L, N, B, 3, 3)

        mano_pose6d = self.linear_pose(
                hand_transformer_out[:, : cfg.mano_shape_indx]
            )  # 6 x 16 x N x 3(9)

        mano_shape = self.linear_shape(
            hand_transformer_out[:, cfg.mano_shape_indx]
        )  # 6 x N x 10
        if self.pose_fan_out == 3:
            mano_pose6d = mano_pose6d+ camera.hand_param[:, :48].view(1, -1, 16, 3).permute(0, 2, 1, 3).contiguous().repeat(mano_pose6d.shape[0], 1, 1, 1)

        elif self.pose_fan_out == 9:
            mano_pose6d_init = angle_axis_to_rotation_matrix(camera.hand_param[:, :48].view(1, B, 16, 3).permute(0, 2, 1, 3).repeat(mano_pose6d.shape[0], 1, 1, 1).contiguous().view(-1, 3)).view(mano_pose6d.shape[0], 16, B, 9)
            mano_pose6d = mano_pose6d + mano_pose6d_init

        mano_shape = mano_shape + camera.hand_param[:, 48:58].view(1, -1, 10).repeat(mano_shape.shape[0], 1, 1)

        pred_mano_results, gt_mano_results = self.mano_head(
            mano_pose6d, mano_shape, mano_params=camera.hand_param_gt
        )


        root_orient, pose_hand, betas, hand_root = pred_mano_results["mano_pose6d"][-1][:, :3] ,\
                                                   pred_mano_results["mano_pose6d"][-1][:, 3:48], \
                                                   pred_mano_results["mano_shape"][-1], \
                                                   camera.hand_param[:, 58:]
        body = self.body_model(global_orient=root_orient, hand_pose=pose_hand, betas=betas, transl=hand_root)
        bone_transforms = body['bone_transforms']

        rot_mats = body['rot_mats']

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).repeat(rot_mats.shape[0], 1, 1, 1).to(rot_mats.device),
                          rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(rot_mats.shape[0], -1, 9).contiguous()

        bone_transforms[:, :, :3, 3] = bone_transforms[:, :, :3, 3] + hand_root.unsqueeze(1)

        Jtrs = get_jtr(body)


        updated_camera = camera.copy()

        updated_camera.update(
            rots= rots.squeeze(0),
            Jtrs= Jtrs.squeeze(0),
            bone_transforms= bone_transforms.squeeze(0),
            hand_param= torch.cat([root_orient, pose_hand, betas, hand_root], dim=-1),
            obj_rots= obj_rot[-1].permute(1, 0, 2, 3).contiguous().mean(1).view(3,3)+camera.obj_rots,
            obj_trans= obj_trans[-1].permute(1, 0, 2).contiguous().mean(1).view(3)+camera.obj_trans,
            pred_joints_mano= pred_mano_results['joints3d'][-1],
            pred_joints=pred_mano_results['joints3d'][-1],
            gt_mano_joints= gt_mano_results['joints3d']
        )

        loss = {}

        # (
        #     loss["kp_mano_mesh"],
        #     loss["kp_mano_joint"],
        #     loss["kp_pose_param"],
        #     loss["kp_shape_param"],
        #     _,
        #     _,
        # ) = self.mano_loss(pred_mano_results, gt_mano_results)


        # loss["kp_obj_rot"] = self.obj_rot_loss(
        #     obj_rot, (updated_camera.obj_rots_gt.view(B, 3, 3)-camera.obj_rots.view(B, 3, 3)).unsqueeze(0).unsqueeze(0).expand_as(obj_rot)
        # )
        # loss["kp_obj_trans"] = self.obj_trans_loss(
        #     obj_trans,
        #     (updated_camera.obj_trans_gt.view(B, 3)-camera.obj_trans.view(B, 3)).unsqueeze(0).unsqueeze(0).expand_as(obj_trans),
        # )
        #
        # obj3DCorners = torch.from_numpy(metadata_obj[int(camera.obj_id)]['obj3DCorners']).to(obj_trans.device).view(B, 8, 3)
        #
        # rotated_corners = torch.matmul(updated_camera.obj_rots.view(B, 3, 3), obj3DCorners.transpose(1, 2)).transpose(1, 2)  # B * 8 * 3
        # obj_corners = rotated_corners + updated_camera.obj_trans.view(B, 1, 3) # B * 8 * 3
        #
        # rotated_corners_gt = torch.matmul(camera.obj_rots_gt.view(B, 3, 3), obj3DCorners.transpose(1, 2)).transpose(1,2)  # B * 8 * 3
        # obj_corners_gt = rotated_corners_gt + camera.obj_trans_gt.view(B, 1, 3)  # B * 8 * 3
        #
        # loss["kp_obj_corner"] = self.obj_trans_loss(obj_corners, obj_corners_gt)

        return updated_camera, loss

    def forward_pose_refine_ho(self, camera, hand_points_posed, obj_points_posed, pixel_feat_h, gaussian_feat_h, pixel_feat_o, gaussian_feat_o,metadata_obj):

        hand_transformer_in = torch.cat([hand_points_posed, gaussian_feat_h, pixel_feat_h], dim=2)

        obj_transformer_in = torch.cat([obj_points_posed, gaussian_feat_o ,pixel_feat_o], dim=2)

        # ho_contact = False
        # if ho_contact:
        #     hand_transformer_in = self.crossTR_hand(hand_transformer_in, obj_transformer_in).permute(0, 2, 1)
        #     obj_transformer_in = self.crossTR_obj(obj_transformer_in, hand_transformer_in).permute(0, 2, 1)

        hand_transformer_in = hand_transformer_in.permute(1, 0, 2).contiguous()
        obj_transformer_in = obj_transformer_in.permute(1, 0, 2).contiguous()

        hand_positions = torch.zeros_like(hand_transformer_in).to(
            hand_transformer_in.device
        )
        obj_positions = torch.zeros_like(obj_transformer_in).to(
            obj_transformer_in.device
        )
        tgt_mask = get_mano_tgt_mask().to(hand_transformer_in.device)
        memory_mask = get_mano_memory_mask().to(hand_transformer_in.device)


        hand_transformer_out, memory, hand_encoder_out, attn_wts = (
            self.hand_transformer(
                src=hand_transformer_in,
                mask=None,
                pos_embed=hand_positions,
                src_mask=None,
                query_embed=self.mano_query_embed.weight,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
        )

        obj_memory, obj_encoder_out = self.obj_transformer(
            src=obj_transformer_in, mask=None, pos_embed=obj_positions, src_mask=None
        )


        obj_rot = self.linear_obj_rot(
            obj_encoder_out[:, : cfg.num_samp_obj]
        )  # 6 x N x 3
        obj_trans = self.linear_obj_rel_trans(obj_encoder_out[:, : cfg.num_samp_obj])


        L, N ,B, _ = obj_rot.shape
        #obj_rot = angle_axis_to_rotation_matrix(obj_rot.view(-1,3)).view(L, N ,B, 3, 3)
        obj_rot = obj_rot.view(L, N, B, 3, 3)

        mano_pose6d = self.linear_pose(
                hand_transformer_out[:, : cfg.mano_shape_indx]
            )  # 6 x 16 x N x 3(9)

        mano_shape = self.linear_shape(
            hand_transformer_out[:, cfg.mano_shape_indx]
        )  # 6 x N x 10
        if self.pose_fan_out == 3:
            mano_pose6d_init = camera['hand_param'][:, :48].view(1, -1, 16, 3).permute(0, 2, 1, 3).repeat(mano_pose6d.shape[0], 1, 1, 1)
            mano_pose6d = mano_pose6d+mano_pose6d_init

        elif self.pose_fan_out == 9:
            mano_pose6d_init = angle_axis_to_rotation_matrix(camera['hand_param'][:, :48].view(1, B, 16, 3).permute(0, 2, 1, 3).repeat(mano_pose6d.shape[0], 1, 1, 1).contiguous().view(-1, 3)).view(mano_pose6d.shape[0], 16, B, 9)
            mano_pose6d = mano_pose6d + mano_pose6d_init
            #mano_pose6d = angle_axis_to_rotation_matrix(mano_pose6d_ori.contiguous().view(-1, 3)).view(mano_pose6d.shape[0], 16, B, 9)

        mano_shape_init = camera['hand_param'][:, 48:58].view(1, -1, 10).repeat(mano_shape.shape[0], 1, 1)
        mano_shape = mano_shape+mano_shape_init

        pred_mano_results, gt_mano_results = self.mano_head(
            mano_pose6d, mano_shape, mano_params=camera['hand_param_gt']
        )
        # mano_results_init, _ = self.mano_head(
        #     mano_pose6d_init, mano_shape_init, mano_params=None
        # )


        root_orient, pose_hand, betas, hand_root = pred_mano_results["mano_pose6d"][-1][:, :3] ,\
                                                   pred_mano_results["mano_pose6d"][-1][:, 3:48], \
                                                   pred_mano_results["mano_shape"][-1], \
                                                   camera['hand_param'][:, 58:]
        body = self.body_model(global_orient=root_orient, hand_pose=pose_hand, betas=betas, transl=hand_root)
        bone_transforms = body['bone_transforms']

        rot_mats = body['rot_mats']

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).repeat(rot_mats.shape[0], 1, 1, 1).to(rot_mats.device),
                          rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(rot_mats.shape[0], -1, 9).contiguous()

        bone_transforms[:, :, :3, 3] = bone_transforms[:, :, :3, 3] + hand_root.unsqueeze(1)
        # use estimated trans
        # bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + camera.hand_param[:, 58:].squeeze(0)

        Jtrs = get_jtr(body)


        updated_camera = camera.copy()
        updated_camera.update({
            'rots': rots,
            'Jtrs': Jtrs,
            'bone_transforms': bone_transforms,
            'hand_param': torch.cat([root_orient, pose_hand, betas, hand_root], dim=-1),
            'obj_rots': obj_rot[-1].permute(1, 0, 2, 3).contiguous().mean(1)+camera["obj_rots"],
            'obj_trans': obj_trans[-1].permute(1, 0, 2).contiguous().mean(1)+camera["obj_trans"],
            'pred_joints_mano': pred_mano_results['joints3d'][-1],
            'gt_mano_joints': gt_mano_results['joints3d'],
        })
        # Make all the predictions
        hand_off = self.linear_handvote(hand_encoder_out[:, : cfg.num_samp_hand])
        hand_cls = self.linear_handcls(hand_encoder_out[:, : cfg.num_samp_hand])

        loss = {}

        (
            loss["mano_mesh"],
            loss["mano_joint"],
            loss["pose_param"],
            loss["shape_param"],
            _,
            _,
        ) = self.mano_loss(pred_mano_results, gt_mano_results)

        # hand vote
        # (
        #     loss["loss_joint_3d"],
        #     loss["loss_joint_cls"],
        #     loss["loss_all_joint_3d"],
        #     hand_joints,
        # ) = self.joints_vote_loss(hand_points_posed, hand_off, hand_cls, gt_mano_results['joints3d'][:, 1:]*1000, pred_mano_results['joints3d'][-1][:, 1:])

        # updated_camera.update({
        #     'pred_joints': torch.cat([gt_mano_results['joints3d'][:, 0].unsqueeze(1), hand_joints[-1]], dim=1),
        # })

        loss["obj_rot"] = self.obj_rot_loss(
            obj_rot, (updated_camera["obj_rots_gt"]-camera["obj_rots"]).unsqueeze(0).unsqueeze(0).expand_as(obj_rot)
        )
        loss["obj_trans"] = self.obj_trans_loss(
            obj_trans,
            (updated_camera["obj_trans_gt"]-camera["obj_trans"]).unsqueeze(0).unsqueeze(0).expand_as(obj_trans),
        )
        #loss["obj_trans_regularization"] = torch.norm(obj_trans, p=2)


        obj3DCorners = []
        for oid in camera['obj_id']:
            obj3DCorners.append(torch.from_numpy(metadata_obj[int(oid)]['obj3DCorners']))

        obj3DCorners = torch.stack(obj3DCorners, dim=0).cuda()

        rotated_corners = torch.matmul(updated_camera['obj_rots'], obj3DCorners.transpose(1, 2)).transpose(1, 2)  # B * 8 * 3
        obj_corners = rotated_corners + updated_camera['obj_trans'].unsqueeze(1)  # B * 8 * 3

        rotated_corners_gt = torch.matmul(camera['obj_rots_gt'], obj3DCorners.transpose(1, 2)).transpose(1,2)  # B * 8 * 3
        obj_corners_gt = rotated_corners_gt + camera['obj_trans_gt'].unsqueeze(1)  # B * 8 * 3

        loss["obj_corner"] = self.obj_trans_loss(obj_corners,obj_corners_gt)

        #loss["regular"] = self.obj_trans_loss(mano_pose6d, mano_pose6d_ori)

        # feat similarity loss
        # l_visual = pixel_feat_h.clone().detach()  # n 1
        # l_pnt =gaussian_feat_h.clone().detach()  # n k
        # ce_logits = torch.cat([l_visual, l_pnt], dim=-1).view(-1,l_visual.shape[-1]+l_pnt.shape[-1])
        # ce_logits /= 0.07
        # labels = torch.zeros(ce_logits.shape[0], dtype=torch.long).to(l_visual.device)
        # loss["moco_loss"] = nn.CrossEntropyLoss()(ce_logits, labels)
        # l_visual = pixel_feat_o.clone().detach()  # n 1
        # l_pnt = gaussian_feat_o.clone().detach()  # n k
        # ce_logits = torch.cat([l_visual, l_pnt], dim=-1).view(-1,l_visual.shape[-1]+l_pnt.shape[-1])
        # ce_logits /= 0.07
        # labels = torch.zeros(ce_logits.shape[0], dtype=torch.long).to(l_visual.device)
        # loss["moco_loss"] += nn.CrossEntropyLoss()(ce_logits, labels)* 1e-4

        # hand-obj contact loss



        obj_points = []
        obj_triangles = []
        for oid in camera['obj_id']:
            obj_points.append(torch.from_numpy(metadata_obj[int(oid)]['obj_points']))
            obj_triangles.append(torch.from_numpy(metadata_obj[int(oid)]['obj_triangles']))
        obj_points = torch.stack(obj_points, dim=0).cuda()
        obj_triangles = torch.stack(obj_triangles, dim=0).cuda()

        obj_triangles = torch.matmul(updated_camera['obj_rots'],obj_triangles.view(B, -1, 3).transpose(1, 2)).transpose(1, 2) \
                        + updated_camera['obj_trans'].unsqueeze(1)
        obj_triangles = obj_triangles.view(B, -1, 3, 3)
        obj_points = torch.matmul(updated_camera['obj_rots'], obj_points.transpose(1, 2)).transpose(1, 2) + \
                     updated_camera['obj_trans'].unsqueeze(1)

        # debug
        # pred_hand = trimesh.Trimesh(vertices=torch.cat([body['v'][0],obj_points[0],obj_triangles.view(B,-1,3)[0]],dim=0).detach().cpu().numpy(), process=False)
        # pred_hand.export('/home/cyc/pycharm/lxy/3DGS/debug/pred_ho.obj')
        # print(camera['hand_param'][0][58:])
        # print(camera['hand_root'][0])
        # print(camera['obj_trans_gt'][0])
        # print(body['v'][0].mean(0))
        # exit()

        loss["contact"], loss["penetration"], _, _ = compute_contact_loss(
            body['v'],
            obj_points,
            obj_triangles,
            # contact_thresh=5 / 1000,
            # collision_thresh=20 / 1000,
            contact_thresh=10 / 1000,
            #contact_mode="dist_sq",
            collision_thresh=20 / 1000,
            #collision_mode="dist_sq",
            contact_zones="zones",
        )

        return updated_camera, loss

    def forward_pose_refine(self, camera, hand_points_posed, obj_points_posed, pixel_feat_h, gaussian_feat_h, pixel_feat_o, gaussian_feat_o,metadata_obj):


        obj_transformer_in = torch.cat([obj_points_posed, gaussian_feat_o ,pixel_feat_o], dim=2)

        obj_transformer_in = obj_transformer_in.permute(1, 0, 2).contiguous()

        obj_positions = torch.zeros_like(obj_transformer_in).to(
            obj_transformer_in.device
        )
        obj_memory, obj_encoder_out = self.obj_transformer(
            src=obj_transformer_in, mask=None, pos_embed=obj_positions, src_mask=None
        )

        obj_rot = self.linear_obj_rot(
            obj_encoder_out[:, : cfg.num_samp_obj]
        )  # 6 x N x 3
        obj_trans = self.linear_obj_rel_trans(obj_encoder_out[:, : cfg.num_samp_obj])


        L, N ,B, _ = obj_rot.shape

        obj_rot = obj_rot.view(L, N, B, 3, 3)

        root_orient, pose_hand, betas, hand_root =camera['hand_param'][:, :3] ,\
                                                   camera['hand_param'][:, 3:48], \
                                                   camera['hand_param'][:, 48:58], \
                                                   camera['hand_param'][:, 58:]
        body = self.body_model(global_orient=root_orient, hand_pose=pose_hand, betas=betas, transl=hand_root)
        bone_transforms = body['bone_transforms']

        rot_mats = body['rot_mats']

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).repeat(rot_mats.shape[0], 1, 1, 1).to(rot_mats.device),
                          rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(rot_mats.shape[0], -1, 9).contiguous()

        bone_transforms[:, :, :3, 3] = bone_transforms[:, :, :3, 3] + hand_root.unsqueeze(1)
        # use estimated trans
        # bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + camera.hand_param[:, 58:].squeeze(0)

        Jtrs = get_jtr(body)


        updated_camera = camera.copy()
        updated_camera.update({
            'rots': rots,
            'Jtrs': Jtrs,
            'bone_transforms': bone_transforms,
            'hand_param': torch.cat([root_orient, pose_hand, betas, hand_root], dim=-1),
            'obj_rots': obj_rot[-1].permute(1, 0, 2, 3).contiguous().mean(1)+camera["obj_rots"],
            'obj_trans': obj_trans[-1].permute(1, 0, 2).contiguous().mean(1)+camera["obj_trans"],
            'pred_joints_mano': Jtrs,
            'gt_mano_joints': Jtrs,
        })

        loss = {}

        loss["obj_rot"] = self.obj_rot_loss(
            obj_rot, (updated_camera["obj_rots_gt"]-camera["obj_rots"]).unsqueeze(0).unsqueeze(0).expand_as(obj_rot)
        )
        loss["obj_trans"] = self.obj_trans_loss(
            obj_trans,
            (updated_camera["obj_trans_gt"]-camera["obj_trans"]).unsqueeze(0).unsqueeze(0).expand_as(obj_trans),
        )
        #loss["obj_trans_regularization"] = torch.norm(obj_trans, p=2)


        obj3DCorners = []
        for oid in camera['obj_id']:
            obj3DCorners.append(torch.from_numpy(metadata_obj[int(oid)]['obj3DCorners']))

        obj3DCorners = torch.stack(obj3DCorners, dim=0).cuda()

        rotated_corners = torch.matmul(updated_camera['obj_rots'], obj3DCorners.transpose(1, 2)).transpose(1, 2)  # B * 8 * 3
        obj_corners = rotated_corners + updated_camera['obj_trans'].unsqueeze(1)  # B * 8 * 3

        rotated_corners_gt = torch.matmul(camera['obj_rots_gt'], obj3DCorners.transpose(1, 2)).transpose(1,2)  # B * 8 * 3
        obj_corners_gt = rotated_corners_gt + camera['obj_trans_gt'].unsqueeze(1)  # B * 8 * 3

        loss["obj_corner"] = self.obj_trans_loss(obj_corners,obj_corners_gt)

        obj_points = []
        obj_triangles = []
        for oid in camera['obj_id']:
            obj_points.append(torch.from_numpy(metadata_obj[int(oid)]['obj_points']))
            obj_triangles.append(torch.from_numpy(metadata_obj[int(oid)]['obj_triangles']))
        obj_points = torch.stack(obj_points, dim=0).cuda()
        obj_triangles = torch.stack(obj_triangles, dim=0).cuda()

        obj_triangles = torch.matmul(updated_camera['obj_rots'],obj_triangles.view(B, -1, 3).transpose(1, 2)).transpose(1, 2) \
                        + updated_camera['obj_trans'].unsqueeze(1)
        obj_triangles = obj_triangles.view(B, -1, 3, 3)
        obj_points = torch.matmul(updated_camera['obj_rots'], obj_points.transpose(1, 2)).transpose(1, 2) + \
                     updated_camera['obj_trans'].unsqueeze(1)

        loss["contact"], loss["penetration"], _, _ = compute_contact_loss(
            body['v'],
            obj_points,
            obj_triangles,
            # contact_thresh=5 / 1000,
            # collision_thresh=20 / 1000,
            contact_thresh=10 / 1000,
            #contact_mode="dist_sq",
            collision_thresh=20 / 1000,
            #collision_mode="dist_sq",
            contact_zones="zones",
        )

        return updated_camera, loss

