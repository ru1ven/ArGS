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

import os
import time

import torch
from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset


class Scene:

    #gaussians : GaussianModel

    def __init__(self, cfg, gaussians_hand_group, gaussians_obj_group, save_dir : str,  multi_batch=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians_hand_group = gaussians_hand_group
        self.gaussians_obj_group = gaussians_obj_group

        self.train_dataset = load_dataset(cfg.dataset, split='train', multi_batch=multi_batch)
        self.metadata = self.train_dataset.metadata
        self.metadata_obj = self.train_dataset.metadata_obj

        if cfg.dataset.name == 'dexycb':
            self.test_dataset_6k = load_dataset(cfg.dataset, split='test', test_split='SDF', multi_batch=multi_batch)
            self.test_dataset_25k = load_dataset(cfg.dataset, split='test', test_split='HOISDF', multi_batch=multi_batch)
        else:
            self.test_dataset = load_dataset(cfg.dataset, split='test', multi_batch=multi_batch)


        self.cameras_extent = self.metadata['right']['cameras_extent']
        for obj_id in self.gaussians_obj_group:
            self.gaussians_obj_group[obj_id].create_from_pcd(self.train_dataset.randomPointCloud(obj_id), spatial_lr_scale=self.cameras_extent)
        for sub_id in self.gaussians_hand_group:
            self.gaussians_hand_group[sub_id]['right'].create_from_pcd(self.train_dataset.readPointCloud(sub_id, 'right'), spatial_lr_scale=self.cameras_extent)
            if 'left' in self.gaussians_hand_group[sub_id].keys():
                self.gaussians_hand_group[sub_id]['left'].create_from_pcd(self.train_dataset.readPointCloud(sub_id, 'left'),
                                                          spatial_lr_scale=self.cameras_extent)
        self.converter = GaussianConverter(cfg, save_dir, self.metadata, self.metadata_obj, self.gaussians_hand_group.keys(), self.gaussians_obj_group.keys()).cuda()

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    # from memory_profiler import profile
    # @profile
    def optimize(self, iteration):

        for sub_id in self.gaussians_hand_group:
            self.gaussians_hand_group[sub_id]['right'].optimizer.step()
            if 'left' in self.gaussians_hand_group[sub_id].keys():
                self.gaussians_hand_group[sub_id]['left'].optimizer.step()
        for obj_id in self.gaussians_obj_group:
            self.gaussians_obj_group[obj_id].optimizer.step()
        for sub_id in self.gaussians_hand_group:
            self.gaussians_hand_group[sub_id]['right'].optimizer.zero_grad(set_to_none=True)
            if 'left' in self.gaussians_hand_group[sub_id].keys():
                self.gaussians_hand_group[sub_id]['left'].optimizer.zero_grad(set_to_none=True)
        for obj_id in self.gaussians_obj_group:
            self.gaussians_obj_group[obj_id].optimizer.zero_grad(set_to_none=True)
        self.converter.optimize(iteration)



    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True, delay=False, prev_data=None):
        # if pose_refine:
        #     return self.convert_gaussians_pose_refine(viewpoint_camera, iteration, compute_loss)
        return self.converter(self.gaussians_hand_group[viewpoint_camera.subject_id]['right'],
                              self.gaussians_hand_group[viewpoint_camera.subject_id]['left'] if 'left' in self.gaussians_hand_group[viewpoint_camera.subject_id].keys() else None,
                                  self.gaussians_obj_group[viewpoint_camera.obj_id], viewpoint_camera, iteration,
                                  compute_loss, delay, prev_camera=prev_data)

    # def convert_gaussians_pose_refine(self, viewpoint_camera, iteration, compute_loss=True):
    #     # 生成随机索引
    #
    #     if self.converter.training:
    #         indices = [torch.randint(0, self.gaussians_hand_group[int(sid)].get_xyz.shape[0],
    #                                  (self.cfg.get('sample_pc',2048),), device="cuda") for sid in viewpoint_camera['subject_id']]
    #         indices_obj = [torch.randint(0, self.gaussians_obj_group[int(oid)].get_xyz.shape[0],
    #                                  (self.cfg.get('sample_pc',2048),), device="cuda") for oid in viewpoint_camera['obj_id']]
    #     else:
    #         indices = [range(self.gaussians_hand_group[int(sid)].get_xyz.shape[0]) for sid in viewpoint_camera['subject_id']]
    #         indices_obj = [range(self.gaussians_obj_group[int(oid)].get_xyz.shape[0]) for oid in viewpoint_camera['obj_id']]
    #
    #     obj_gaussians = torch.stack([torch.cat([self.gaussians_obj_group[int(oid)].get_xyz[indices_obj[i], :],
    #                                             self.gaussians_obj_group[int(oid)].get_covariance()[indices_obj[i], :],
    #                                             self.gaussians_obj_group[int(oid)].get_features.reshape(-1, 48)[
    #                                             indices_obj[i], :],
    #                                             self.gaussians_obj_group[int(oid)].get_opacity[indices_obj[i], :]],
    #                                            dim=-1) for i, oid in enumerate(viewpoint_camera['obj_id'])],
    #                                 dim=0).cuda()
    #
    #     hand_gaussians = torch.stack([torch.cat([self.gaussians_hand_group[int(sid)].get_xyz[indices[i], :],
    #                                              self.gaussians_hand_group[int(sid)].get_covariance()[indices[i],
    #                                              :],
    #                                              self.gaussians_hand_group[int(sid)].get_features.reshape(-1, 48)[
    #                                              indices[i], :],
    #                                              self.gaussians_hand_group[int(sid)].get_opacity[indices[i], :]],
    #                                             dim=-1)
    #                                   for i, sid in enumerate(viewpoint_camera['subject_id'])], dim=0).cuda()
    #
    #     loss_reg, updated_camera, deltas_hand,deltas_obj, color_precompute, objcolor_precompute = \
    #         self.converter(hand_gaussians, obj_gaussians, viewpoint_camera, iteration, compute_loss,pose_refine=True)
    #     color_precompute, objcolor_precompute = color_precompute.squeeze(0), objcolor_precompute.squeeze(0)
    #     if self.converter.training:
    #         return loss_reg, updated_camera
    #
    #     assert viewpoint_camera['subject_id'].shape[0] == 1
    #     deformer_hand = getattr(self.converter, f"deformer_hand_{int(viewpoint_camera['subject_id'][0])}")
    #     deformer_obj = getattr(self.converter, f"deformer_obj_{int(viewpoint_camera['obj_id'][0])}")
    #
    #     updated_camera = Viewpoint_data(updated_camera)
    #
    #     # 1
    #     #print('1')
    #     refined_gaussians_hand = self.gaussians_hand_group[int(updated_camera.subject_id)].clone()
    #     refined_gaussians_obj = self.gaussians_obj_group[int(updated_camera.obj_id)].clone()
    #     refined_gaussians_hand = deformer_hand.non_rigid.apply_non_rigid_trans(
    #         self.gaussians_hand_group[int(updated_camera.subject_id)], refined_gaussians_hand,deltas_hand.squeeze(0))
    #     refined_gaussians_obj = deformer_obj.non_rigid.apply_non_rigid_trans(
    #         self.gaussians_obj_group[int(updated_camera.obj_id)], refined_gaussians_obj,deltas_obj.squeeze(0))
    #     deformed_gaussians_hand = deformer_hand.rigid(refined_gaussians_hand, iteration, updated_camera, self.converter.pose_model_hand)
    #     deformed_gaussians_obj = deformer_obj.rigid(refined_gaussians_obj, iteration, updated_camera, self.converter.pose_model_obj)
    #
    #
    #     #color_precompute = self.converter.texture(deformed_gaussians_hand, data)
    #     #objcolor_precompute = self.converter.objtexture(deformed_gaussians_obj, data)
    #
    #     # 2
    #     # print('2')
    #     # deformed_gaussians_hand, deformed_gaussians_obj, loss_reg, \
    #     # _, _, updated_camera, _, _ = self.converter(self.gaussians_hand_group[int(updated_camera.subject_id)],
    #     #                       self.gaussians_obj_group[int(updated_camera.obj_id)], updated_camera, iteration,
    #     #                       compute_loss, delay=False)
    #
    #     return deformed_gaussians_hand, deformed_gaussians_obj, loss_reg, \
    #            color_precompute, objcolor_precompute, updated_camera, refined_gaussians_hand, refined_gaussians_obj



    def get_canonical_gaussians(self, viewpoint_camera):
        _, _, _, color_precompute, objcolor_precompute, _, refined_gaussians_hand, refined_gaussians_obj \
            = self.converter(self.gaussians_hand_group[viewpoint_camera.subject_id],
                             self.gaussians_obj_group[viewpoint_camera.obj_id], viewpoint_camera,
                             self.cfg.model.gaussian.get('until', 360000),compute_loss=False, delay=False)
        return refined_gaussians_hand, refined_gaussians_obj, color_precompute, objcolor_precompute

    def get_skinning_loss(self, subject_id):
        loss_reg_r = getattr(self.converter,f"deformer_hand_{subject_id}_r").rigid.regularization()
        loss_skinning = loss_reg_r.get('loss_skinning', torch.tensor(0.).cuda())
        if hasattr(self.converter, f"deformer_hand_{subject_id}_l"):
            loss_reg_l = getattr(self.converter, f"deformer_hand_{subject_id}_l").rigid.regularization()
            loss_skinning += loss_reg_l.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        for sub_id in self.gaussians_hand_group:
            self.gaussians_hand_group[sub_id]['right'].save_ply(os.path.join(point_cloud_path, "point_cloud_hand_{}_r.ply").format(sub_id))
            if 'left' in self.gaussians_hand_group[sub_id].keys():
                self.gaussians_hand_group[sub_id]['left'].save_ply(
                os.path.join(point_cloud_path, "point_cloud_hand_{}_l.ply").format(sub_id))
        for obj_id in self.gaussians_obj_group:
            self.gaussians_obj_group[obj_id].save_ply(os.path.join(point_cloud_path, "point_cloud_obj_{}.ply".format(obj_id)))

    def save_canonical(self, iteration, pc_hand, sub_id, pc_obj, obj_id):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/canonical_iteration_{}".format(iteration))
        pc_hand.save_ply(os.path.join(point_cloud_path, "canonical_hand_{}.ply").format(sub_id))
        pc_obj.save_ply(os.path.join(point_cloud_path, "canonical_obj_{}.ply".format(obj_id)))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        if 'left' in self.gaussians_hand_group[list(self.gaussians_hand_group.keys())[0]].keys():
            torch.save(({sub_id+'right': self.gaussians_hand_group[sub_id]['right'].capture() for sub_id in self.gaussians_hand_group},
                    {sub_id + 'left': self.gaussians_hand_group[sub_id]['left'].capture() for sub_id in
                     self.gaussians_hand_group},
                    {obj_id: self.gaussians_obj_group[obj_id].capture() for obj_id in self.gaussians_obj_group},
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")
        else:
            torch.save(({sub_id+'right': self.gaussians_hand_group[sub_id]['right'].capture() for sub_id in self.gaussians_hand_group},
                    {obj_id: self.gaussians_obj_group[obj_id].capture() for obj_id in self.gaussians_obj_group},
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")


    def load_checkpoint(self, path, strict=True):
        if 'left' in self.gaussians_hand_group[list(self.gaussians_hand_group.keys())[0]].keys():
            (gaussian_params_r, gaussian_params_l, objgaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        else:
             (gaussian_params_r, objgaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        for sub_id in self.gaussians_hand_group:
            self.gaussians_hand_group[sub_id]['right'].restore(gaussian_params_r[sub_id+'right'], self.cfg.opt)
            if 'left' in self.gaussians_hand_group[sub_id].keys():
                self.gaussians_hand_group[sub_id]['left'].restore(gaussian_params_l[sub_id + 'left'], self.cfg.opt)
        for obj_id in self.gaussians_obj_group:
            self.gaussians_obj_group[obj_id].restore(objgaussian_params[obj_id], self.cfg.opt)

        missing_keys, unexpected_keys = self.converter.load_state_dict(converter_sd, strict=strict)

        if missing_keys:
            assert ("missing keys:", missing_keys)
        if unexpected_keys:
            print("ignored unexpected keys:")
            print(unexpected_keys)
        if strict:
            self.converter.optimizer.load_state_dict(converter_opt_sd)
            self.converter.scheduler.load_state_dict(converter_scd_sd)


    def load_ref_ckpt(self, path):
        (gaussian_params, objgaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        # for sub_id in self.gaussians_hand_group:
        #     self.gaussians_hand_group[sub_id].restore(gaussian_params[sub_id], self.cfg.opt)
        #print(self.converter.state_dict().keys())

        #self.converter.load_state_dict(converter_sd,True)
        self.converter.backbone.load_state_dict(converter_sd, False)

        #print(self.converter.deformer_obj_group)
        #self.converter.optimizer.load_state_dict(converter_opt_sd)
        #self.converter.scheduler.load_state_dict(converter_scd_sd)

        # for param in self.converter.backbone.parameters():
        #     param.requires_grad = False
