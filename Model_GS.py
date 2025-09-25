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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import torch
from transformers import CLIPModel

from gaussian_renderer import render, render_hand
from scipy.spatial.transform import Rotation
from scene.cameras import Camera
from scene import Scene, GaussianModel
from utils.general_utils import fix_random
from utils.graphics_utils import focal2fov
from right_hand_model.body_models import MANO

import hydra
from omegaconf import OmegaConf


class Model_GS(object):
    def __init__(self, config):
        model = config.model
        dataset = config.dataset
        self.gaussians_hand_group = {}
        self.gaussians_obj_group = {}
        for obj_id in dataset._YCB_CLASSES:
            self.gaussians_obj_group[int(obj_id)] = GaussianModel(model.gaussian)
        for subject in dataset._SUBJECTS:
            self.gaussians_hand_group[int(subject.split('-')[-1])] = GaussianModel(model.gaussian)

        self.gaussians = GaussianModel(model.gaussian)
        self.objgaussians = GaussianModel(model.gaussian)
        exp_dir = config.get('exp_dir') or os.path.join('/home/cyc/pycharm/lxy/3DGS/results', config.name)
        self.scene = Scene(config, self.gaussians_hand_group, self.gaussians_obj_group, exp_dir, multi_batch=False)
        self.scene.eval()

        self.scene.load_checkpoint(config.ckpt_dir, strict=False)

        #self.scene.converter.backbone = CLIPModel.from_pretrained("/home/cyc/pycharm/lxy/3DGS/lib/clip-vit-base-patch32/").cuda()

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.pipe = config.pipeline
        self.K = [[906.96, 0.0, 956.75], [0.0, 906.79, 547.23], [0.0, 0.0, 1.0]]
        self.R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        self.T = [[0.0], [0.0], [0.0]]

        self.W = 1920
        self.H = 1080
        self.w = 640
        self.h = 480

        self.body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/')  # .cuda()
        self.faces = np.load("/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/misc/faces.npz")['faces']
        self.J_regressor = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/J_regressors.npz')['rightHand']

    def get_in_the_wild_data(self, data_dict):
        K = np.array(self.K, dtype=np.float32).copy()
        R = np.array(self.R, np.float32)
        T = np.array(self.T, np.float32)

        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # compute posed mano hand

        trans = np.array(data_dict['handTrans']).reshape(1, -1)
        pose = np.array(data_dict['handPose'][3:]).reshape(1, -1)
        rot = np.array(data_dict['handPose'][:3]).reshape(1, -1)
        betas = np.array(data_dict['handBeta']).reshape(1, -1)
        rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
        pose_torch = torch.from_numpy(pose)  # .cuda()
        betas_torch = torch.from_numpy(betas)  # .cuda()
        new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1, 3]).astype(np.float32)
        new_trans = trans.reshape([1, 3]).astype(np.float32)
        new_root_orient_torch = torch.from_numpy(new_root_orient)  # .cuda()
        new_trans_torch = torch.from_numpy(new_trans)  # .cuda()

        body = self.body_model(betas=betas_torch)
        minimal_shape = body['v'].detach().cpu().numpy()[0]

        body = self.body_model(global_orient=new_root_orient_torch, hand_pose=pose_torch, betas=betas_torch,
                               transl=new_trans_torch)

        bone_transforms = body['bone_transforms'].detach().cpu().numpy()
        J_regressor = self.J_regressor
        Jtr = np.dot(J_regressor, minimal_shape)

        # Also get GT SMPL poses
        root_orient = data_dict['root_orient'].astype(np.float32)
        pose = data_dict['pose'].astype(np.float32)
        pose = np.concatenate([root_orient, pose], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))
        pose_mat_full = pose.as_matrix()
        pose_mat = pose_mat_full[1:, ...].copy()
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])
        ###

        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        bone_transforms1 = np.repeat(np.eye(4)[np.newaxis, ...], 16, axis=0)
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms1)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:, :3, 3] += trans

        obj_rot = np.eye(3)
        obj_trans = np.zeros((3, 1))

        return {"K": K, "R": R, "T": np.squeeze(T),
                "FoVx": FovX,
                "FoVy": FovY,
                "data_device": "cuda",
                "rots": torch.from_numpy(pose_rot).float().unsqueeze(0),
                "Jtrs": torch.from_numpy(Jtr_norm).float().unsqueeze(0),
                "bone_transforms": torch.from_numpy(bone_transforms),
                # obj params
                "obj_rots": torch.from_numpy(obj_rot).float().unsqueeze(0),
                "obj_trans": torch.from_numpy(obj_trans).float().unsqueeze(0)
                }

    def render(self, inputs):
        meta_info_file_list = []
        for input in inputs:
            data = self.get_in_the_wild_data(input)
            render_pkg = render(data, 0, self.scene, self.background, self.pipe, compute_loss=False,
                                return_opacity=True)

            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)
            obj_image = torch.clamp(render_pkg["obj_render"], 0.0, 1.0)
            obj_opacity_image = torch.clamp(render_pkg["obj_opacity_render"], 0.0, 1.0)
            full_image = torch.clamp(render_pkg["full_render"], 0.0, 1.0)
            full_opacity_image = torch.clamp(render_pkg["full_opacity_render"], 0.0, 1.0)

            meta_info = {
                "render_hand": image,
                "opacity_hand": opacity_image,
                "render_obj": obj_image,
                "opacity_obj": obj_opacity_image,
                "render_ho": full_image,
                "opacity_ho": full_opacity_image,
            }

            meta_info_file_list.append(meta_info)
        return meta_info_file_list, None


@hydra.main(version_base=None, config_path="configs", config_name="config_wild")
def main(config):
    print(OmegaConf.to_yaml(config))

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    #torch.cuda.set_device(0)
    model_gs = Model_GS(config)
    validation_configs = ({'name': 'test', 'cameras': list(range(len(model_gs.scene.test_dataset)))},)
    print("test_samples:", len(model_gs.scene.test_dataset))
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(model_gs.scene, config['name'] + '_dataset')[data_idx]

                # r
                #
                # image = torch.clamp(render_pkg["render"].permute(1,2,0), 0.0, 1.0)*255
                # full_image = torch.clamp(render_pkg["full_render"].permute(1,2,0), 0.0, 1.0)*255
                # full_gt_image = torch.clamp(data.full_image.to("cuda").permute(1,2,0), 0.0, 1.0)*255

                # cv2.imwrite('./outputs/render_hand_{}.png'.format(data.image_name),
                #             cv2.cvtColor(np.uint8(image.detach().cpu()), cv2.COLOR_BGR2RGB))
                # cv2.imwrite('./outputs/gt_{}.png'.format(data.image_name), cv2.cvtColor(np.uint8(full_gt_image.detach().cpu()), cv2.COLOR_BGR2RGB))
                # cv2.imwrite('./outputs/render_ho_{}.png'.format(data.image_name),
                #             cv2.cvtColor(np.uint8(full_image.detach().cpu()), cv2.COLOR_BGR2RGB))
                #print('load_ply'+'/home/cyc/pycharm/lxy/3DGS/PoseGS/outputs/canonical_hand_{}.ply'.format(data.subject_id))
                color = model_gs.scene.gaussians_hand_group[data.subject_id].load_colored_ply\
                    ('/home/cyc/pycharm/lxy/3DGS/PoseGS/outputs/canonical_hand_color_{}.ply'.format(data.subject_id))
                canonical_hand = model_gs.scene.gaussians_hand_group[data.subject_id]

                deformer_hand = getattr(model_gs.scene.converter, f"deformer_hand_{data.subject_id}")
                deformed_gaussians_hand = deformer_hand.rigid(canonical_hand, 360000, data,
                                                              model_gs.scene.converter.pose_model_hand)

                color_precompute = model_gs.scene.converter.texture(deformed_gaussians_hand, data)

                render_pkg = render_hand(data, deformed_gaussians_hand, model_gs.pipe, model_gs.background,
                                         colors_precomp=torch.from_numpy(color).float().cuda())
                image = torch.clamp(render_pkg["render"].permute(1,2,0), 0.0, 1.0)*255
                os.makedirs('./outputs/wild_lxy_colored_9/',exist_ok=True)
                cv2.imwrite('./outputs/wild_lxy_colored_9/render_hand_{}.png'.format(data.image_name),
                            cv2.cvtColor(np.uint8(image.detach().cpu()), cv2.COLOR_BGR2RGB))



    # handPose = [-5.0082648e-01, 2.8636034e+00, 1.0859632e+00, 1.0412924e-01,
    #             - 1.3629951e-01, 8.0177844e-02, 1.8768187e-01, 0.0000000e+00,
    #             - 5.5683022e-03, 0.0000000e+00, 0.0000000e+00, 1.5664257e-01,
    #             0.0000000e+00, 8.5665613e-02, 9.4764054e-02, 1.9926904e-01,
    #             0.0000000e+00, 1.4938904e-01, 0.0000000e+00, 0.0000000e+00,
    #             2.2401641e-01, 2.0685506e-03, 2.7393934e-01, 3.6708921e-01,
    #             0.0000000e+00, 1.8089482e-03, 7.3866338e-02, 0.0000000e+00,
    #             0.0000000e+00, 1.0983388e-01, 3.9926848e-01, 1.1577919e-01,
    #             2.6811796e-01, 2.1300247e-01, 0.0000000e+00, 1.1372937e-02,
    #             0.0000000e+00, 0.0000000e+00, 1.9638607e-01, 7.3896486e-01,
    #             - 1.7991401e-01, 5.0306249e-01, 1.5101859e-01, 0.0000000e+00,
    #             - 5.9254896e-03, 0.0000000e+00, 2.3616867e-03, 4.7636814e-02]
    # handBeta = [-0.8101279, 0.77720827, 1.9707527, 0.35107753, 0.64986867, 2.711966,
    #             1.1160069, 0.6333117, 0.75000185, 0.4505857]
    # handTrans = [-0.17006603, 0.03036311, 0.41327086]
    #
    # inputs = [{'handPose': handPose, 'handBeta': handBeta, 'handTrans': handTrans}]
    #
    # render_pkg = model_gs.render(inputs)[0]
    # cv2.imwrite('./output/render_hand.png', cv2.cvtColor(np.uint8(render_pkg['render_hand']), cv2.COLOR_BGR2RGB))
    # cv2.imwrite('./output/opacity_hand.png', render_pkg['opacity_hand'])
    # cv2.imwrite('./output/render_obj.png', cv2.cvtColor(np.uint8(render_pkg['render_obj']), cv2.COLOR_BGR2RGB))
    # cv2.imwrite('./output/opacity_ho.png', render_pkg['opacity_ho'])


if __name__ == "__main__":
    main()
