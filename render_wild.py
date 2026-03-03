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

import json
import os

from scene.cameras import QueryCamerasLoader
from utils.camera_utils import camera_to_JSON
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import shutil


# import hydra
import cv2
from omegaconf import OmegaConf
import swanlab


from right_hand_model import MANO
#import pyiqa # ok
from utils.loss_utils import l1_loss, ssim
#import pyiqa #  nan
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, PSEvaluator, save_axis, tensor_to_numpy_image, save_deltas
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss

import numpy as np
import torch
import torch.nn.functional as F

import pyiqa #  nan

import hydra
from random import randint


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value


#@profile
def training(config):
    model = config.model
    dataset = config.dataset
    pipe = config.pipeline
    gaussians_hand_group = {}
    gaussians_obj_group = {}


    for obj_id in dataset._YCB_CLASSES:
        gaussians_obj_group[obj_id] = None
    for subject in dataset._SUBJECTS:
        gaussians_hand_group[subject] = {'right':None}


    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id]['right'] = GaussianModel(model.gaussian)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id] = GaussianModel(model.gaussian)

    scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir)
    scene.eval()
    #scene.eval()
    print("training_samples:", len(scene.train_dataset))

    load_ckpt = config.checkpoint
    if load_ckpt is None:
        load_ckpt = os.path.join(config.exp_dir, 'ckpt25000.pth')
    print(load_ckpt)
    scene.load_checkpoint(load_ckpt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    validation(25001, (25001 <= config.rigid_iter), scene, (pipe, background))
        



def validation(iteration, rigid_delay, scene: Scene, renderArgs):
    scene.eval()
    torch.cuda.empty_cache()
    only_save_img = False
    
    print("test_samples:",len(scene.test_dataset))

    vis_dir = os.path.join(scene.save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    psnr_train = 0.0
    ssim_train = 0.0
    lpips_train = 0.0

    for idx, data in tqdm(enumerate(scene.train_dataset)):

        novel_cacmera = scene.test_dataset[idx]
        
        if only_save_img:
            full_gt_image = torch.clamp(data.full_image.to("cuda"), 0.0, 1.0)
            full_gt_image_novel = torch.clamp(novel_cacmera.full_image.to("cuda"), 0.0, 1.0)
            cv2.imwrite(vis_dir+'/gt_{}.png'.format(idx),
                        cv2.cvtColor(np.uint8(full_gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255),
                                     cv2.COLOR_BGR2RGB))
            # cv2.imwrite(vis_dir+'/gt_novel_{}.png'.format(idx),
            #             cv2.cvtColor(np.uint8(full_gt_image_novel.permute(1, 2, 0).detach().cpu().numpy() * 255),
            #                          cv2.COLOR_BGR2RGB))
            continue
        novel_cacmera = None
        render_pkg = render(data, iteration+idx, scene, *renderArgs, compute_loss=True,
                            return_opacity=True, delay=rigid_delay, novel_data=novel_cacmera)
        examples = []

        movable_prob = render_pkg["movable_prob"]
        pivot = render_pkg["pivot"]
        axis = render_pkg["axis"]

        full_image = torch.clamp(render_pkg["full_render"], 0.0, 1.0)
        full_gt_image = torch.clamp(data.full_image.to("cuda"), 0.0, 1.0)
        #full_image_novel = torch.clamp(render_pkg["novel_render"], 0.0, 1.0)
        #full_gt_image_novel = torch.clamp(novel_cacmera.full_image.to("cuda"), 0.0, 1.0)

        if idx % 1 == 0:
            
            # cv2.imwrite(vis_dir+'/novel_{}.png'.format(idx),
            #                         cv2.cvtColor(np.uint8(full_image_novel.permute(1, 2, 0).detach().cpu().numpy() * 255),
            #                                      cv2.COLOR_BGR2RGB))
            cv2.imwrite(vis_dir+'/render_{}.png'.format(idx),
                        cv2.cvtColor(np.uint8(full_image.permute(1, 2, 0).detach().cpu().numpy() * 255),
                                     cv2.COLOR_BGR2RGB))
            # cv2.imwrite(vis_dir+'/gt_novel_{}.png'.format(idx),
            #             cv2.cvtColor(np.uint8(full_gt_image_novel.permute(1, 2, 0).detach().cpu().numpy() * 255),
            #                          cv2.COLOR_BGR2RGB))
            cv2.imwrite(vis_dir+'/gt_{}.png'.format(idx),
                        cv2.cvtColor(np.uint8(full_gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255),
                                     cv2.COLOR_BGR2RGB))
        

        updated_camera = render_pkg['updated_camera']
       
        pc_obj = render_pkg['obj_deformed_gaussian']
        coord_min = torch.min(pc_obj._xyz.detach(), dim=0).values
        coord_max = torch.max(pc_obj._xyz.detach(), dim=0).values

        # 如果你希望使原点为中心（例如，将点云的中心移到原点），可以通过以下方式计算中心偏移量
        center = (coord_min + coord_max) / 2
        # 将点云的坐标移动到原点
        pc_obj._xyz -= center
        coord_min -= center
        coord_max -= center  
        pc_obj.save_ply(os.path.join(vis_dir, 'point_cloud','iteration_{}'.format(idx),'point_cloud.ply'), save_dynamic=False)

        if iter in []:
            pc_obj = scene.gaussians_obj_group[list(scene.gaussians_obj_group.keys())[0]]
            coord_min = torch.min(pc_obj._xyz.detach(), dim=0).values
            coord_max = torch.max(pc_obj._xyz.detach(), dim=0).values

            # 如果你希望使原点为中心（例如，将点云的中心移到原点），可以通过以下方式计算中心偏移量
            center = (coord_min + coord_max) / 2
            # 将点云的坐标移动到原点
            pc_obj._xyz -= center
            coord_min -= center
            coord_max -= center  
            pc_obj.save_ply(os.path.join(vis_dir, 'canonicalGS','iteration_{}'.format(idx),'point_cloud.ply'))

        if not rigid_delay:
        
            save_axis(render_pkg['obj_deformed_gaussian']._xyz.detach().cpu().numpy(), pivot.detach().cpu().numpy(), 
                    axis.detach().cpu().numpy(), os.path.join(vis_dir,'articulation','iteration_{}'.format(idx)))
            # render_pkg['pc_articulated'].save_ply(
            #                 os.path.join(vis_dir, 'articulation','iteration_{}'.format(idx),'point_cloud.ply'))
            render_pkg['pc_articulated'].save_parted_ply(
                            os.path.join(vis_dir, 'articulation','iteration_{}'.format(idx)))
            
            np.save(os.path.join(vis_dir,'articulation','iteration_{}'.format(idx),"pivot.npy"), pivot.detach().cpu().numpy())
            np.save(os.path.join(vis_dir,'articulation','iteration_{}'.format(idx),"axis.npy"), axis.detach().cpu().numpy())



        # generate camera
        if idx == 0:
            
            print(coord_min)
            print(coord_max)
            cams = QueryCamerasLoader(coord_min, coord_max, cam_num=512).get_cam
            json_cams = []
            for cam_id, cam in enumerate(cams):
                camera_entry = camera_to_JSON(cam_id, cam)
                
                json_cams.append(camera_entry)
            with open( os.path.join(vis_dir, 'cameras.json'), 'w') as file:
                json.dump(json_cams, file)

            print(updated_camera.K)
           

        # save mesh
        #mesh_path = os.path.join(vis_dir, 'obj_mesh_{}.obj'.format(idx))

    psnr_test /= len(scene.test_dataset)
    ssim_test /= len(scene.test_dataset)
    lpips_test /= len(scene.test_dataset)
    psnr_train /= len(scene.train_dataset)
    ssim_train /= len(scene.train_dataset)
    lpips_train /= len(scene.train_dataset)

    print("\n[ITER {}] Evaluating {}: lpips {} PSNR {} SSIM {}".format(iteration, 'test', lpips_test, psnr_test, ssim_test))
   
    torch.cuda.empty_cache()

    scene.train()


@hydra.main(version_base=None, config_path="configs", config_name="config_wild")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    config.exp_dir = config.get('exp_dir') or os.path.join('/mnt/sda1/lxy/ARGS_Wild_results/', config.dataset._YCB_CLASSES[0],config.name)
    config.dataset.white_background = True


    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)
    os.makedirs(os.path.join(config.exp_dir,'code'), exist_ok=True)
    if not config.wandb_disable:
        try:
            shutil.copyfile('train_wild.py', config.exp_dir + '/code/train_wild.py')
            shutil.copytree('./scene', config.exp_dir + '/code/scene')
            shutil.copytree('./models', config.exp_dir + '/code/models')
            shutil.copytree('./configs', config.exp_dir + '/code/configs')
            shutil.copytree('./dataset', config.exp_dir + '/code/dataset')
            shutil.copytree('./utils', config.exp_dir + '/code/utils')
        except Exception as e:
            print(f"[Warning] Failed to save codes: {e}")


    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)

    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()  #
