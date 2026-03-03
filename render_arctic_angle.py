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
import math
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
        gaussians_hand_group[subject] = {'right':None, 'left':None,}


    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id]['right'] = GaussianModel(model.gaussian)
        gaussians_hand_group[sub_id]['left'] = GaussianModel(model.gaussian)
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
    only_save_img = True
    
    print("test_samples:",len(scene.test_dataset))

    vis_dir = os.path.join(scene.save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    psnr_train = 0.0
    ssim_train = 0.0
    lpips_train = 0.0
    obj_radians_gt = []
    for idx, data in tqdm(enumerate(scene.train_dataset)):
        
        novel_cacmera = scene.test_dataset[idx]
        obj_radian_gt = novel_cacmera.obj_radian
        print(float(obj_radian_gt))
        obj_radians_gt.append(float(obj_radian_gt))
        
        
        #novel_cacmera = None
        render_pkg = render(data, iteration+idx, scene, *renderArgs, compute_loss=True,
                            return_opacity=True, delay=rigid_delay, novel_data=novel_cacmera)
       

        if idx == len(scene.train_dataset)-1:
            angle_history = getattr(scene.converter, 'deformer_obj_{}'.format(
                    list(scene.gaussians_obj_group.keys())[0])).rigid.angle_history
            
            frames = sorted(angle_history.keys())
            frame_angles = []
            aae = []
            aae_inv = []
            for f in frames:
                angle = float(angle_history[f].detach())
                angle -= float(angle_history[frames[0]].detach())
                pred_radian = angle
                frame_angles.append(pred_radian)
                #print(f,pred_radian-obj_radians_gt[f])

                pred_degree = pred_radian / math.pi * 180  # degree
                gt_degree = (obj_radians_gt[f]-obj_radians_gt[0]) / math.pi * 180  # degree

                err_deg = np.abs(pred_degree - gt_degree).tolist()
                err_deg_inverse =  np.abs(pred_degree + gt_degree)
                
                aae.append(np.array(err_deg, dtype=np.float32))
                aae_inv.append(np.array(err_deg_inverse, dtype=np.float32))
                print(err_deg)
                    
            np.save(os.path.join(vis_dir,'articulation',"angle.npy"), np.array(frame_angles))
            summary_filename = os.path.join(vis_dir,'articulation', "eval_articulated.txt")

            with open(summary_filename, "w") as f:
                aae = "AAE : {}\n".format(min(np.mean(aae),np.mean(aae_inv)))
                print(aae); f.write(aae)



        

    psnr_test /= len(scene.test_dataset)
    ssim_test /= len(scene.test_dataset)
    lpips_test /= len(scene.test_dataset)
    psnr_train /= len(scene.train_dataset)
    ssim_train /= len(scene.train_dataset)
    lpips_train /= len(scene.train_dataset)

    print("\n[ITER {}] Evaluating {}: lpips {} PSNR {} SSIM {}".format(iteration, 'test', lpips_test, psnr_test, ssim_test))
   
    torch.cuda.empty_cache()

    scene.train()


@hydra.main(version_base=None, config_path="configs", config_name="config_arctic")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    config.exp_dir = config.get('exp_dir') or os.path.join('/mnt/sda2/lxy/ARGS_results/', config.dataset._YCB_CLASSES[0],config.name)
    config.dataset.white_background = True


    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)
    os.makedirs(os.path.join(config.exp_dir,'code'), exist_ok=True)
    if not config.wandb_disable:
        try:
            shutil.copyfile('train_arctic.py', config.exp_dir + '/code/train_arctic.py')
            shutil.copyfile('cocoify_arctic.py', config.exp_dir + '/code/cocoify_arctic.py')
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
