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
import copy
import gc
import os

from transformers import CLIPModel

from utils.network_utils import Pointnet2_Ssg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import tracemalloc

import timm
from memory_profiler import profile
from memory_profiler import memory_usage
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint

from torch import nn

from models.KeypointTR import KeypointTR
from models.network_utils import VanillaCondMLP
from right_hand_model import MANO
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator, cal_pose_error, relative_pose_error, \
    compute_obj_metrics_ycb, cfg_from_yaml_file
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss
import pyiqa
import hydra
from omegaconf import OmegaConf
import wandb
from submodules import lpips
import random
from torch.utils.data import DataLoader

from utils.pointbert.point_encoder import PointTransformer_Colored


@contextmanager
def profile_block():
    # Record memory usage before the block
    mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
    yield
    # Record memory usage after the block
    mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)
    print(f"Memory usage before: {mem_usage_before[0]} MiB")
    print(f"Memory usage after: {mem_usage_after[0]} MiB")
    print(f"Memory increment: {mem_usage_after[0] - mem_usage_before[0]} MiB")

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
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    debug_dir = '/mnt/sda1/lxy/debug_render/' +'final_demo_'+ config.tag+config.refine_tag
    os.makedirs(debug_dir, exist_ok=True)

    # generate obj_id and subject_id
    gaussians_hand_group = {}
    gaussians_obj_group = {}

    if dataset.name == 'dexycb':
        for obj_id in dataset._YCB_CLASSES:
            gaussians_obj_group[int(obj_id)] = None
        for subject in dataset._SUBJECTS:
            gaussians_hand_group[int(subject.split('-')[-1])] = None
    else:
        for view in dataset.train_view:
            subject_id = int(view.split('-')[0])
            obj_id = int(view.split('-')[1])
            gaussians_obj_group[obj_id] = None
            gaussians_hand_group[subject_id] = None

    evaluator = PSEvaluator()

    first_iter = 360000

    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id] = GaussianModel(model.gaussian)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id] = GaussianModel(model.gaussian)

    scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir, multi_batch=True)
    scene.eval()

   # scene.converter.backbone_refine = CLIPModel.from_pretrained("/home/cyc/pycharm/lxy/3DGS/lib/clip-vit-base-patch32/")
    scene.converter.pc_refine_setup()
    scene.load_checkpoint(config.ckpt_dir)


    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id].refine_setup(opt)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id].refine_setup(opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    data_stack = None
    ema_loss_for_log = 0.0

    iteration = first_iter + 1

    trainLoader = DataLoader(scene.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    #trainLoader = DataLoader(scene.train_dataset, batch_size=1, shuffle=False, num_workers=8)
    testLoader_6k = DataLoader(scene.test_dataset_6k, batch_size=config.test_batch_size, shuffle=False, num_workers=8)
    testLoader_25k = DataLoader(scene.test_dataset_25k, batch_size=config.test_batch_size, shuffle=False, num_workers=8)

    end_iter = trainLoader.__len__() * config.refine_epoch + first_iter + 1

    #scene.converter.pc_refine_setup(trainLoader.__len__() * config.refine_epoch)

    progress_bar = tqdm(range(first_iter, end_iter), desc="Training progress")


    validation(iteration, scene, evaluator, testLoader_25k,  scene.test_dataset_25k, debug_dir, (pipe, background))




def validation(iteration, scene: Scene, evaluator, testLoader, testset, debug_dir, renderArgs):

    scene.eval()
    torch.cuda.empty_cache()
    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/').cuda()

    validation_configs = ({'name': 'test', 'cameras': list(range(len(testset)))},)
    for config in validation_configs:

        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            mpjpe = 0.0
            mpjpe_mano = 0.0
            e_ADDS = 0.0
            e_MCE = 0.0
            e_OCE = 0.0
            render_eval_filter_num = 0
            examples = []
            for idx, data in enumerate(testLoader):
                if not (
                        idx <= 650 or
                        idx >= 6420 and idx <= 6740 or

                        idx >= 8930 and idx <= 9040 or

                        idx >= 10310 and idx <= 10560 or

                        idx >= 12270 and idx <= 12650 or

                        idx >= 14460 and idx <= 14950 or

                        idx >= 16650 and idx <= 16860 or

                        idx >= 24010 and idx <= 24260 or

                        idx > 22790 and idx < 23100):
                    continue



                data = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value
                        for key, value in data.items()}

                render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=True, return_opacity=True, pose_refine=True)
                updated_camera = render_pkg['updated_camera']
                examples = []

                full_image = torch.clamp(render_pkg["full_render"], 0.0, 1.0)
                full_gt_image = torch.clamp(updated_camera.full_image.to("cuda"), 0.0, 1.0)
                full_image_ori = torch.clamp(updated_camera.full_image_ori.to("cuda"), 0.0, 1.0)

                if idx >0:

                    full_render = cv2.cvtColor((full_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    full_ground_truth = cv2.cvtColor((full_gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    full_image_ori = cv2.cvtColor(
                        (full_image_ori.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8),
                        cv2.COLOR_BGR2RGB)

                    cv2.imwrite(os.path.join(debug_dir, '{}_full_render_{}.png'.format(idx, str(data['image_name'][0]))),full_render)
                    cv2.imwrite(os.path.join(debug_dir, '{}_full_ground_truth_{}.png'.format(idx, str(data['image_name'][0]))),
                                full_ground_truth)
                    cv2.imwrite(
                        os.path.join(debug_dir, '{}_full_ori_{}.png'.format(idx, str(data['image_name'][0]))),
                        full_image_ori)

                    render_ROI, _ = testset.generate_patch_image(full_image.permute(1, 2, 0).detach().cpu().numpy(),
                                                                 data['bbox'][0].detach().cpu().numpy(), [256, 256])
                    ground_truth_ROI, _ = testset.generate_patch_image(
                        full_gt_image.permute(1, 2, 0).detach().cpu().numpy(), data['bbox'][0].detach().cpu().numpy(),[256, 256])

                    cv2.imwrite(os.path.join(debug_dir, '{}_roi_render_{}.png'.format(idx, str(data['image_name'][0]))),
                                cv2.cvtColor((render_ROI * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(debug_dir, '{}_roi_ground_truth_{}.png'.format(idx, str(data['image_name'][0]))),
                        cv2.cvtColor((ground_truth_ROI * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

                # if idx % 40 == 0:
                #     # print(full_image.shape)
                #     # cv2.imwrite('/home/cyc/pycharm/lxy/3DGS/debug/refine_dexycb.png',cv2.cvtColor(np.uint8(full_image.permute(1,2,0).detach().cpu().numpy()*255), cv2.COLOR_BGR2RGB))
                #     # cv2.imwrite('/home/cyc/pycharm/lxy/3DGS/debug/refine_dexycb_gt.png',
                #     #             cv2.cvtColor(np.uint8(full_gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255),
                #     #                          cv2.COLOR_BGR2RGB))
                #
                #     wandb_img = wandb.Image(full_image[None],
                #                             caption=str(len(testset)) + "_view_{}/full_render".format(updated_camera.image_name))
                #     examples.append(wandb_img)
                #     wandb_img = wandb.Image(full_gt_image[None],
                #                             caption=str(len(testset)) + "_view_{}/full_ground_truth".format(
                #                                 updated_camera.image_name))
                #     examples.append(wandb_img)
                #
                #     #  cropped
                #     #  cropped
                #     render_ROI,_ = testset.generate_patch_image(full_image.permute(1, 2, 0).detach().cpu().numpy(),
                #                                               updated_camera.bbox.detach().cpu().numpy(), [256, 256])
                #     ground_truth_ROI,_ = testset.generate_patch_image(
                #         full_gt_image.permute(1, 2, 0).detach().cpu().numpy(), updated_camera.bbox.detach().cpu().numpy(),
                #         [256, 256])
                #     wandb_img = wandb.Image((render_ROI * 255).astype(np.uint8),
                #                             caption=str(len(testset)) + "_view_{}/roi_render".format(updated_camera.image_name))
                #     examples.append(wandb_img)
                #     wandb_img = wandb.Image((ground_truth_ROI * 255).astype(np.uint8),
                #                             caption=str(len(testset)) + "_view_{}/roi_ground_truth".format(
                #                                 updated_camera.image_name))
                #     examples.append(wandb_img)
                #
                #     wandb.log({config['name'] + "_images": examples})
                #     examples.clear()

                if config['name'] == 'test':
                    metrics = evaluator(full_image, full_gt_image)
                    #print(metrics)
                    if len(np.where(updated_camera.full_mask.cpu().numpy())[0]) == 0:
                        render_eval_filter_num += 1
                    else:
                        psnr_test += metrics['psnr'].cpu().item()
                        ssim_test += metrics['ssim'].cpu().item()
                        lpips_test += metrics['lpips'].cpu().item()

                    mpjpe += cal_pose_error(updated_camera, body_model)
                    mpjpe_mano += np.linalg.norm(
                        updated_camera.pred_joints_mano.detach().cpu().numpy() - updated_camera.gt_mano_joints.detach().cpu().numpy(),
                        axis=-1).mean()

                    OCE = relative_pose_error(render_pkg["updated_camera"])
                    ADDS, MCE = compute_obj_metrics_ycb(render_pkg["updated_camera"].obj_rots_gt.unsqueeze(0),
                                                            render_pkg["updated_camera"].obj_trans_gt.unsqueeze(0),
                                                            render_pkg["updated_camera"].obj_rots.unsqueeze(0),
                                                            render_pkg["updated_camera"].obj_trans.unsqueeze(0),
                                                            render_pkg["updated_camera"].obj_id)

                    # contact_resilt = eval_contact()

                    e_ADDS += ADDS.item()
                    e_MCE += MCE.item()
                    e_OCE += OCE.mean()

                    wandb.log({
                        config['name'] + '/psnr': metrics['psnr'].cpu().item(),
                        config['name'] + '/ssim': metrics['ssim'].cpu().item(),
                        config['name'] + '/lpips': metrics['lpips'].cpu().item(),
                        config['name'] + '/MCE': MCE.item(),
                    })

            psnr_test /= (len(config['cameras']) - render_eval_filter_num)
            ssim_test /= (len(config['cameras']) - render_eval_filter_num)
            lpips_test /= (len(config['cameras']) - render_eval_filter_num)
            l1_test /= (len(config['cameras']) - render_eval_filter_num)
            mpjpe /= len(config['cameras'])
            mpjpe_mano /= len(config['cameras'])
            e_ADDS /= len(config['cameras'])
            e_MCE /= len(config['cameras'])
            e_OCE /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: lpips {} PSNR {} MPJPE {} t_error {} ".format(iteration, config['name'], lpips_test,
                                                                                psnr_test, mpjpe, e_OCE))
            print('filter_num',render_eval_filter_num)
            wandb.log({
                config['name'] + str(len(testset)) + ' - psnr': psnr_test,
                config['name'] + str(len(testset)) + ' - ssim': ssim_test,
                config['name'] + str(len(testset)) + ' - lpips': lpips_test,
                config['name'] + str(len(testset)) + ' - MPJPE': mpjpe,
                config['name'] + str(len(testset)) + ' - MPJPE_MANO': mpjpe_mano,
                config['name'] + str(len(testset)) + ' - ADDS': e_ADDS,
                config['name'] + str(len(testset)) + ' - MCE': e_MCE,
                config['name'] + str(len(testset)) + ' - OCE': e_OCE,
            })
    # wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians_hand_group[list(scene.gaussians_hand_group.keys())[0]].get_xyz.shape[0]})
    wandb.log(
        {'total_points_obj': scene.gaussians_obj_group[list(scene.gaussians_obj_group.keys())[0]].get_xyz.shape[0]})
    torch.cuda.empty_cache()

    scene.train()




@hydra.main(version_base=None, config_path="configs", config_name="config_refine")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    config.exp_dir = os.path.join('/mnt/sda1/lxy/HOGS_results/', config.name, config.refine_tag)
    config.ckpt_dir = '/mnt/sda1/lxy/HOGS_results/dexycb-hogs_prerigid_centeredbbox/obj_pose_refine_tanh_1020/ckpt10.pth'
    #config.ckpt_dir = '/mnt/sda1/lxy/HOGS_results/dexycb-HOGS_color.3/obj_refine_tanh_1020_color_jit/ckpt10.pth'

    os.makedirs(config.exp_dir, exist_ok=True)
    config.dataset.train_sample_rate = 10
    config.checkpoint_iterations.append(config.opt.iterations)
    os.makedirs(os.path.join(config.exp_dir,'code'), exist_ok=True)
    if config.save_code:
        shutil.copyfile('render_dexycb.py', config.exp_dir + '/code/render_dexycb.py')
        shutil.copytree('./scene',config.exp_dir + '/code/scene')
        shutil.copytree('./models',config.exp_dir + '/code/models')
        shutil.copytree('./configs', config.exp_dir + '/code/configs')
        shutil.copytree('./dataset', config.exp_dir + '/code/dataset')
        shutil.copytree('./utils', config.exp_dir + '/code/utils')


    # set wandb logger
    wandb_name = config.name+config.refine_tag
    wandb.init(
        #mode="disabled" if config.wandb_disable else None,
        mode="disabled",
        name=wandb_name,
        project='3DGS_1233',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    #torch.cuda.set_device(1)
    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()  #

