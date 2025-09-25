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

    # generate obj_id and subject_id
    gaussians_hand_group = {}
    gaussians_obj_group = {}

    for obj_id in dataset._YCB_CLASSES:
        gaussians_obj_group[int(obj_id)] = None
    for subject in dataset._SUBJECTS:
        gaussians_hand_group[int(subject.split('-')[-1])] = None

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'alex')
    loss_fn_vgg = pyiqa.create_metric('lpips', device='cuda', as_loss=True)
    evaluator = PSEvaluator()

    first_iter = 200000

    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id] = GaussianModel(model.gaussian)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id] = GaussianModel(model.gaussian)

    scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir, multi_batch=True)
    scene.train()
    load_ckpt = config.get('load_ckpt', None)
    if load_ckpt is None:
        load_ckpt = os.path.join(config.ckpt_dir, "ckpt" + str(first_iter) + ".pth")

    # scene.converter.pose_model_hand.pointnet.sa1.mlp_convs[0] = nn.Conv2d(3 + 103 +304, 64, 1).cuda()
    # scene.converter.pose_model_obj.pointnet.sa1.mlp_convs[0] = nn.Conv2d(3 + 103+304, 64, 1).cuda()
    scene.converter.pose_model_hand.pose_mlp = VanillaCondMLP(768+128, 45 + 3 + 10 + 3, 45 + 3 + 10 + 3,
                                   config.model.deformer.non_rigid.pose_correction.mlp).cuda()
    scene.converter.pose_model_obj.pose_mlp =VanillaCondMLP(768+128, 128, 6,
                                   config.model.deformer.non_rigid.pose_correction.mlp).cuda()
    scene.converter.pose_model_hand.pointnet.sa1.mlp_convs[0] = nn.Conv2d(3 + 103-48, 64, 1).cuda()
    scene.converter.pose_model_obj.pointnet.sa1.mlp_convs[0] = nn.Conv2d(3 + 103-48, 64, 1).cuda()

    scene.load_checkpoint(load_ckpt)

    # scene.converter.pose_model_hand.pose_mlp = VanillaCondMLP(768+256, 45 + 3 + 10 + 3, 45 + 3 + 10 + 3,
    #                                config.model.deformer.non_rigid.pose_correction.mlp).cuda()
    # scene.converter.pose_model_obj.pose_mlp =VanillaCondMLP(768+256, 128, 6,
    #                                config.model.deformer.non_rigid.pose_correction.mlp).cuda()
    #
    #
    scene.converter.pose_model_hand.pointnet.sa1.mlp_convs[0] = nn.Conv2d(3 + 103, 64, 1).cuda()
    scene.converter.pose_model_obj.pointnet.sa1.mlp_convs[0] = nn.Conv2d(3 + 103, 64, 1).cuda()
    #
    scene.converter.kpTR = KeypointTR(config).cuda()

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
    testLoader = DataLoader(scene.test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=8)

    end_iter = trainLoader.__len__() * config.refine_epoch + first_iter + 1

    scene.converter.pc_refine_setup(trainLoader.__len__() * config.refine_epoch)

    progress_bar = tqdm(range(first_iter, end_iter), desc="Training progress")


    #for iteration in range(first_iter, opt.iterations + 1):
    for epoch in range(config.refine_epoch):
        if epoch == 0:
            validation(first_iter, scene, evaluator, testLoader, (pipe, background))
        for ii, data in enumerate(trainLoader):
            data = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value
                                for key, value in data.items()}
            iteration = ii + trainLoader.__len__() * epoch + first_iter + 1
            iter_start.record()

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            lambda_mask = C(iteration, config.opt.lambda_mask)
            use_mask = lambda_mask > 0.

            loss_reg, updated_camera = scene.convert_gaussians(data, iteration, compute_loss=True, pose_refine=True)

            #regularization
            loss = torch.tensor(0.).cuda()
            for name in loss_reg.keys():
                lbd = opt.get(f"lambda_{name}", 1.)
                lbd = C(iteration, lbd)

                # if loss_reg[name].shape != torch.Size([]):
                #     print(name)
                #     print(loss_reg[name].shape)
                #     print(loss_reg[name])

                loss_reg[name] *= lbd
                loss += loss_reg[name]
            #loss.backward()

            # loss_reg["obj_rot"] *= float(opt.obj_rot_weight)
            # loss_reg["obj_trans"] *= float(opt.obj_trans_weight)
            # loss_reg["obj_corner"] *= float(opt.obj_corner_weight)


            loss.backward()

            iter_end.record()
            torch.cuda.synchronize()

            with torch.no_grad():
                elapsed = iter_start.elapsed_time(iter_end)
                log_loss = {

                    'loss/total_loss': loss.item(),
                    'iter_time': elapsed,
                }
                log_loss.update({
                    'loss/loss_' + k: v for k, v in loss_reg.items()
                })
                wandb.log(log_loss)

                # Progress bar

                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == end_iter-1:
                    progress_bar.close()

                # Optimizer step
                scene.converter.optimize()

                if ii == trainLoader.__len__()-1 and epoch == config.refine_epoch-1:
                    scene.save_checkpoint(epoch)


        validation(iteration, scene, evaluator, testLoader, (pipe, background))


def validation_pose(iteration, scene: Scene, testLoader):
    scene.eval()
    torch.cuda.empty_cache()
    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/').cuda()

    # validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
    #                       {'name': 'train', 'cameras' : list(range(len(scene.train_dataset)))})
    validation_configs = ({'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},)
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

            examples = []
            #for ii, data in enumerate(testLoader):
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                loss_reg, updated_camera = scene.convert_gaussians(data, iteration, compute_loss=True, pose_refine=True)

                if config['name'] == 'test':

                    mpjpe += np.linalg.norm(updated_camera['pred_joints'].detach().cpu().numpy() - updated_camera['gt_mano_joints'].detach().cpu().numpy(), axis=-1).mean()
                    mpjpe_mano += np.linalg.norm(updated_camera['pred_joints_mano'].detach().cpu().numpy() - updated_camera[
                        'gt_mano_joints'].detach().cpu().numpy(), axis=-1).mean()

                    OCE = np.linalg.norm(updated_camera['obj_trans'].detach().cpu().numpy() - updated_camera['obj_trans_gt'].detach().cpu().numpy(), axis=-1)
                    ADDS, MCE = compute_obj_metrics_ycb(updated_camera['obj_rots_gt'],
                                                           updated_camera['obj_trans_gt'],
                                                           updated_camera['obj_rots'],
                                                           updated_camera['obj_trans'],
                                                           updated_camera['obj_id'])

                    e_ADDS += ADDS.mean().item()
                    e_MCE += MCE.mean().item()
                    e_OCE += OCE.mean()


            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            mpjpe /= len(config['cameras'])
            mpjpe_mano /= len(config['cameras'])
            e_ADDS /= len(config['cameras'])
            e_MCE /= len(config['cameras'])
            e_OCE /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: MPJPE {} t_error {} ADDS {} MCE {}".format(iteration, config['name'], mpjpe, e_OCE, e_ADDS, e_MCE))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
                config['name'] + '/loss_viewpoint - MPJPE': mpjpe,
                config['name'] + '/loss_viewpoint - MPJPE_MANO': mpjpe_mano,
                config['name'] + '/loss_viewpoint - ADDS': e_ADDS,
                config['name'] + '/loss_viewpoint - MCE': e_MCE,
                config['name'] + '/loss_viewpoint - OCE': e_OCE,
            })
    # wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians_hand_group[list(scene.gaussians_hand_group.keys())[0]].get_xyz.shape[0]})
    wandb.log(
        {'total_points_obj': scene.gaussians_obj_group[list(scene.gaussians_obj_group.keys())[0]].get_xyz.shape[0]})
    torch.cuda.empty_cache()

    scene.train()

def validation(iteration, scene: Scene, evaluator, testLoader, renderArgs):

    scene.eval()
    torch.cuda.empty_cache()
    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/').cuda()

    # validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
    #                       {'name': 'train', 'cameras' : list(range(len(scene.train_dataset)))})
    validation_configs = ({'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},)
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

            examples = []
            for idx, data in enumerate(testLoader):
                data = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value
                        for key, value in data.items()}

                render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=True, return_opacity=True, pose_refine=True)
                updated_camera = render_pkg['updated_camera']
                examples = []


                full_image = torch.clamp(render_pkg["full_render"], 0.0, 1.0)
                full_gt_image = torch.clamp(updated_camera.full_image.to("cuda"), 0.0, 1.0)

                if idx % 200 == 0:
                    #print(full_image.shape)
                    # cv2.imwrite('/home/cyc/pycharm/lxy/3DGS/debug/refine_ho3d.png',cv2.cvtColor(np.uint8(full_image.permute(1,2,0).detach().cpu().numpy()*255), cv2.COLOR_BGR2RGB))
                    # cv2.imwrite('/home/cyc/pycharm/lxy/3DGS/debug/refine_ho3d_gt.png',
                    #             cv2.cvtColor(np.uint8(full_gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255),
                    #                          cv2.COLOR_BGR2RGB))

                    wandb_img = wandb.Image(full_image[None],
                                            caption=config['name'] + "_view_{}/full_render".format(updated_camera.image_name))
                    examples.append(wandb_img)
                    wandb_img = wandb.Image(full_gt_image[None],
                                            caption=config['name'] + "_view_{}/full_ground_truth".format(
                                                updated_camera.image_name))
                    examples.append(wandb_img)

                    wandb.log({config['name'] + "_images": examples})
                    examples.clear()

                if config['name'] == 'test':

                    metrics = evaluator(full_image, full_gt_image)

                    psnr_test += metrics['psnr'].cpu().item()
                    ssim_test += metrics['ssim'].cpu().item()
                    lpips_test += metrics['lpips'].cpu().item()


                    mpjpe += cal_pose_error(updated_camera, body_model)
                    mpjpe_mano += np.linalg.norm(
                        updated_camera.pred_joints_mano.detach().cpu().numpy() - updated_camera.gt_mano_joints.detach().cpu().numpy(), axis=-1).mean()

                    OCE = relative_pose_error(updated_camera)
                    ADDS, MCE = compute_obj_metrics_ycb(updated_camera.obj_rots_gt.unsqueeze(0),
                                                           updated_camera.obj_trans_gt.unsqueeze(0),
                                                           updated_camera.obj_rots.unsqueeze(0),
                                                           updated_camera.obj_trans.unsqueeze(0),
                                                           updated_camera.obj_id)

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


            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            mpjpe /= len(config['cameras'])
            mpjpe_mano /= len(config['cameras'])
            e_ADDS /= len(config['cameras'])
            e_MCE /= len(config['cameras'])
            e_OCE /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: lpips {} PSNR {} MPJPE {} t_error {} ".format(iteration, config['name'], lpips_test,
                                                                                psnr_test, mpjpe, e_OCE))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
                config['name'] + '/loss_viewpoint - MPJPE': mpjpe,
                config['name'] + '/loss_viewpoint - MPJPE_MANO': mpjpe_mano,
                config['name'] + '/loss_viewpoint - ADDS': e_ADDS,
                config['name'] + '/loss_viewpoint - MCE': e_MCE,
                config['name'] + '/loss_viewpoint - OCE': e_OCE,
            })
    # wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians_hand_group[list(scene.gaussians_hand_group.keys())[0]].get_xyz.shape[0]})
    wandb.log(
        {'total_points_obj': scene.gaussians_obj_group[list(scene.gaussians_obj_group.keys())[0]].get_xyz.shape[0]})
    torch.cuda.empty_cache()

    scene.train()




@hydra.main(version_base=None, config_path="configs", config_name="config_refine_ho3d")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    config.exp_dir = os.path.join('/mnt/sda1/lxy/HOGS_results/', config.name, config.refine_tag)
    config.ckpt_dir = os.path.join('/mnt/sda1/lxy/HOGS_results/', config.name)

    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)
    os.makedirs(os.path.join(config.exp_dir,'code'), exist_ok=True)
    if config.save_code:
        shutil.copyfile('./train_refine_ho3d.py', config.exp_dir + '/code/train_refine_ho3d.py')
        shutil.copytree('./scene',config.exp_dir + '/code/scene')
        shutil.copytree('./models',config.exp_dir + '/code/models')
        shutil.copytree('./configs', config.exp_dir + '/code/configs')
        shutil.copytree('./dataset', config.exp_dir + '/code/dataset')
        shutil.copytree('./utils', config.exp_dir + '/code/utils')


    # set wandb logger
    wandb_name = config.name+config.refine_tag
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='3DGS_poseRefine_1226',
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
