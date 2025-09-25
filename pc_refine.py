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
import gc
import os
import shutil
import tracemalloc
from memory_profiler import profile
from memory_profiler import memory_usage
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint

from right_hand_model import MANO
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator, cal_pose_error
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss
import pyiqa
import hydra
from omegaconf import OmegaConf
import wandb
from submodules import lpips
import random

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

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'alex')
    #loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda()  # for training
    # loss_fn_vgg_h = lpips.LPIPS(net=lpips_type).cuda()  # for training
    # loss_fn_vgg_o = lpips.LPIPS(net=lpips_type).cuda()  # for training
    loss_fn_vgg = pyiqa.create_metric('lpips', device='cuda', as_loss=True)
    # loss_fn_vgg_h = pyiqa.create_metric('lpips-vgg', device='cuda', as_loss=True)
    # loss_fn_vgg_o = pyiqa.create_metric('lpips-vgg', device='cuda', as_loss=True)
    # evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()
    evaluator = PSEvaluator()

    first_iter = 360000

    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id] = GaussianModel(model.gaussian)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id] = GaussianModel(model.gaussian)

    scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir)
    scene.train()
    load_ckpt = config.get('load_ckpt', None)
    if load_ckpt is None:
        load_ckpt = os.path.join(config.ckpt_dir, "ckpt" + str(first_iter) + ".pth")
    scene.load_checkpoint(load_ckpt)


    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id].refine_setup(opt)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id].refine_setup(opt)

    scene.converter.pc_refine_setup()


    if dataset.refine:
        scene.load_ref_ckpt(dataset.refine_ckpt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    data_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack) - 1))
        # data_idx = data_stack.pop(0)
        data = scene.train_dataset[data_idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.

        pc_hand, pc_obj, loss_reg, colors_precomp, obj_colors_precomp, updated_camera = scene.convert_gaussians(data, iteration, compute_loss=True)


        # regularization
        loss = torch.tensor(0.).cuda()
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(iteration, lbd)
            loss += lbd * value / config.accumulation_steps


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
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(iteration, testing_iterations, testing_interval, scene, evaluator, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Optimizer step
            # if iteration < opt.iterations:
            #     scene.optimize_gaussians(iteration)
            if iteration < opt.iterations and iteration % config.accumulation_steps == 0:
                scene.converter.optimize()


            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)







def validation(iteration, testing_iterations, testing_interval, scene: Scene, evaluator, renderArgs):
    # Report test and samples of training set
    if testing_interval > 0:
        if not iteration % testing_interval == 0:
            return
    else:
        if not iteration in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    body_model_pca = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/',
                          use_pca=True, num_pca_comps=48, flat_hand_mean=False).cuda()

    # validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
    #                       {'name': 'train', 'cameras' : list(range(len(scene.train_dataset)))})
    validation_configs = ({'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},)
    for config in validation_configs:

        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0

            examples = []
            mpjpe = 0.0
            for idx, data_idx in enumerate(config['cameras']):

                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=True, return_opacity=True)
                examples = []
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)

                obj_image = torch.clamp(render_pkg["obj_render"], 0.0, 1.0)
                obj_gt_image = torch.clamp(data.obj_image.to("cuda"), 0.0, 1.0)
                obj_opacity_image = torch.clamp(render_pkg["obj_opacity_render"], 0.0, 1.0)

                full_image = torch.clamp(render_pkg["full_render"], 0.0, 1.0)
                full_gt_image = torch.clamp(data.full_image.to("cuda"), 0.0, 1.0)
                full_opacity_image = torch.clamp(render_pkg["full_opacity_render"], 0.0, 1.0)
                if idx % 20 == 0:
                    # wandb_img = wandb.Image(opacity_image[None], "view_{}/render_opacity".format(data.image_name))
                    # examples.append(wandb_img)
                    # wandb_img = wandb.Image(image[None], "view_{}/render".format(data.image_name))
                    # examples.append(wandb_img)
                    # wandb_img = wandb.Image(gt_image[None], "view_{}/ground_truth".format(data.image_name))
                    # examples.append(wandb_img)
                    # wandb_img = wandb.Image(obj_opacity_image[None],
                    #                         "view_{}/obj_render_opacity".format(data.image_name))
                    # examples.append(wandb_img)
                    # wandb_img = wandb.Image(obj_image[None], "view_{}/obj_render".format(data.image_name))
                    # examples.append(wandb_img)
                    # wandb_img = wandb.Image(obj_gt_image[None], "view_{}/obj_ground_truth".format(data.image_name))
                    # examples.append(wandb_img)
                    # wandb_img = wandb.Image(full_opacity_image[None],
                    #                         caption=config['name'] + "_view_{}/full_render_opacity".format(
                    #                             data.image_name))
                    # examples.append(wandb_img)
                    wandb_img = wandb.Image(full_image[None],
                                            caption=config['name'] + "_view_{}/full_render".format(data.image_name))
                    examples.append(wandb_img)
                    wandb_img = wandb.Image(full_gt_image[None],
                                            caption=config['name'] + "_view_{}/full_ground_truth".format(
                                                data.image_name))
                    examples.append(wandb_img)

                    wandb.log({config['name'] + "_images": examples})
                    examples.clear()

                if config['name'] == 'test':
                    metrics = evaluator(full_image, full_gt_image)
                    psnr_test += metrics['psnr']
                    ssim_test += metrics['ssim']
                    lpips_test += metrics['lpips']
                    mpjpe += cal_pose_error(render_pkg["updated_camera"],body_model_pca)

                    wandb.log({
                        config['name'] + '/psnr': metrics['psnr'],
                        config['name'] + '/ssim': metrics['ssim'],
                        config['name'] + '/lpips': metrics['lpips'],
                    })

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            mpjpe /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: lpips {} PSNR {} MPJPE {}".format(iteration, config['name'], lpips_test, psnr_test, mpjpe))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
                config['name'] + '/loss_viewpoint - MPJPE': mpjpe,
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
    config.exp_dir = os.path.join('/home/cyc/pycharm/lxy/3DGS/result', config.name, config.refine_tag)
    config.ckpt_dir = os.path.join('/home/cyc/pycharm/lxy/3DGS/result', config.name)

    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)
    os.makedirs(os.path.join(config.exp_dir,'code'), exist_ok=True)
    if config.save_code:
        shutil.copyfile('./train.py', config.exp_dir + '/code/train.py')
        shutil.copytree('./scene',config.exp_dir + '/code/scene')
        shutil.copytree('./models',config.exp_dir + '/code/models')


    # set wandb logger
    wandb_name = config.name+config.refine_tag
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='3DGS_701',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    torch.cuda.set_device(1)
    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()  #
