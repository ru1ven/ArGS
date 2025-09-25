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

import shutil


# import hydra

from omegaconf import OmegaConf
import swanlab


from right_hand_model import MANO
#import pyiqa # ok
from utils.loss_utils import l1_loss, ssim
#import pyiqa #  nan
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, PSEvaluator, tensor_to_numpy_image, save_deltas
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss

import numpy as np
import torch
import torch.nn.functional as F

import pyiqa #  nan

import hydra
from random import randint


#
# import hydra
# import pyiqa
#
# import swanlab
# from omegaconf import OmegaConf
#
# from right_hand_model import MANO
# from utils.loss_utils import l1_loss, ssim
# from gaussian_renderer import render
# from scene import Scene, GaussianModel
# from utils.general_utils import PSEvaluator
# from tqdm import tqdm
# from utils.loss_utils import full_aiap_loss
# import numpy as np
# import torch
# import torch.nn.functional as F
# from random import randint
#
# from utils.general_utils import fix_random


# @contextmanager
# def profile_block():
#     # Record memory usage before the block
#     mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
#     yield
#     # Record memory usage after the block
#     mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)
#     print(f"Memory usage before: {mem_usage_before[0]} MiB")
#     print(f"Memory usage after: {mem_usage_after[0]} MiB")
#     print(f"Memory increment: {mem_usage_after[0] - mem_usage_before[0]} MiB")

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
        gaussians_obj_group[obj_id] = None
    for subject in dataset._SUBJECTS:
        gaussians_hand_group[subject] = {'right':None, 'left':None,}


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

    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id]['right'] = GaussianModel(model.gaussian)
        gaussians_hand_group[sub_id]['left'] = GaussianModel(model.gaussian)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id] = GaussianModel(model.gaussian)

    scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir)
    scene.train()
    #scene.eval()
    print("training_samples:", len(scene.train_dataset))

    for sub_id in gaussians_hand_group:
        gaussians_hand_group[sub_id]['right'].training_setup(opt)
        gaussians_hand_group[sub_id]['left'].training_setup(opt)
    for obj_id in gaussians_obj_group:
        gaussians_obj_group[obj_id].training_setup(opt)

    # if checkpoint:
    #     scene.load_checkpoint(checkpoint)

    # load_ckpt = os.path.join('/home/cyc/pycharm/lxy/3DGS/result/dexycb-hogs_util30k_delay0/', "ckpt" + str(config.opt.iterations) + ".pth")
    # print(load_ckpt)
    # scene.load_checkpoint(load_ckpt)
    #scene.converter.backbone = CLIPModel.from_pretrained("/home/cyc/pycharm/lxy/3DGS/lib/clip-vit-base-patch32/").cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    data_stack = None
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # tracemalloc.start()
    for iteration in range(first_iter, opt.iterations + 1):

        if iteration == config.rigid_iter+1:
            with torch.no_grad():
                render_pkg = render(scene.train_dataset[0], 1, scene, pipe, background, compute_loss=True,
                                    return_opacity=True,
                                    white_bg=dataset.white_background, delay=False, novel_data=None)

            for obj_id in gaussians_obj_group:
                gaussians_obj_group[obj_id].copy_state_from(render_pkg['obj_refined_gaussian'])

        iter_start.record()

        for sub_id in gaussians_hand_group:
            gaussians_hand_group[sub_id]['right'].update_learning_rate(iteration)
            gaussians_hand_group[sub_id]['left'].update_learning_rate(iteration)

        for obj_id in gaussians_obj_group:
            gaussians_obj_group[obj_id].update_learning_rate(iteration)

        # for obj_id in gaussians_obj_group:
        #     gaussians_obj_group[obj_id].update_learning_rate(iteration-2000 if iteration >= config.rigid_iter+2000 else iteration)
        # scene.converter.scheduler.last_epoch = iteration - 3000 if iteration >= config.rigid_iter+5000 else iteration


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % (1000 * (len(gaussians_obj_group) + len(gaussians_hand_group) // 2)) == 0:
            for sub_id in gaussians_hand_group:
                gaussians_hand_group[sub_id]['right'].oneupSHdegree()
                gaussians_hand_group[sub_id]['left'].oneupSHdegree()
            for obj_id in gaussians_obj_group:
                gaussians_obj_group[obj_id].oneupSHdegree()
        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack) - 1))
        #data_idx = data_stack.pop(0)
        data = scene.train_dataset[data_idx]
        #prev_data = scene.train_dataset[max(0, data_idx - 1)]
        prev_data = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.

        render_pkg = render(data, iteration, scene, pipe, background, compute_loss=True, return_opacity=True,
                            white_bg=dataset.white_background, delay=(iteration <= config.rigid_iter), prev_data=prev_data)
        if iteration % 1000 == 0:
            save_deltas_path = os.path.join(scene.save_dir, 'movable','iteration_{}'.format(iteration), 'movable.obj')
            os.makedirs(os.path.dirname(save_deltas_path), exist_ok=True)
            delta_norm = save_deltas({0: render_pkg['movable_prob'].detach()},
                                     xyz=scene.gaussians_obj_group[
                                         list(scene.gaussians_obj_group.keys())[0]].get_xyz.detach().cpu().numpy(),
                                     filename=save_deltas_path, thrs=[0.25,0.5,0.75], norm=False)
            try:
                delta_norm = save_deltas(getattr(scene.converter, 'deformer_obj_{}'.format(
                    list(scene.gaussians_obj_group.keys())[0])).non_rigid.delta_history,
                                         xyz=scene.gaussians_obj_group[
                                             list(scene.gaussians_obj_group.keys())[0]].get_xyz.detach().cpu().numpy(),
                                         filename=os.path.join(scene.save_dir, 'movable',
                                                               'iteration_{}'.format(iteration), 'delta_nr_pcl.obj'))
            except Exception as e:
                print(f"[Warning] Failed to save deltas: {e}")
                delta_norm = None

        if iteration % 500 == 0:
            # or iteration>=model.deformer.non_rigid.delay and iteration<=model.deformer.non_rigid.delay+2000:
            examples = []
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
            image_ROI = torch.clamp(data.img_ROI.to("cuda"), 0.0, 1.0)
            opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)
            wandb_img = swanlab.Image(tensor_to_numpy_image(opacity_image), caption="h_opacity_{}".format(data.image_name), mode='L', size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(image), caption="h_render_{}".format(data.image_name), size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(gt_image), caption="h_GT_{}".format(data.image_name), size=500)
            examples.append(wandb_img)

            obj_image = torch.clamp(render_pkg["obj_render"], 0.0, 1.0)
            obj_gt_image = torch.clamp(data.obj_image.to("cuda"), 0.0, 1.0)
            obj_opacity_image = torch.clamp(render_pkg["obj_opacity_render"], 0.0, 1.0)
            wandb_img = swanlab.Image(tensor_to_numpy_image(obj_opacity_image), caption="o_opacity_{}".format(data.image_name), mode='L',size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(obj_image), caption="o_render_{}".format(data.image_name), size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(obj_gt_image), caption="o_GT_{}".format(data.image_name), size=500)
            examples.append(wandb_img)

            full_image = torch.clamp(render_pkg["full_render"], 0.0, 1.0)
            full_gt_image = torch.clamp(data.full_image.to("cuda"), 0.0, 1.0)
            full_opacity_image = torch.clamp(render_pkg["full_opacity_render"], 0.0, 1.0)
            wandb_img = swanlab.Image(tensor_to_numpy_image(full_opacity_image), caption="opacity_{}".format(data.image_name), mode='L',size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(full_image), caption="render_{}".format(data.image_name), size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(full_gt_image), caption="GT_{}".format(data.image_name), size=500)
            examples.append(wandb_img)

            wandb_img = swanlab.Image(tensor_to_numpy_image(image_ROI), caption="ROI_{}".format(data.image_name), size=224)
            examples.append(wandb_img)

            #swanlab.log({config['name'] + "_images": examples})
            swanlab.log({"train_images": examples})
            examples.clear()

        image, viewspace_point_tensor_r, visibility_filter_r,radii_r, viewspace_point_tensor_l, visibility_filter_l, radii_l,  = \
        render_pkg["render"], render_pkg["viewspace_points_r"], render_pkg["visibility_filter_r"],render_pkg["radii_r"],\
        render_pkg[ "viewspace_points_l"], render_pkg["visibility_filter_l"], render_pkg["radii_l"]

        obj_image, obj_viewspace_point_tensor, obj_visibility_filter, obj_radii = render_pkg["obj_render"], render_pkg[
            "obj_viewspace_points"], render_pkg["obj_visibility_filter"], render_pkg["obj_radii"]
        full_image, full_viewspace_point_tensor, full_visibility_filter, full_radii = render_pkg["full_render"], \
                                                                                      render_pkg[
                                                                                          "full_viewspace_points"], \
                                                                                      render_pkg[
                                                                                          "full_visibility_filter"], \
                                                                                      render_pkg["full_radii"]

        opacity = render_pkg["opacity_render"] if use_mask else None
        obj_opacity = render_pkg["obj_opacity_render"] if use_mask else None
        full_opacity = render_pkg["full_opacity_render"] if use_mask else None

        # Loss
        gt_image = data.original_image.cuda()
        obj_gt_image = data.obj_image.cuda()
        full_gt_image = data.full_image.cuda()

        gt_mask = data.original_mask.cuda()
        obj_mask = data.obj_mask.cuda()
        full_mask = data.full_mask.cuda()

        # bg_color = torch.tensor(1) if dataset.white_background else torch.tensor(0)
        # image = torch.where(gt_mask == 0, bg_color, image)
        # obj_image = torch.where(obj_mask == 0, bg_color, obj_image)
        # full_image = torch.where(full_mask == 0, bg_color, full_image)

        lambda_l1 = C(iteration, config.opt.lambda_l1)
        lambda_dssim = C(iteration, config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image) + l1_loss(obj_image, obj_gt_image) + l1_loss(full_image, full_gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image) + 1.0 - ssim(obj_image, obj_gt_image) + 1.0 - ssim(full_image,
                                                                                                        full_gt_image)
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # perceptual loss
        lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))

        if lambda_perceptual > 0:
            # crop the foreground
            try:
                # full
                with torch.no_grad():
                    mask = np.where(data.full_mask.cpu().numpy())
                    y1, y2 = mask[1].min(), mask[1].max() + 1
                    x1, x2 = mask[2].min(), mask[2].max() + 1
                    fg_image = full_image[:, y1:y2, x1:x2]
                    gt_fg_image = full_gt_image[:, y1:y2, x1:x2]
                    #loss_perceptual = loss_fn_vgg(fg_image.cuda(), gt_fg_image.cuda(), normalize=True).mean()
                    loss_perceptual = loss_fn_vgg(fg_image.cuda(), gt_fg_image.cuda())
            except Exception as e:
                loss_perceptual = torch.tensor(0.)

            try:
                mask = np.where(data.original_mask.cpu().numpy())
                y1, y2 = mask[1].min(), mask[1].max() + 1
                x1, x2 = mask[2].min(), mask[2].max() + 1
                fg_image = image[:, y1:y2, x1:x2]
                gt_fg_image = gt_image[:, y1:y2, x1:x2]
                #loss_perceptual += loss_fn_vgg(fg_image.cuda(), gt_fg_image.cuda(), normalize=True).mean()
                loss_perceptual += loss_fn_vgg(fg_image.cuda(), gt_fg_image.cuda())
            except Exception as e:
                pass

            # obj
            try:
                mask = np.where(data.obj_mask.cpu().numpy())
                y1, y2 = mask[1].min(), mask[1].max() + 1
                x1, x2 = mask[2].min(), mask[2].max() + 1
                fg_image = obj_image[:, y1:y2, x1:x2]
                gt_fg_image = obj_gt_image[:, y1:y2, x1:x2]
                #loss_perceptual += loss_fn_vgg(fg_image.cuda(), gt_fg_image.cuda(), normalize=True).mean()
                loss_perceptual += loss_fn_vgg(fg_image.cuda(), gt_fg_image.cuda())
            except Exception as e:
                pass

            loss += lambda_perceptual * loss_perceptual

        else:
            loss_perceptual = torch.tensor(0.)

        # mask loss

        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1. - 1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)

            obj_opacity = torch.clamp(obj_opacity, 1.e-3, 1. - 1.e-3)
            loss_mask += F.binary_cross_entropy(obj_opacity, obj_mask)

            full_opacity = torch.clamp(full_opacity, 1.e-3, 1. - 1.e-3)
            loss_mask += F.binary_cross_entropy(full_opacity, full_mask)

        elif config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
            loss_mask += F.l1_loss(obj_opacity, obj_mask)
            loss_mask += F.l1_loss(full_opacity, full_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        # skinning loss
        lambda_skinning = C(iteration, config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss(data.subject_id)
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians_hand_group[data.subject_id]['right'],
                                                          render_pkg["deformed_gaussian_r"])
            loss_aiap_xyz_l, loss_aiap_cov_l = full_aiap_loss(scene.gaussians_hand_group[data.subject_id]['left'],
                                                          render_pkg["deformed_gaussian_l"])
            # obj_loss_aiap_xyz, obj_loss_aiap_cov = full_aiap_loss(scene.gaussians_obj_group[data.obj_id],
            #                                                       render_pkg['obj_deformed_gaussian'])
            obj_loss_aiap_xyz, obj_loss_aiap_cov = full_aiap_loss(scene.gaussians_obj_group[data.obj_id],
                                                                  render_pkg['obj_deformed_gaussian'], articulated=True)
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
            loss_aiap_xyz_l = torch.tensor(0.).cuda()
            loss_aiap_cov_l = torch.tensor(0.).cuda()
            obj_loss_aiap_cov = torch.tensor(0.).cuda()
            obj_loss_aiap_xyz = torch.tensor(0.).cuda()

        loss += lambda_aiap_cov * (loss_aiap_cov+loss_aiap_cov_l) + lambda_aiap_xyz * (loss_aiap_xyz+loss_aiap_xyz_l)
        loss += lambda_aiap_cov * obj_loss_aiap_cov + lambda_aiap_xyz * obj_loss_aiap_xyz

        # regularization

        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(iteration, lbd)
            loss_reg[name] *= lbd
            loss += loss_reg[name]

        loss.backward()


        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            swanlab.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(iteration, testing_iterations, testing_interval, (iteration <= config.rigid_iter), scene, evaluator, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:

                # Keep track of max radii in image-space for pruning
                gaussians_hand_group[data.subject_id]['left'].max_radii2D[visibility_filter_l] = torch.max(
                    gaussians_hand_group[data.subject_id]['left'].max_radii2D[visibility_filter_l],
                    radii_l[visibility_filter_l])
                gaussians_hand_group[data.subject_id]['left'].add_densification_stats(viewspace_point_tensor_l, visibility_filter_l)

                gaussians_hand_group[data.subject_id]['right'].max_radii2D[visibility_filter_r] = torch.max(
                    gaussians_hand_group[data.subject_id]['right'].max_radii2D[visibility_filter_r],
                    radii_r[visibility_filter_r])
                gaussians_hand_group[data.subject_id]['right'].add_densification_stats(viewspace_point_tensor_r,
                                                                                      visibility_filter_r)

                gaussians_obj_group[data.obj_id].max_radii2D[obj_visibility_filter] = torch.max(
                    gaussians_obj_group[data.obj_id].max_radii2D[obj_visibility_filter],
                    obj_radii[obj_visibility_filter])
                gaussians_obj_group[data.obj_id].add_densification_stats(obj_viewspace_point_tensor,
                                                                         obj_visibility_filter)
                #if iteration < config.rigid_iter or iteration > config.rigid_iter + 2000:

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    for sub_id in gaussians_hand_group:
                        gaussians_hand_group[sub_id]['right'].densify_and_prune(opt, scene, size_threshold)
                        gaussians_hand_group[sub_id]['left'].densify_and_prune(opt, scene, size_threshold)
                    for obj_id in gaussians_obj_group:
                        gaussians_obj_group[obj_id].densify_and_prune(opt, scene, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    for sub_id in gaussians_hand_group:
                        gaussians_hand_group[sub_id]['right'].reset_opacity()
                        gaussians_hand_group[sub_id]['left'].reset_opacity()
                    for obj_id in gaussians_obj_group:
                        gaussians_obj_group[obj_id].reset_opacity()

            # Optimizer step
            # if iteration < config.rigid_iter or iteration > config.rigid_iter + 2000:
            scene.optimize(iteration)
            # else:
            #     scene.converter.optimize()
            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)




def validation(iteration, testing_iterations, testing_interval, rigid_delay, scene: Scene, evaluator, renderArgs):
    # Report test and samples of training set
    if testing_interval > 0:
        if not iteration % testing_interval == 0:
            return
    else:
        if not iteration in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()

    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/').cuda()


    print("test_samples:",len(scene.test_dataset))


    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    psnr_train = 0.0
    ssim_train = 0.0
    lpips_train = 0.0

    for idx, data in enumerate(scene.train_dataset):

        novel_cacmera = scene.test_dataset[idx]
        #novel_cacmera = None
        render_pkg = render(data, iteration+idx, scene, *renderArgs, compute_loss=True,
                            return_opacity=True, delay=rigid_delay, novel_data=novel_cacmera)
        examples = []

        movable_prob = render_pkg["movable_prob"]

        full_image = torch.clamp(render_pkg["full_render"], 0.0, 1.0)
        full_gt_image = torch.clamp(data.full_image.to("cuda"), 0.0, 1.0)
        full_image_novel = torch.clamp(render_pkg["novel_render"], 0.0, 1.0)
        full_gt_image_novel = torch.clamp(novel_cacmera.full_image.to("cuda"), 0.0, 1.0)

        if idx % 2 == 0:
            # import cv2
            # cv2.imwrite('/mnt/sda2/lxy/NonrigidGS_results/debug/full_image_novel.png',
            #                         cv2.cvtColor(np.uint8(full_image_novel.permute(1, 2, 0).detach().cpu().numpy() * 255),
            #                                      cv2.COLOR_BGR2RGB))
            # cv2.imwrite('/mnt/sda2/lxy/NonrigidGS_results/debug/full_image.png',
            #             cv2.cvtColor(np.uint8(full_image.permute(1, 2, 0).detach().cpu().numpy() * 255),
            #                          cv2.COLOR_BGR2RGB))
            # cv2.imwrite('/mnt/sda2/lxy/NonrigidGS_results/debug/full_gt_image_novel.png',
            #             cv2.cvtColor(np.uint8(full_gt_image_novel.permute(1, 2, 0).detach().cpu().numpy() * 255),
            #                          cv2.COLOR_BGR2RGB))
            wandb_img = swanlab.Image(tensor_to_numpy_image(full_image),
                                    caption="render_view_{}".format(data.image_name), size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(full_gt_image),
                                    caption="GT_view_{}".format(
                                        data.image_name), size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(full_image_novel),
                                      caption= "render_novel_{}".format(data.image_name), size=500)
            examples.append(wandb_img)
            wandb_img = swanlab.Image(tensor_to_numpy_image(full_gt_image_novel),
                                      caption="GT_novel_{}".format(
                                          data.image_name), size=500)

            examples.append(wandb_img)

            swanlab.log({'test'+ "_{}".format(iteration): examples})
            examples.clear()


        metrics = evaluator(full_image, full_gt_image)

        psnr_train += metrics['psnr']
        ssim_train += metrics['ssim']
        lpips_train += metrics['lpips']
        updated_camera = render_pkg['updated_camera']

        swanlab.log({
            'test' + '/psnr': metrics['psnr'],
            'test' + '/ssim': metrics['ssim'],
            'test' + '/lpips': metrics['lpips'],
        })
        metrics_novel = evaluator(full_image_novel, full_gt_image_novel)

        psnr_test += metrics_novel['psnr']
        ssim_test += metrics_novel['ssim']
        lpips_test += metrics_novel['lpips']

        swanlab.log({
            'test' + '/novel_psnr': metrics_novel['psnr'],
            'test' + '/novel_ssim': metrics_novel['ssim'],
            'test' + '/novel_lpips': metrics_novel['lpips'],
        })
    try:
        delta_norm = save_deltas(getattr(scene.converter, 'deformer_obj_{}'.format(
            list(scene.gaussians_obj_group.keys())[0])).non_rigid.delta_history,
                                 xyz=scene.gaussians_obj_group[
                                     list(scene.gaussians_obj_group.keys())[0]].get_xyz.detach().cpu().numpy(),
                                 filename=os.path.join(scene.save_dir, 'movable','iteration_{}'.format(iteration),'delta_nr_pcl.obj'))
    except Exception as e:
        print(f"[Warning] Failed to save deltas: {e}")
        delta_norm = None
    #
    # delta_norm = save_deltas( {0: movable_prob},
    #                          xyz=scene.gaussians_obj_group[
    #                              list(scene.gaussians_obj_group.keys())[0]].get_xyz.detach().cpu().numpy(),
    #                          filename=os.path.join(scene.save_dir, 'movable_prob.obj'))

    psnr_test /= len(scene.test_dataset)
    ssim_test /= len(scene.test_dataset)
    lpips_test /= len(scene.test_dataset)
    psnr_train /= len(scene.train_dataset)
    ssim_train /= len(scene.train_dataset)
    lpips_train /= len(scene.train_dataset)

    print("\n[ITER {}] Evaluating {}: lpips {} PSNR {} SSIM {}".format(iteration, 'test', lpips_test, psnr_test, ssim_test))
    swanlab.log({

        'test' + '/loss_viewpoint - psnr': psnr_train,
        'test' + '/loss_viewpoint - ssim': ssim_train,
        'test' + '/loss_viewpoint - lpips': lpips_train,

        'test' + '/novel - psnr': psnr_test,
        'test' + '/novel - ssim': ssim_test,
        'test' + '/novel - lpips': lpips_test,

    })
    # wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    swanlab.log({'p_num_r': scene.gaussians_hand_group[list(scene.gaussians_hand_group.keys())[0]]['right'].get_xyz.shape[0]})
    swanlab.log({'p_num_l': scene.gaussians_hand_group[list(scene.gaussians_hand_group.keys())[0]]['left'].get_xyz.shape[0]})
    swanlab.log(
        {'p_num_obj': scene.gaussians_obj_group[list(scene.gaussians_obj_group.keys())[0]].get_xyz.shape[0]})
    torch.cuda.empty_cache()

    scene.train()


@hydra.main(version_base=None, config_path="configs", config_name="config_arctic")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    config.exp_dir = config.get('exp_dir') or os.path.join('/mnt/sda2/lxy/NonrigidGS_results/', config.dataset._YCB_CLASSES[0],config.name)

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



    # set wandb logger
    #os.makedirs(os.path.join(config.exp_dir, 'swanlab'), exist_ok=True)

    wandb_name = config.name
    enable_swanlab = not getattr(config, "wandb_disable", False)

    swanlab_log = os.path.join('/mnt/sda2/lxy/NonrigidGS_results/', config.dataset._YCB_CLASSES[0],'swanlab')
    os.makedirs(swanlab_log, exist_ok=True)
    swanlab.init(
        name=wandb_name,
        project='NonrigidGS_715',
        config=OmegaConf.to_container(config, resolve=True),
        logdir=swanlab_log,
        mode='local' if enable_swanlab else 'disabled'
    )


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
