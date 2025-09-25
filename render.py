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

import torch
import numpy as np

from right_hand_model import MANO
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import fix_random, cal_pose_error, relative_pose_error
from scene import GaussianModel

from utils.general_utils import Evaluator, PSEvaluator

import hydra
from omegaconf import OmegaConf
import wandb

def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)



def test(config):
    # generate obj_id and subject_id
    gaussians_hand_group = {}
    gaussians_obj_group = {}

    if config.dataset.name == 'dexycb':
        for obj_id in config.dataset._YCB_CLASSES:
            gaussians_obj_group[int(obj_id)] = None
        for subject in config.dataset._SUBJECTS:
            gaussians_hand_group[int(subject.split('-')[-1])] = None
    else:
        for view in config.dataset.train_view:
            subject_id = int(view.split('-')[0])
            obj_id = int(view.split('-')[1])
            gaussians_obj_group[obj_id] = None
            gaussians_hand_group[subject_id] = None
    with torch.no_grad():

        for sub_id in gaussians_hand_group:
            gaussians_hand_group[sub_id] = GaussianModel(config.model.gaussian)
        for obj_id in gaussians_obj_group:
            gaussians_obj_group[obj_id] = GaussianModel(config.model.gaussian)

        scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/').cuda()

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator()

        psnrs = []
        ssims = []
        lpipss = []
        times = []
        mpjpes = []
        r_error = []
        t_error = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=True)

            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["full_render"]
            mask = render_pkg["full_opacity_render"]

            gt = view.full_image[:3, :, :]
            if  idx % 20 == 0:
                wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
                             wandb.Image(mask[None], caption='mask_{}'.format(view.image_name)),
                             wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))]

                wandb.log({'test_images': wandb_img})

                torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))
                torchvision.utils.save_image(mask, os.path.join(render_path, f"mask_{view.image_name}.png"))

            # evaluate
            if config.evaluate:
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])

                mpjpes.append(cal_pose_error(render_pkg["updated_camera"], body_model))

                r_e, t_e = relative_pose_error(render_pkg["updated_camera"])
                r_error.append(r_e)
                t_error.append(t_e)

                wandb.log({
                    'loss/psnr': metrics['psnr'],
                    'loss/ssim': metrics['ssim'],
                    'loss/lpips': metrics['lpips'],
                })
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
                mpjpes.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)

        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _mpjpe = np.mean(np.stack(mpjpes))
        _r_error = np.mean(np.stack(r_error))
        _t_error = np.mean(np.stack(t_error))
        _time = np.mean(times[1:])
        wandb.log({'test/psnr': _psnr,
                   'test/ssim': _ssim,
                   'test/lpips': _lpips,
                   'test/mpjpe': _mpjpe,
                   'test/r_error': _r_error,
                   'test/t_error': _t_error
                   })
        print('mpjpe:',_mpjpe)
        print('lpips:', _lpips)
        print('psnr:', _psnr)
        print('ssim:', _ssim)
        print('r_error:', _r_error)
        print('t_error:', _t_error)
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy(),
                 time=_time)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('/home/cyc/pycharm/lxy/3DGS/results', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='3DGS_617',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)
    #torch.cuda.set_device(1)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

if __name__ == "__main__":

    main()