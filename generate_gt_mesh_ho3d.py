import cv2
import hydra
import torch
import os

from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPModel

from models.network_utils import VanillaCondMLP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os import makedirs

from omegaconf import OmegaConf

from scene.query_cam_loader import QueryCamerasLoader
from gaussian_renderer import integrate, render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from scene import Scene, GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra

Seq2YCB_CLASSES= {'ABF12':'021_bleach_cleanser', 'ABF14':'021_bleach_cleanser', 'BB12':'011_banana', 'BB13':'011_banana',
                  'GPMF12':'010_potted_meat_can','GPMF14':'010_potted_meat_can','GSF12':'037_scissors','GSF13':'037_scissors',
                'MC1':'003_cracker_box', 'MC4':'003_cracker_box', 'MDF12':'035_power_drill','MDF14':'035_power_drill',
                'ShSu10':'004_sugar_box','ShSu12':'004_sugar_box','SM2':'006_mustard_bottle','SM4':'006_mustard_bottle',
                 'SMu1':'025_mug','SMu40':'025_mug'
            }

def extract_mesh(config, ckpt_path, filter_mesh: bool, texture_mesh: bool,
                 near: float, far: float):
    with torch.no_grad():
        dataset = config.dataset
        model = config.model
        pipeline = config.pipeline

        gaussians_hand_group = {}
        gaussians_obj_group = {}
        rendered_hand = {}
        rendered_obj = {}

        for obj_id in dataset._YCB_CLASSES:
            gaussians_obj_group[int(obj_id)] = None
            rendered_obj[int(obj_id)] = False
        for subject in dataset._SUBJECTS:
            gaussians_hand_group[int(subject.split('-')[-1])] = None
            rendered_hand[int(subject.split('-')[-1])] = False


        for sub_id in gaussians_hand_group:
            gaussians_hand_group[sub_id] = GaussianModel(model.gaussian)
        for obj_id in gaussians_obj_group:
            gaussians_obj_group[obj_id] = GaussianModel(model.gaussian)

        scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir)
        scene.eval()

        load_ckpt = os.path.join(ckpt_path, "ckpt200000.pth")

        scene.load_checkpoint(load_ckpt)
        ply_path = os.path.join(ckpt_path,"mesh")


        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = 0.0

        first_iter = 0

        gt_path = os.path.join(ckpt_path,  'gt_mesh')
        os.makedirs(gt_path,exist_ok=True)

        progress_bar = tqdm(range(scene.test_dataset.__len__()), desc="Training progress")
        thers = [0.5]
        for ther in thers:
            for obj_id in dataset._YCB_CLASSES:
                rendered_obj[int(obj_id)] = False
            for subject in dataset._SUBJECTS:
                rendered_hand[int(subject.split('-')[-1])] = False
            for idx, data_idx in enumerate(list(range(len(scene.test_dataset)))):
                data = scene.test_dataset[data_idx]

                obj_id = int(data.obj_id)
                sub_id = int(data.subject_id)

                file_name = str(sub_id) + '_' + str(data.cam_id) + '_' + str(int(data.frame_id))

                if rendered_obj[obj_id] is False or rendered_hand[sub_id] is False:
                    out_path = os.path.join(gt_path, file_name+'_obj.ply')
                    gt_obj_mesh_path = os.path.join('/home/cyc/pycharm/lxy/3DGS/lib/YCB_models/',
                                                    Seq2YCB_CLASSES[file_name.split('_')[1]],
                                                    'textured_simple.obj')

                    gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)
                    obj_rots_gt = data.obj_rots_gt.cpu().detach().numpy().reshape(3,3)
                    obj_trans_gt = data.obj_trans_gt.cpu().detach().numpy().reshape(3,)

                    transform_matrix = np.eye(4)  # 创建一个 4x4 单位矩阵
                    transform_matrix[:3, :3] = obj_rots_gt  # 将旋转矩阵填充到 4x4 矩阵的左上角
                    transform_matrix[:3, 3] = obj_trans_gt  # 将平移向量填充到 4x4 矩阵的最后一列

                    # 应用变换矩阵
                    gt_obj_mesh.apply_transform(transform_matrix)
                    gt_obj_mesh.export(out_path)


                    progress_bar.update(1)


        progress_bar.close()

@hydra.main(version_base=None, config_path="configs", config_name="config_ho3d")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    config.exp_dir = config.get('exp_dir',os.path.join('/home/cyc/pycharm/lxy/3DGS/results', config.name))
    os.makedirs(config.exp_dir, exist_ok=True)
    #config.dataset.train_sample_rate=10

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument("--ckpt_path", default='/mnt/sda1/lxy/HOGS_results/ho3d-hogs_prerigid_HOLD/')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--near", default=0.01, type=float)
    parser.add_argument("--far", default=100, type=float)

    args = parser.parse_args()
    print("Rendering " + os.path.abspath(args.ckpt_path))

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    #torch.cuda.set_device(1)

    extract_mesh(config, args.ckpt_path, args.filter_mesh, args.texture_mesh, args.near, args.far)


if __name__ == "__main__":
    main()