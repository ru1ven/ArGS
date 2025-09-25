import cv2
import hydra
import torch
import os
from os import makedirs

from omegaconf import OmegaConf

from scene.query_cam_loader import QueryCamerasLoader
from gaussian_renderer import integrate, render_template
import random
from tqdm import tqdm
from argparse import ArgumentParser
from scene import Scene, GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra


@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for i, view in enumerate(tqdm(views, desc="Rendering progress")):
            render = render_template(view, gaussians, pipeline, background)

            render_image = render["render"]
            image = torch.clamp(render_image, 0.0, 1.0)*255
            print(image.shape)
            image = image.squeeze(0).cpu().numpy().astype(np.uint8)  # 转换为整数类型
            # 使用 OpenCV 保存图像
            cv2.imwrite('/home/cyc/pycharm/lxy/3DGS/debug/mesh/query_render_{}.png'.format(i), image)




@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, views, gaussians, pipeline, background,
                                           kernel_size, filter_mesh: bool, texture_mesh: bool, near: float, far: float):
    render_path = os.path.join(model_path, 'mesh')

    makedirs(render_path, exist_ok=True)

    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points(views, near, far)

    # evaluate alpha
    alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size)


    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))

def extract_mesh(config, ply_path, filter_mesh: bool, texture_mesh: bool,
                 near: float, far: float):
    with torch.no_grad():
        dataset = config.dataset
        model = config.model
        pipeline = config.pipeline

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

        for sub_id in gaussians_hand_group:
            gaussians_hand_group[sub_id] = GaussianModel(model.gaussian)
        for obj_id in gaussians_obj_group:
            gaussians_obj_group[obj_id] = GaussianModel(model.gaussian)

        scene = Scene(config, gaussians_hand_group, gaussians_obj_group, config.exp_dir)
        scene.eval()

        for sub_id in gaussians_hand_group:
            gaussians_hand_group[sub_id].load_ply(os.path.join(ply_path, "point_cloud_hand_{}.ply".format(sub_id)))
        for obj_id in gaussians_obj_group:
            gaussians_obj_group[obj_id].load_ply(os.path.join(ply_path, "point_cloud_obj_{}.ply".format(obj_id)))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = 0.0

        cams = QueryCamerasLoader(scene.metadata['aabb']).get_cam
        marching_tetrahedra_with_binary_search('/home/cyc/pycharm/lxy/3DGS/debug/', "mesh_hand_1.ply", cams,
                                               gaussians_hand_group[1], pipeline,
                                               background, kernel_size, filter_mesh, texture_mesh, near, far)
        marching_tetrahedra_with_binary_search('/home/cyc/pycharm/lxy/3DGS/debug/', "mesh_obj_1.ply", cams,
                                               gaussians_obj_group[1], pipeline,
                                               background, kernel_size, filter_mesh, texture_mesh, near, far)

        # for sub_id in gaussians_hand_group:
        #     cams = QueryCamerasLoader(scene.metadata['aabb']).get_cam
        #     marching_tetrahedra_with_binary_search(ply_path, "mesh_hand_{}.ply".format(sub_id), cams, gaussians_hand_group[sub_id], pipeline,
        #                                        background, kernel_size, filter_mesh, texture_mesh, near, far)
        # for obj_id in gaussians_obj_group:
        #     cams = QueryCamerasLoader(scene.metadata_obj[obj_id]['obj_aabb']).get_cam
        #     marching_tetrahedra_with_binary_search(ply_path, "mesh_obj_{}.ply".format(obj_id), cams, gaussians_obj_group[obj_id], pipeline,
        #                                        background, kernel_size, filter_mesh, texture_mesh, near, far)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    # print(config.name)
    config.exp_dir = config.get('exp_dir','../result/mesh_test/')
    os.makedirs(config.exp_dir, exist_ok=True)
    config.dataset.train_sample_rate=10
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument("--ply_path", default='../result/dexycb-hoisdf_test_sample10/point_cloud/iteration_360000/')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--near", default=0.01, type=float)
    parser.add_argument("--far", default=100, type=float)

    args = parser.parse_args()
    print("Rendering " + os.path.abspath(args.ply_path))

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    extract_mesh(config, args.ply_path, args.filter_mesh, args.texture_mesh, args.near, args.far)


if __name__ == "__main__":
    main()