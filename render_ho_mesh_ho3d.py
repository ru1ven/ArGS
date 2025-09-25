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


@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    #final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    final_alpha = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            if torch.any(torch.isnan(points)):
                raise Exception("Tensor contains NaN values.")

            if torch.any(torch.isinf(points)):
                raise Exception("Tensor contains Inf values.")
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            render = ret["render"]
            full_image = torch.clamp(render, 0.0, 1.0)*255
            #print(full_image.shape)
            if return_color:
                color_integrated = ret["color_integrated"]
                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated,
                                          final_color)


            #final_alpha = torch.min(final_alpha, alpha_integrated)
            #final_alpha = final_alpha - torch.log(alpha_integrated+ 1e-6)
            #final_alpha = final_alpha-(torch.log(alpha_integrated)/(alpha_integrated+ 1e-6)).long()

            if torch.any(torch.isnan(final_alpha)):
                print("Tensor contains NaN values.")
            if torch.any(torch.isinf(final_alpha)):
                print("Tensor contains Inf values.")


            alpha_integrated[alpha_integrated==0]=-100000
            final_alpha = final_alpha+alpha_integrated

        final_alpha/=len(views)

        #alpha = final_alpha

        alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha



@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, views, gaussians, pipeline, background,
                                           kernel_size, ther, filter_mesh: bool, texture_mesh: bool, near: float, far: float, trans=np.array(.0)):
    render_path = os.path.join(model_path, 'sample10_mesh_posed_16_'+str(ther), 'sdf_mesh')

    makedirs(render_path, exist_ok=True)

    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points(views, near, far)
    # load cell if exists
    # if os.path.exists(os.path.join(render_path, "cells.pt")):
    #     print("load existing cells")
    #     cells = torch.load(os.path.join(render_path, "cells.pt"))
    # else:
    #     # create cell and save cells
    #     print("create cells and save")
    cells = cpp.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        #torch.save(cells, os.path.join(render_path, "cells.pt"))

    # evaluate alpha

    alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    #print(vertices.shape, tets.shape, alpha.shape)

    def alpha_to_sdf(alpha, ther=0.4):
        sdf = alpha - ther
        sdf[sdf > 1] = 1
        sdf[sdf < -1] = -1
        sdf = sdf[None]
        return sdf

    sdf = alpha_to_sdf(alpha)

    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()

    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]

    faces = faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.

    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    n_binary_steps = 6
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)

        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

        points = (left_points + right_points) / 2
        if step not in [5]:
            continue

        if texture_mesh:
            _, color = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=True)
            vertex_colors = (color.cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors = None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        #mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.15, iterations=10)

        # filter
        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)

        # if pose_deform:
        #     mesh = pose_deform(points,)
        mesh.vertices += trans
        mesh.export(os.path.join(render_path, name))

    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))

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

        #testLoader_6k = DataLoader(scene.test_dataset_6k, batch_size=1, shuffle=False,num_workers=0)
        progress_bar = tqdm(range(scene.test_dataset.__len__()), desc="Training progress")
        thers = [0.999]
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

                    if 'GSF' not in file_name:
                        continue


                    pc_hand, pc_obj, loss_reg, colors_precomp, obj_colors_precomp, updated_camera, canonical_hand, canonical_obj = \
                        scene.convert_gaussians(data, 200001, compute_loss=True, pose_refine=False, delay=False)
                    pc_hand._xyz -= updated_camera.hand_root.detach()
                    pc_obj._xyz -= updated_camera.obj_trans.detach()

                    render_pkg = render(data, 200001, scene, pipeline, background, compute_loss=True, return_opacity=True,
                                        pose_refine=False)
                    render_image = render_pkg["full_render"]
                    image = torch.clamp(render_image, 0.0, 1.0) * 255
                    #print(image.shape)
                    image = image.permute(1,2,0).cpu().numpy().astype(np.uint8)  # 转换为整数类型
                    # 使用 OpenCV 保存图像
                    #cv2.imwrite('/home/cyc/pycharm/lxy/3DGS/debug/mesh/query_render_posed.png', image)


                    cams = QueryCamerasLoader(scene.metadata_obj[obj_id]['obj_aabb'], cam_num=32).get_cam
                    #cams = QueryCamerasLoader(scene.metadata_obj[obj_id]['obj_aabb'], trans=torch.tensor(.0),cam_num=12).get_cam
                    # marching_tetrahedra_with_binary_search(ply_path, file_name+"_obj_{}.ply".format(obj_id), cams, canonical_obj, pipeline,
                    #                                        background, kernel_size, ther, filter_mesh, texture_mesh, near, far, trans=updated_camera.obj_trans.detach().cpu().numpy())

                    marching_tetrahedra_with_binary_search(ply_path, file_name + "_obj.ply", cams,
                                                           pc_obj, pipeline,
                                                           background, kernel_size, ther, filter_mesh, texture_mesh,
                                                           near, far,
                                                           trans=updated_camera.obj_trans.detach().cpu().numpy())

                    # cams = QueryCamerasLoader(scene.metadata['aabb'], cam_num=16).get_cam
                    # #cams = QueryCamerasLoader(scene.metadata['aabb'], trans=torch.tensor(.0),cam_num=12).get_cam
                    # marching_tetrahedra_with_binary_search(ply_path, file_name+"_hand.ply", cams, pc_hand, pipeline,
                    #                                        background, kernel_size, ther, filter_mesh, texture_mesh, near, far, trans=updated_camera.hand_root.detach().cpu().numpy())

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