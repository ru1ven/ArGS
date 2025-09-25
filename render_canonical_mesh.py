import cv2
import hydra
import torch
import os

from torch import nn

from models.network_utils import VanillaCondMLP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os import makedirs

from omegaconf import OmegaConf

from scene.query_cam_loader import QueryCamerasLoader
from gaussian_renderer import integrate
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


            alpha_integrated[alpha_integrated==0]=-1000
            final_alpha = final_alpha+alpha_integrated

        final_alpha/=len(views)

        #alpha = final_alpha

        alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha



@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, views, gaussians, pipeline, background,
                                           kernel_size, ther, filter_mesh: bool, texture_mesh: bool, near: float, far: float):
    render_path = os.path.join(model_path, 'mesh_'+str(ther))

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

    print(vertices.shape, tets.shape, alpha.shape)

    def alpha_to_sdf(alpha, ther=0.5):
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

    n_binary_steps = 8
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
        if step not in [7]:
            continue

        if texture_mesh:
            _, color = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=True)
            vertex_colors = (color.cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors = None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)

        # filter
        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)

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

        load_ckpt = os.path.join(ckpt_path, "ckpt" + str(config.opt.iterations) + ".pth")

        scene.load_checkpoint(load_ckpt)
        ply_path = os.path.join(ckpt_path,"point_cloud/iteration_360000/")
        # for sub_id in gaussians_hand_group:
        #     gaussians_hand_group[sub_id].load_ply(os.path.join(ply_path, "point_cloud_hand_{}.ply".format(sub_id)))
        # for obj_id in gaussians_obj_group:
        #     gaussians_obj_group[obj_id].load_ply(os.path.join(ply_path, "point_cloud_obj_{}.ply".format(obj_id)))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = 0.0

        first_iter = 0
        progress_bar = tqdm(range(first_iter, len(rendered_hand.keys())+len(rendered_obj.keys())), desc="Training progress")

        thers = [0.4]
        for ther in thers:
            for obj_id in dataset._YCB_CLASSES:
                rendered_obj[int(obj_id)] = False
            for subject in dataset._SUBJECTS:
                rendered_hand[int(subject.split('-')[-1])] = False
            while any(value is False for value in rendered_hand.values()) \
                    or any(value is False for value in rendered_obj.values()):
                data_stack = list(range(len(scene.train_dataset)))
                data_idx = data_stack.pop(random.randint(0, len(data_stack) - 1))
                # data_idx = data_stack.pop(0)
                data = scene.train_dataset[data_idx]
                obj_id = data.obj_id
                sub_id = data.subject_id
                if rendered_obj[obj_id] is False or rendered_hand[sub_id] is False:
                    canonical_hand, canonical_obj, color_precompute, objcolor_precompute = scene.get_canonical_gaussians(data)
                #print(scene.gaussians_hand_group[data.subject_id].get_xyz.shape)
                #canonical_obj = scene.gaussians_obj_group[data.obj_id]
                #print(pc_hand.get_xyz.mean(0))

                if rendered_obj[obj_id] is False:
                    cams = QueryCamerasLoader(scene.metadata_obj[obj_id]['obj_aabb']).get_cam
                    marching_tetrahedra_with_binary_search(ply_path, "mesh_obj_{}.ply".format(obj_id), cams, canonical_obj, pipeline,
                                                           background, kernel_size, ther, filter_mesh, texture_mesh, near, far)
                    rendered_obj[obj_id] = True
                    progress_bar.update(1)

                if rendered_hand[sub_id] is False:
                    cams = QueryCamerasLoader(scene.metadata['aabb']).get_cam
                    marching_tetrahedra_with_binary_search(ply_path, "mesh_hand_{}.ply".format(sub_id), cams, canonical_hand, pipeline,
                                                           background, kernel_size, ther, filter_mesh, texture_mesh, near, far)
                    rendered_hand[sub_id] = True
                    progress_bar.update(1)


        progress_bar.close()



        # cams = QueryCamerasLoader(scene.metadata['aabb']).get_cam
        #
        # marching_tetrahedra_with_binary_search('/home/cyc/pycharm/lxy/3DGS/debug/', "mesh_hand_1.ply", cams,
        #                                        gaussians_hand_group[1], pipeline,
        #                                        background, kernel_size, filter_mesh, texture_mesh, near, far)
        #cams = QueryCamerasLoader(scene.metadata_obj[obj_id]['obj_aabb']).get_cam
        # marching_tetrahedra_with_binary_search('/home/cyc/pycharm/lxy/3DGS/debug/', "mesh_obj_1.ply", cams,
        #                                        gaussians_obj_group[1], pipeline,
        #                                        background, kernel_size, filter_mesh, texture_mesh, near, far)

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
    config.exp_dir = config.get('exp_dir',os.path.join('/home/cyc/pycharm/lxy/3DGS/results', config.name))
    os.makedirs(config.exp_dir, exist_ok=True)
    config.dataset.train_sample_rate=10
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument("--ckpt_path", default='/mnt/sda1/lxy/HOGS_results/dexycb-hogs_prerigid_centeredbbox/')
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