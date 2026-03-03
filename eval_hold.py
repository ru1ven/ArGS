#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import copy
import json
import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm

from multiprocessing import Process, Queue
import pandas as pd
import trimesh
import shutil
from scipy.spatial import cKDTree as KDTree
from utils.solver import icp_rts, icp_ts
import open3d as o3d
from open3d.geometry import TriangleMesh

mesh_paths = [
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/box/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/mixer/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/waffleiron/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/ketchup/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/espressomachine/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/microwave/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/laptop/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/capsulemachine/mesh_cano/mesh_cano_object_step_misc.obj",
    "/mnt/sda2/lxy/HOLD_results/arctic_ckpts/notebook/mesh_cano/mesh_cano_object_step_misc.obj",
    
]



test_splits = [
    "s01/box_use_02/1",
    "s01/mixer_use_01/1",
    "s01/waffleiron_use_01/1",
    "s01/ketchup_use_02/1",
    "s01/espressomachine_use_01/3",
    "s01/microwave_use_01/6",
    "s01/laptop_use_01/1",
    "s01/capsulemachine_use_01/1",
    "s01/notebook_use_01/1",
]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_proc', default=10, type=int)
    args = parser.parse_args()

    return args

import copy

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
import copy

from tqdm import tqdm


def calculate_metrics(aligned_mesh, target_mesh, is_sqrt=False):
    vertices_source = np.asarray(aligned_mesh.vertices) * 100
    vertices_target = np.asarray(target_mesh.vertices) * 100

    # dist_bidirectional = chamferDist(vertices_source, vertices_target, bidirectional=True,point_reduction = "mean") #* 0.001
    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)

    if is_sqrt:  # square-root chamfer
        gt_to_gen_chamfer = np.mean(one_distances)
    else:  # squared chamfer
        gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)

    if is_sqrt:
        gen_to_gt_chamfer = np.mean(two_distances)
    else:
        gen_to_gt_chamfer = np.mean(np.square(two_distances))

    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = 0.01 # 1#voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source, target, source_fpfh, target_fpfh, voxel_size, init_alignment
):
    distance_threshold =0.01  # voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)

    # result_ransac.transformation
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_alignment,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
    )
    return result


def compute_icp_metrics(
    target_mesh, source_mesh, num_iters=60, no_tqdm=False, is_sqrt=False
):
    voxel_size = 0.005
    # print(":: Load two meshes")
    # exp_id = "42465ff7e"

    # assert os.path.exists(target_deform_p)
    # assert os.path.exists(pred_deform_p)
    # target_mesh = o3d.io.read_triangle_mesh(target_deform_p)
    # source_mesh = o3d.io.read_triangle_mesh(pred_deform_p)

    center_mass = source_mesh.get_center()
    source_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(source_mesh.vertices) - center_mass
    )
    source_copy = copy.deepcopy(source_mesh)
    center_mass = target_mesh.get_center()
    target_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(target_mesh.vertices) - center_mass
    )

    # o3d.io.write_triangle_mesh("target.obj", target_mesh)
    # o3d.io.write_triangle_mesh("source.obj", source_mesh)

    # print(":: Sample mesh to point cloud")
    target = target_mesh.sample_points_uniformly(1000)
    source = source_mesh.sample_points_uniformly(1000)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    trans_init = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result_icp_ra = refine_registration(
        source, target, source_fpfh, target_fpfh, voxel_size, trans_init
    )

    # Apply the transformation to align the source mesh with the target
    aligned_source_mesh = source_copy.transform(result_icp_ra.transformation)


    cd_ra, f5_ra, f10_ra = calculate_metrics(aligned_source_mesh, target_mesh, is_sqrt)
    best_cd = cd_ra
    best_f5 = f5_ra
    best_f10 = f10_ra

    if no_tqdm:
        pbar = range(num_iters)
    else:
        pbar = tqdm(range(num_iters))
    # for iter in tqdm(range(num_iters)):
    for iter in pbar:
        try:
            result_ransac = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size
            )

            result_icp_nra = refine_registration(
                source,
                target,
                source_fpfh,
                target_fpfh,
                voxel_size,
                result_ransac.transformation,
            )
        except:
            print("Error in ICP: Skipping")
        aligned_source_mesh_ransac = copy.deepcopy(source_mesh).transform(
            result_icp_nra.transformation
        )

        cd_nra, f5_nra, f10_nra = calculate_metrics(
            aligned_source_mesh_ransac, target_mesh
        )
        # print(cd_nra)
        if cd_nra < best_cd:
            best_cd, best_f5, best_f10 = cd_nra, f5_nra, f10_nra

    # print("CD F5 F10", best_cd, best_f5, best_f10 )
    return best_cd, best_f5, best_f10, aligned_source_mesh, target_mesh

import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from pytorch3d.loss import chamfer_distance
import sys

sys.path = [".."] + sys.path
from tqdm import tqdm
import common.metrics as metrics


def compute_bounding_box_centers(vertices):
    """
    Compute the centers of the tight bounding box for a moving point cloud.

    Parameters:
    - vertices: A numpy array of shape (frames, num_verts, 3) representing the vertices of the object over time.

    Returns:
    - A numpy array of shape (frames, 3) where each row represents the center of the bounding box for each frame.
    """

    if isinstance(vertices, list):
        bbox_centers = []
        for verts in vertices:
            assert verts.shape[1] == 3
            bmin = np.min(verts, axis=0)
            bmax = np.max(verts, axis=0)
            bbox_center = (bmin + bmax) / 2
            bbox_centers.append(bbox_center)
        bbox_centers = np.stack(bbox_centers, axis=0)
    else:
        bbox_min = np.min(vertices, axis=1)
        bbox_max = np.max(vertices, axis=1)
        bbox_centers = (bbox_min + bbox_max) / 2
    return bbox_centers


def convert_to_tensors(data):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint32:
                data[key] = torch.from_numpy(value.astype(np.int64))
            elif value.dtype.kind == "f":  # Check if it's a floating-point type
                data[key] = torch.from_numpy(
                    value.astype(np.float32)
                )  # Convert to Float32
            else:
                data[key] = torch.from_numpy(value)
    return data


def eval_icp_first_frame_arctic(data_pred, data_gt, metric_dict):
    """
    Warning:
    In the CVPR HOLD paper, we used HO3D and followed the evaluation protocol from IHOR.
    The protocol uses squared chamfer distance (dist**2) version for evaluation.
    However, this metric is not in the metric space. Therefore, for ARCTIC here, we use squared-root CD (dist) for evaluation.
    """
    faces = data_pred["faces"]["object"]
    
    from open3d.geometry import TriangleMesh
    from open3d.utility import Vector3dVector, Vector3iVector

    v3d_o_ra = Vector3dVector(data_pred["v3d_ra.object"][0].numpy())
    faces_o = Vector3iVector(faces.cpu().numpy())
    v3d_o_ra_gt = Vector3dVector(data_gt["v3d_ra.object"][0].numpy())
    faces_o_gt = Vector3iVector(data_gt["faces_o"].numpy())
    source_mesh = TriangleMesh(v3d_o_ra, faces_o)
    target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)
    best_cd, best_f5, best_f10 = compute_icp_metrics(
        # target_mesh, source_mesh, num_iters=60, is_sqrt=True
        target_mesh,
        source_mesh,
        num_iters=600,
        is_sqrt=True,
    )
    metric_dict["cd_icp"] = best_cd
    metric_dict["f5_icp"] = best_f5 * 100.0
    metric_dict["f10_icp"] = best_f10 * 100.0
    return metric_dict



def eval_mrrpe_ho_right(data_pred, data_gt, metric_dict):
    j3d_h_c_pred = data_pred["j3d_c.right"]
    root_o_pred = data_pred["root.object"]

    j3d_h_c_gt = data_gt["j3d_c.right"]
    root_o_gt = data_gt["root.object"]
    is_valid = data_gt["is_valid"]

    root_h_gt = j3d_h_c_gt[:, 0]
    root_h_pred = j3d_h_c_pred[:, 0]
    mrrpe_ho = (
        metrics.compute_mrrpe(
            root_h_gt,
            root_o_gt,
            root_h_pred,
            root_o_pred,
            is_valid,
        )
        * 1000
    )
    not_valid = (1 - is_valid).numpy().astype(bool)
    mrrpe_ho[not_valid] = np.nan

    metric_dict["mrrpe_ho"] = mrrpe_ho
    return metric_dict


def calculate_chamfer_f_scores(vertices_source, vertices_target, is_sqrt=False):
    vertices_source = vertices_source * 100
    vertices_target = vertices_target * 100

    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)
    if is_sqrt:
        gt_to_gen_chamfer = np.mean(one_distances)
    else:
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
    # other direction
    gt_points_kd_tree = KDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)
    if is_sqrt:
        gen_to_gt_chamfer = np.mean(two_distances)
    else:
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10


def compute_iou_per_frame(insta_map_pred, insta_map_gt):
    classes = [0, 100, 200]
    ious = []

    for frame_idx in range(insta_map_pred.shape[0]):
        iou_per_class = []
        for cls in classes:
            pred_mask = insta_map_pred[frame_idx] == cls
            gt_mask = insta_map_gt[frame_idx] == cls
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union != 0 else 0
            iou_per_class.append(iou)
        ious.append(
            np.mean(iou_per_class)
        )  # Assuming you want the mean IoU for all classes per frame

    return np.array(ious)


def eval_cd_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c_ra"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c_ra"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_ra = (
        chamfer_distance(
            v3d_o_c_pred_ra[:, rand_pred_idx],
            v3d_o_c_gt_ra[:, rand_gt_idx],
            batch_reduction=None,
        )[0]
        * 1000
    )

    metric_dict["cd_ra"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    return metric_dict


def eval_cd_f(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(v3d_o_c_pred_ra.shape[0]):
        cd_error, f5, f10 = calculate_chamfer_f_scores(
            v3d_o_c_pred_ra[idx, rand_pred_idx].numpy(),
            v3d_o_c_gt_ra[idx, rand_gt_idx].numpy(),
        )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd"] = cd_list
    metric_dict["f5"] = f5_list * 100.0
    metric_dict["f10"] = f10_list * 100.0
    return metric_dict


def eval_cd_f_right_arctic(data_pred, data_gt, metric_dict):
    return eval_cd_f_arctic(data_pred, data_gt, metric_dict, "right")


def eval_cd_f_left_arctic(data_pred, data_gt, metric_dict):
    return eval_cd_f_arctic(data_pred, data_gt, metric_dict, "left")


def eval_cd_f_hand_arctic(data_pred, data_gt, metric_dict):
    eval_cd_f_left_arctic(data_pred, data_gt, metric_dict)
    eval_cd_f_right_arctic(data_pred, data_gt, metric_dict)
    cd_hand = np.stack([metric_dict["cd_r"], metric_dict["cd_l"]], axis=1)
    metric_dict["cd_h"] = cd_hand.mean(axis=1)
    return metric_dict


def eval_cd_f_arctic(data_pred, data_gt, metric_dict, flag):
    v3d_o_c_pred_ra = data_pred[f"v3d_{flag}.object"]
    v3d_o_c_gt_ra = data_gt[f"v3d_{flag}.object"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]
        v3d_gt = v3d_o_c_gt_ra[idx]

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:3000]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(),
                v3d_gt[rand_gt_idx].numpy(),
                is_sqrt=True,
            )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    indicator = "r" if flag == "right" else "l"
    metric_dict[f"cd_{indicator}"] = cd_list
    # metric_dict[f"f5_{indicator}"] = f5_list * 100.0
    # metric_dict[f"f10_{indicator}"] = f10_list * 100.0
    return metric_dict


def eval_cd_f_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_ra.object"]
    v3d_o_c_gt_ra = data_gt["v3d_ra.object"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)
    # rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    # rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            v3d_gt = v3d_o_c_gt_ra[idx]

            num_pts = min(3000, v3d_pred.shape[0])
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:num_pts]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(), v3d_gt[rand_gt_idx].numpy()
            )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd_ra"] = cd_list
    metric_dict["f5_ra"] = f5_list * 100.0
    metric_dict["f10_ra"] = f10_list * 100.0
    return metric_dict


def eval_mpjpe_right(data_pred, data_gt, metric_dict):
    return eval_mpjpe(data_pred, data_gt, metric_dict, "right")


def eval_mpjpe_left(data_pred, data_gt, metric_dict):
    return eval_mpjpe(data_pred, data_gt, metric_dict, "left")


def eval_mpjpe_hand(data_pred, data_gt, metric_dict):
    eval_mpjpe(data_pred, data_gt, metric_dict, "left")
    eval_mpjpe(data_pred, data_gt, metric_dict, "right")
    mpjpe_hand = np.stack(
        [metric_dict["mpjpe_ra_l"], metric_dict["mpjpe_ra_r"]], axis=1
    )
    metric_dict["mpjpe_ra_h"] = mpjpe_hand.mean(axis=1)
    return metric_dict


def eval_mpjpe(data_pred, data_gt, metric_dict, flag):
    j3d_h_c_pred_ra = data_pred[f"j3d_ra.{flag}"]
    j3d_h_c_gt_ra = data_gt[f"j3d_ra.{flag}"]
    is_valid = data_gt["is_valid"]

    mpjpe_ra = metrics.compute_joint3d_error(j3d_h_c_gt_ra, j3d_h_c_pred_ra, is_valid)
    mpjpe_ra = mpjpe_ra.mean(axis=1) * 1000  # Use dim instead of axis for PyTorch

    indicator = "r" if flag == "right" else "l"
    metric_dict[f"mpjpe_ra_{indicator}"] = mpjpe_ra
    return metric_dict


def eval_ious(data_pred, data_gt, metric_dict):
    masks_pred = data_pred["masks_pred"].long().numpy()
    masks_gt = data_gt["masks_gt"].long().numpy()
    is_valid = data_gt["is_valid"]
    ious = compute_iou_per_frame(masks_pred, masks_gt)
    not_valid = (1 - is_valid).numpy().astype(bool)
    ious[not_valid] = np.nan
    metric_dict["ious"] = ious * 100.0
    return metric_dict



def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh

def normalize_to_gt(source_mesh, target_mesh):
    """
    将 source（预测）mesh 归一化到 target（GT）的尺度空间
    但不改变 target 自身的坐标系
    """
    # 获取 GT 尺度
    target_vertices = np.asarray(target_mesh.vertices)
    target_scale = np.linalg.norm(target_vertices, axis=1).max()
    target_center = target_vertices.mean(axis=0)

    # 归一化预测 mesh
    src_vertices = np.asarray(source_mesh.vertices)
    src_center = src_vertices.mean(axis=0)
    src_vertices = src_vertices - src_center

    # 将预测缩放到与GT相同的尺度
    src_scale = np.linalg.norm(src_vertices, axis=1).max()
    scale_factor = target_scale / src_scale
    src_vertices = src_vertices * scale_factor + target_center

    source_mesh.vertices = o3d.utility.Vector3dVector(src_vertices)
    return source_mesh, scale_factor, src_center, target_center



def evaluate_canonical(mesh_path, obj_gt_path, obj_gt_names):
   
    source_mesh = o3d.io.read_triangle_mesh(mesh_path)
    target_mesh = o3d.io.read_triangle_mesh(os.path.join(obj_gt_path, obj_gt_names[0] + '.obj'))

    center_mass = source_mesh.get_center()
    source_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(source_mesh.vertices) - center_mass
    )
    
    center_mass = target_mesh.get_center()
    
    target_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(target_mesh.vertices) - center_mass
    )
    source_mesh, scale_factor, s_center, t_center = normalize_to_gt(source_mesh, target_mesh)
    source_copy = copy.deepcopy(source_mesh)

    best_cd, best_f5, best_f10, aligned_source_mesh, target_mesh = compute_icp_metrics(
        # target_mesh, source_mesh, num_iters=60, is_sqrt=True
        target_mesh,
        source_mesh,
        num_iters=600,
        is_sqrt=True,
    )

    print(best_cd)

    o3d.io.write_triangle_mesh(mesh_path.replace("mesh_cano_object_step_misc.obj",'mesh_aligned.ply'), aligned_source_mesh)
    o3d.io.write_triangle_mesh(mesh_path.replace("mesh_cano_object_step_misc.obj",'mesh_gt.ply'), target_mesh)


    return best_cd, best_f5, best_f10, aligned_source_mesh, target_mesh
    

def evaluate(mesh_path, obj_gt_path, obj_gt_names):
    chamfers_obj = []
    fscores_obj_5 = []
    fscores_obj_10 = []

    best_cd, best_f5, best_f10, mesh_canonical, target_mesh = evaluate_canonical(mesh_path, obj_gt_path, obj_gt_names)

    # for idx, obj_gt_filename in tqdm(enumerate([obj_gt_names[0]])):
        
        #mesh_obj = trimesh.load(mesh_path, process=False)
        
        # vertices = np.asarray(mesh_canonical.vertices)
        # faces = np.asarray(mesh_canonical.triangles)
        # mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # vertices = np.asarray(target_mesh.vertices)
        # faces = np.asarray(target_mesh.triangles)
        # mesh_obj_gt = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # # apply GT Rotation

        # #mesh_obj_gt = trimesh.load(os.path.join(obj_gt_path, obj_gt_filename + '.obj'), process=False)
        
        # # # 模型居中
        # # coord_min = np.min(mesh_obj_gt.vertices, axis=0)
        # # coord_max = np.max(mesh_obj_gt.vertices, axis=0)
        # # center = (coord_min + coord_max) / 2
        # # mesh_obj_gt.vertices -= center

        # # coord_min = np.min(mesh_obj.vertices, axis=0)
        # # coord_max = np.max(mesh_obj.vertices, axis=0)
        # # center = (coord_min + coord_max) / 2
        # # mesh_obj.vertices -= center


        # # # ICP alignment
        # # #pred_obj_mesh = mesh_obj
        # # icp_solver = icp_ts(mesh_obj, mesh_obj_gt)
        # # icp_solver.sample_mesh(30000, 'both')
        # # icp_solver.run_icp_f(max_iter=100)
        # # pred_obj_mesh = icp_solver.get_source_mesh()

        # # if idx == 0:
        # #     pred_obj_mesh.export(os.path.join(mesh_path,'mesh_aligned.ply'))
        # #     mesh_obj_gt.export(os.path.join(mesh_path, 'mesh_gt.ply'))


        # pred_obj_points, _ = trimesh.sample.sample_surface(mesh_obj, 30000)
        # gt_obj_points, _ = trimesh.sample.sample_surface(mesh_obj_gt, 30000)
        # pred_obj_points *= 100.
        # gt_obj_points *= 100.

        # # one direction
        # gen_points_kd_tree = KDTree(pred_obj_points)
        # one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
        # gt_to_gen_chamfer = np.mean(np.square(one_distances))
        # # other direction
        # gt_points_kd_tree = KDTree(gt_obj_points)
        # two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
        # gen_to_gt_chamfer = np.mean(np.square(two_distances))
        # chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
        # print(chamfer_obj)

        # threshold = 0.5 # 5 mm
        # precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        # precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        # fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        # threshold = 1.0 # 10 mm
        # precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        # precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        # fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        # chamfers_obj.append(chamfer_obj)
        # fscores_obj_5.append(fscore_obj_5)
        # fscores_obj_10.append(fscore_obj_10)
        # print(chamfer_obj)
    return [best_cd], [best_f5], [best_f10]




# python eval.py
def main():
    for i, mesh_path in enumerate(mesh_paths):

        # argument parse and create log
        args = parse_args()
        
        meta = os.path.join("/mnt/sda2/lxy/dataset/hand/arctic_seqs/splits/train/{}.npy"
                            .format(test_splits[i].replace("/","_")))
        with open("/mnt/sda2/lxy/dataset/hand/arctic/meta/misc.json", "r") as f:
            misc = json.load(f)
    
        ioi_offset = misc['s01']["ioi_offset"]
        meta_data = np.load(meta, allow_pickle=True).item()

        obj_gt_names = []
        obj_gt_path = '/mnt/sda2/lxy/dataset/hand/arctic_seqs/mesh_obj/'+test_splits[i]
        
        imgnames = meta_data["imgnames"]
        
        for imgname in imgnames:
            sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
            vidx = int(image_idx.split(".")[0]) - ioi_offset
            if vidx % 2 == 1:
                obj_gt_names.append(image_idx.split(".")[0])

        chamfers_obj, fscores_obj_5, fscores_obj_10 = evaluate(mesh_path, obj_gt_path, obj_gt_names)
        #summary_filename = os.path.join(mesh+path, "eval_result_{}.txt".format(test_splits[i]).replace("/","_"))
        os.makedirs("/mnt/sda2/lxy/HOLD_results/", exist_ok=True)
        summary_filename = os.path.join("/mnt/sda2/lxy/HOLD_results/", "eval_result_{}.txt".format(test_splits[i]).replace("/","_"))

        with open(summary_filename, "w") as f:
            eval_result = [[] for i in range(3)]
            name_list = ['sample_id',  'chamfer obj', 'fs_obj@5mm', 'fs_obj@10mm']
            data_list = []
            for idx, obj_name in enumerate([obj_gt_names[0]]):
                result = chamfers_obj[idx], fscores_obj_5[idx], fscores_obj_10[idx]
                data_sample = [obj_name]
                for i in range(3):
                    eval_result[i].append(result[i])
                    data_sample.append(result[i])
                data_list.append(data_sample)
            f.write(pd.DataFrame(data_list, columns=name_list, index=[''] * len(obj_gt_names), dtype=str).to_string())
            f.write('\n')

            for idx, _ in enumerate(eval_result):
                new_array = []
                for number in eval_result[idx]:
                    if not np.isnan(number):
                        new_array.append(number)
                eval_result[idx] = new_array

           
            mean_chamfer_obj = "mean obj chamfer: {}\n".format(np.mean(eval_result[0]))
            median_chamfer_obj = "median obj chamfer: {}\n".format(np.median(eval_result[0]))
            fscore_obj_1 = "f-score obj @ 5mm: {}\n".format(np.mean(eval_result[1]))
            fscore_obj_5 = "f-score obj @ 10mm: {}\n".format(np.mean(eval_result[2]))
            
            print(mean_chamfer_obj); f.write(mean_chamfer_obj)
            print(median_chamfer_obj); f.write(median_chamfer_obj)
            print(fscore_obj_1); f.write(fscore_obj_1)
            print(fscore_obj_5); f.write(fscore_obj_5)
            
           

if __name__ == "__main__":
    main()
