#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import argparse

import hydra
import yaml
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue
import pandas as pd
import trimesh
from scipy.spatial import cKDTree as KDTree
import shutil

_YCB_CLASSES = {
   1: '002_master_chef_can',
   2: '003_cracker_box',
   3: '004_sugar_box',
   4: '005_tomato_soup_can',
   5: '006_mustard_bottle',
   6: '007_tuna_fish_can',
   7: '008_pudding_box',
   8: '009_gelatin_box',
   9: '010_potted_meat_can',
   10: '011_banana',
   11: '019_pitcher_base',
   12: '021_bleach_cleanser',
   13: '024_bowl',
   14: '025_mug',
   15: '035_power_drill',
   16: '036_wood_block',
   17: '037_scissors',
   18: '040_large_marker',
   19: '051_large_clamp',
   20: '052_extra_large_clamp',
   21: '061_foam_brick',
}

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dir', '-e', default='../result/dexycb-hogs_gpu1_without_kpTR/point_cloud/iteration_360000/zero1k_avg_mesh_0.7/', type=str)
    parser.add_argument('--dir', '-e',default='../results/dexycb-hogs_util30k_delay0_color.3/point_cloud/iteration_360000/',type=str)
    parser.add_argument('--testset', '-testset', default='dexycb', type=str)
    parser.add_argument('--model_dir', default='../lib/YCB_models/', type=str)
    parser.add_argument('--num_proc', default=10, type=int)
    args = parser.parse_args()

    return args

import pyfqmr

def evaluate( mesh_dir, model_dir):
    error_dict = {'chamfer_obj':[],'fscore_obj_5':[],'fscore_obj_10':[]}
    for obj_file_name in os.listdir(mesh_dir):
        if 'ply' not in obj_file_name or 'obj' not in obj_file_name :
            continue
        obj_id = obj_file_name.split('_')[-1][:-4]
        ycb_id = _YCB_CLASSES[int(obj_id)]
        pred_obj_mesh_path = os.path.join(mesh_dir, obj_file_name)
        gt_obj_mesh_path = os.path.join(model_dir, ycb_id, 'textured_simple.obj')

        pred_obj_mesh = trimesh.load(pred_obj_mesh_path, process=False)
        gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)
        #

        simply=True
        target_count = 7000
        if simply:

            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(pred_obj_mesh.vertices, pred_obj_mesh.faces)
            mesh_simplifier.simplify_mesh(target_count=target_count, aggressiveness=7, preserve_border=True, verbose=True)
            #mesh_simplifier.simplify_mesh(target_count=target_count)
            vertices, faces, normals = mesh_simplifier.getMesh()
            pred_obj_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            #pred_obj_mesh = trimesh.smoothing.filter_laplacian(pred_obj_mesh, lamb=0.1, iterations=10)
            #os.makedirs(os.path.dirname(mesh_dir)+'/mesh0.6_7k/',exist_ok=True)

            #pred_obj_mesh.export(os.path.dirname(mesh_dir)+'/mesh0.6_7k/'+obj_file_name)

            print(pred_obj_mesh.faces.shape)
            #exit()


        pred_obj_points, _ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
        gt_obj_points, _ = trimesh.sample.sample_surface(gt_obj_mesh, 30000)


        pred_obj_points *= 100.
        gt_obj_points *= 100.

        # one direction
        gen_points_kd_tree = KDTree(pred_obj_points)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
        # other direction
        gt_points_kd_tree = KDTree(gt_obj_points)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer

        threshold = 0.5  # 5 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_obj_1 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        threshold = 1.0  # 10 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        error_dict['chamfer_obj'].append(chamfer_obj)
        #print(obj_id,ycb_id,chamfer_obj)
        error_dict['fscore_obj_5'].append(fscore_obj_1)
        error_dict['fscore_obj_10'].append(fscore_obj_5)
    return error_dict


# python eval.py -e /home/cyc/pycharm/lxy/gSDF/outputs_test/sdf_pcl/ -testset dexycb
def main():
    # argument parse and create log
    args = parse_args()

    for dir in os.listdir(args.dir):
        if 'mesh0.6' not in dir:
            continue
        print(dir)
        dir = os.path.join(args.dir,dir)
        #dir = args.dir
        eval_result = evaluate(dir, args.model_dir)

        mean_chamfer_obj = "mean obj chamfer: {}\n".format(np.mean(eval_result['chamfer_obj']))
        median_chamfer_obj = "median obj chamfer: {}\n".format(np.median(eval_result['chamfer_obj']))
        fscore_obj_1 = "f-score obj @ 5mm: {}\n".format(np.mean(eval_result['fscore_obj_5']))
        fscore_obj_5 = "f-score obj @ 10mm: {}\n".format(np.mean(eval_result['fscore_obj_10']))


        print(mean_chamfer_obj)
        print(median_chamfer_obj)
        print(fscore_obj_1)
        print(fscore_obj_5)






if __name__ == "__main__":
    main()
