#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

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

# mesh_paths = [
#     "/mnt/sda2/lxy/ARGS_results/box/arctic_box-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/mixer/arctic_mixer-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/waffleiron/arctic_waffleiron-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/phone/arctic_phone-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/ketchup/arctic_ketchup-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/espressomachine/arctic_espressomachine-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/microwave/arctic_microwave-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/scissors/arctic_scissors-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/laptop/arctic_laptop-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/capsulemachine/arctic_capsulemachine-vis_ours/vis/",
#     "/mnt/sda2/lxy/ARGS_results/notebook/arctic_notebook-vis_ours/vis/",
# ]

#tag = '3dgs-avatar'
tag = 'vis_ours'

mesh_paths = [

    "/mnt/sda2/lxy/ARGS_results/box/arctic_box-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/mixer/arctic_mixer-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/waffleiron/arctic_waffleiron-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/phone/arctic_phone-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/ketchup/arctic_ketchup-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/espressomachine/arctic_espressomachine-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/microwave/arctic_microwave-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/scissors/arctic_scissors-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/laptop/arctic_laptop-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/capsulemachine/arctic_capsulemachine-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/notebook/arctic_notebook-{}/vis/".format(tag),
    
]

test_splits = [
    "s01/box_use_02/1",
    "s01/mixer_use_01/1",
    "s01/waffleiron_use_01/1",
    "s01/phone_use_01/1",
    "s01/ketchup_use_02/1",
    "s01/espressomachine_use_01/3",
    "s01/microwave_use_01/6",
    "s01/scissors_use_01/1",
    "s01/laptop_use_01/1",
    "s01/capsulemachine_use_01/1",
    "s01/notebook_use_01/1",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_proc', default=10, type=int)
    args = parser.parse_args()

    return args


def evaluate(mesh_path, obj_gt_path, obj_gt_names):
    chamfers_obj = []
    fscores_obj_5 = []
    fscores_obj_10 = []

    for idx, obj_gt_filename in tqdm(enumerate(obj_gt_names)):
        mesh_obj = trimesh.load(os.path.join(mesh_path,'point_cloud', f'iteration_{idx}', 'fused_mesh.ply'), process=False)
        mesh_obj_gt = trimesh.load(os.path.join(obj_gt_path, obj_gt_filename + '.obj'), process=False)

        coord_min = np.min(mesh_obj_gt.vertices, axis=0)
        coord_max = np.max(mesh_obj_gt.vertices, axis=0)
        center = (coord_min + coord_max) / 2

        # 模型居中
        mesh_obj_gt.vertices -= center

        # ICP alignment
        #pred_obj_mesh = mesh_obj
        icp_solver = icp_ts(mesh_obj, mesh_obj_gt)
        icp_solver.sample_mesh(30000, 'both')
        icp_solver.run_icp_f(max_iter=100)
        pred_obj_mesh = icp_solver.get_source_mesh()

        if idx == 0:
            pred_obj_mesh.export(os.path.join(mesh_path,'point_cloud', f'iteration_{idx}', 'mesh_aligned.ply'))
            mesh_obj_gt.export(os.path.join(mesh_path,'point_cloud', f'iteration_{idx}', 'mesh_gt.ply'))

        pred_obj_points, _ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
        gt_obj_points, _ = trimesh.sample.sample_surface(mesh_obj_gt, 30000)
        pred_obj_points *= 100.
        gt_obj_points *= 100.

        # one direction
        gen_points_kd_tree = KDTree(pred_obj_points)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
        gt_to_gen_chamfer_sqrt = np.mean(one_distances)
        # other direction
        gt_points_kd_tree = KDTree(gt_obj_points)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        gen_to_gt_chamfer_sqrt = np.mean(two_distances)
        chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
        chamfer_obj_sqrt = gt_to_gen_chamfer_sqrt + gen_to_gt_chamfer_sqrt
        print(chamfer_obj)
        print(chamfer_obj_sqrt)

        threshold = 0.5 # 5 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        threshold = 1.0 # 10 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        chamfers_obj.append(chamfer_obj)
        fscores_obj_5.append(fscore_obj_5)
        fscores_obj_10.append(fscore_obj_10)
    return chamfers_obj, fscores_obj_5, fscores_obj_10




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
        os.makedirs(os.path.join("/mnt/sda2/lxy/ARGS_results/mesh_results",tag), exist_ok=True)
        summary_filename = os.path.join("/mnt/sda2/lxy/ARGS_results/mesh_results",tag, "eval_result_{}.txt".format(test_splits[i]).replace("/","_"))

        with open(summary_filename, "w") as f:
            eval_result = [[] for i in range(3)]
            name_list = ['sample_id',  'chamfer obj', 'fs_obj@5mm', 'fs_obj@10mm']
            data_list = []
            for idx, obj_name in enumerate(obj_gt_names):
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
