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


def evaluate(mesh_path, obj_radians_gt):
    
    aae = []

    for idx, obj_radian_gt in tqdm(enumerate(obj_radians_gt)):
        pivot = np.load(os.path.join(mesh_path,'articulation', f'iteration_{idx}', 'pivot.npy'))
        pred_radian = np.load(os.path.join(mesh_path,'articulation', f'iteration_{idx}', 'axis.npy'))
        pred_degree = pred_radian / math.pi * 180  # degree
        gt_degree = gt_radian / math.pi * 180  # degree
        err_deg = np.abs(pred_degree - gt_degree).tolist()
        
        aae.append(np.array(err_deg, dtype=np.float32))
    return aae




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
        
        imgnames = meta_data["imgnames"]
        obj_radians_gt = []
        
        for imgname in imgnames:
            sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
            vidx = int(image_idx.split(".")[0]) - ioi_offset
            print(image_idx)
            seq_data = meta_data['data_dict'][f"{sid}/{seq_name}"]
            data_params = seq_data["params"]
            obj_radian = data_params["obj_arti"][vidx].copy()
            if vidx % 2 == 1:
                obj_radians_gt.append(obj_radian)

        aae = evaluate(mesh_path, obj_radians_gt)
       
        os.makedirs(os.path.join("/mnt/sda2/lxy/ARGS_results/mesh_results",tag), exist_ok=True)
        summary_filename = os.path.join("/mnt/sda2/lxy/ARGS_results/mesh_results",tag, "eval_articulated_{}.txt".format(test_splits[i]).replace("/","_"))

        with open(summary_filename, "w") as f:
            
           
            aae = "AAE : {}\n".format(np.mean(aae))
           
            print(aae); f.write(aae)
           

if __name__ == "__main__":
    main()
