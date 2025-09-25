#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import json
import multiprocessing
from distutils.log import debug
import numpy as np
import torch
import os
import cv2

import trimesh
from PIL import Image
from tqdm import tqdm

from common import data_utils
from common.body_models import build_mano_aa
from common.mesh import Mesh
from common.object_tensors import ObjectTensors
from right_hand_model import MANO
from utils.dataset_utils import apply_w2c_pose_numpy, apply_w2c_pose_with_center

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pytorch3d.structures import Meshes

from fire import Fire
import pickle
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras, OpenGLPerspectiveCameras, RasterizationSettings,MeshRasterizer
import sys
#
# from dataset.dataset_utils import load_object, decode_seq_cat, quat_to_rotmat, rotmat_to_aa, quat_to_aa

sys.path.insert(0, '..')
device='cuda'

def preprocess(data_root, subject, imgnames, data_dict):
    image_root = os.path.join(data_root, 'arctic/')

    hand_mesh_data_root = os.path.join(data_root, 'arctic_seqs',  'mesh_hand')
    obj_mesh_data_root = os.path.join(data_root,  'arctic_seqs',  'mesh_obj')
    seg_root = os.path.join(data_root,  'arctic_seqs', 'seg')
    os.makedirs(hand_mesh_data_root, exist_ok=True)
    os.makedirs(obj_mesh_data_root, exist_ok=True)
    os.makedirs(seg_root, exist_ok=True)


    #print(os.path.join(image_root,data['imgnames'][0].replace('./arctic_data/data/images/', 'cropped_images/')))

    #print(data['data_dict']['s01/box_grab_01']['params'].keys())

    body_model_r = MANO(model_path='./hand_models/mano/',flat_hand_mean=False).cuda()
    body_model_l = MANO(model_path='/mnt/sda2/lxy/arctic/unpack/body_models/mano/',
                             is_rhand=False,flat_hand_mean=False).cuda()

    #smplx_r =  build_mano_aa(True, create_transl=True, flat_hand=False)
    #smplx_l = build_mano_aa(False, create_transl=True, flat_hand=False)
    object_tensors = ObjectTensors('cpu')

    ioi_offset = {}
    with open("/mnt/sda2/lxy/dataset/hand/arctic/meta/misc.json", "r") as f:
        misc = json.load(f)

    # unpack
    world2cams = {}
    intris_mat = {}
    image_sizes = {}
    subjects = list(misc.keys())
    for subject in subjects:
        #world2cam[subject] = misc[subject]["world2cam"]
        intris_mat[subject] = misc[subject]["intris_mat"]
        ioi_offset[subject] = misc[subject]["ioi_offset"]
        image_sizes[subject] = misc[subject]["image_size"]
        world2cams[subject] = misc[subject]["world2cam"]

    pbar = tqdm(enumerate(imgnames), total=len(imgnames), desc="Generating Mesh & Seg", mininterval=0)
    for idx, imgname in pbar:
        img_path = os.path.join(image_root,imgname.replace('./arctic_data/data/images/', 'cropped_images/'))
        hand_mesh_path_r = os.path.join(hand_mesh_data_root,
                                      imgname.replace('./arctic_data/data/images/', '').replace('.jpg', '_r.obj'))
        hand_mesh_path_l = os.path.join(hand_mesh_data_root,
                                      imgname.replace('./arctic_data/data/images/', '').replace('.jpg', '_l.obj'))
        obj_mesh_path = os.path.join(obj_mesh_data_root,
                                      imgname.replace('./arctic_data/data/images/', '').replace('jpg', 'obj'))
        seg_path = os.path.join(seg_root,
                                     imgname.replace('./arctic_data/data/images/', '').replace('jpg', 'png'))

        # if os.path.isfile(seg_path) and os.path.getsize(seg_path) > 0 \
        #         and os.path.isfile(obj_mesh_path) and os.path.getsize(obj_mesh_path) > 0:
        #     pbar.update(1)
        #     continue

        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = data_dict[f"{sid}/{seq_name}"]
        #print(seq_data.keys())
        data_params = seq_data["params"]
        #print(data_params.keys())
        # v3d_r = cam_data["verts.right"][:, view_idx]
        # v3d_l = cam_data["verts.left"][:, view_idx]
        vidx = int(image_idx.split(".")[0]) - ioi_offset[sid]


        pose_r = data_params["pose_r"][vidx].copy()
        #print(pose_r.shape)
        #print(pose_r)
        trans_r = data_params["trans_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        rot_r = data_params["rot_r"][vidx].copy()

        pose_l = data_params["pose_l"][vidx].copy()
        trans_l = data_params["trans_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()
        rot_l = data_params["rot_l"][vidx].copy()

        world2cam = np.array(world2cams[sid][view_idx-1])

        #root_r = body_model_r()['v'][0][0].detach().numpy()

        #rot_r, trans_r = apply_w2c_pose_with_center(rot_r, trans_r,root_r, world2cam)
        #rot_r, trans_r = apply_w2c_pose_numpy(rot_r, trans_r, world2cam)
        #rot_l, trans_l = apply_w2c_pose_numpy(rot_l, trans_l, world2cam)


        v3d_r = body_model_r(global_orient=torch.from_numpy(rot_r).float().reshape(-1, 3).to(device),
                                  hand_pose=torch.from_numpy(pose_r).float().reshape(-1, 45).to(device),
                                   betas=torch.from_numpy(betas_r).float().reshape(-1, 10).to(device),
                                   transl=torch.from_numpy(trans_r).float().reshape(-1, 3).to(device),
                                   )['v'][0].detach().cpu().numpy()
        # print(v3d_r[0])
        # print(trans_r)
        #
        # print(v3d_r[0])
        # print('smplx')
        # output = smplx_r()
        # print(list(output.keys()))
        # v3d_r = smplx_r(global_orient=torch.from_numpy(rot_r).float().reshape(-1, 3),
        #                      hand_pose=torch.from_numpy(pose_r).float().reshape(-1, 45),
        #                      betas=torch.from_numpy(betas_r).float().reshape(-1, 10),
        #                      transl=torch.from_numpy(trans_r).float().reshape(-1, 3),
        #                      )['vertices'][0].detach().numpy()
        # print(v3d_r[0])
        # print(trans_r)
        # v3d_r_0 = smplx_r()['vertices'][0].detach().numpy()
        # print(v3d_r_0[0])
        v3d_l = body_model_l(global_orient=torch.from_numpy(rot_l).float().reshape(-1, 3).to(device),
                             hand_pose=torch.from_numpy(pose_l).float().reshape(-1, 45).to(device),
                             betas=torch.from_numpy(betas_l).float().reshape(-1, 10).to(device),
                             transl=torch.from_numpy(trans_l).float().reshape(-1, 3).to(device),
                             )['v'][0].detach().cpu().numpy()

        v3d_l = (world2cam[:3, :3] @ v3d_l.T).T + world2cam[:3, 3]
        v3d_r = (world2cam[:3, :3] @ v3d_r.T).T + world2cam[:3, 3]

        f3d_r = body_model_r.faces
        f3d_l = body_model_l.faces

        f3d_o = Mesh(
            filename=f"/mnt/sda2/lxy/dataset/hand/arctic/meta/object_vtemplates/{obj_name}/mesh.obj"
        ).faces

        obj_rot = data_params["obj_rot"][vidx].copy()
        obj_trans = data_params["obj_trans"][vidx].copy()
        angles = data_params["obj_arti"][vidx].copy()

        obj_rot, obj_trans = apply_w2c_pose_numpy(obj_rot, obj_trans / 1000, world2cam)
        #print(angles)
        # if 'use' not in imgname:
        #     #continue
        #     #v3d_o = cam_data["verts.object"][:, view_idx]
        #     obj_rot, _ = cv2.Rodrigues(obj_rot)
        #     v3d_o = Mesh(
        #         filename=f"/mnt/sda2/lxy/dataset/hand/arctic/meta/object_vtemplates/{obj_name}/mesh.obj"
        #     ).vertices
        #
        #     v3d_o = (v3d_o/1000@obj_rot.T) +obj_trans
        # else:

        obj_meta = object_tensors(torch.from_numpy(np.array([angles])).unsqueeze(0).to('cpu'),
                                  torch.from_numpy(obj_rot).unsqueeze(0).to('cpu'),
                                  torch.from_numpy(obj_trans).unsqueeze(0).to('cpu'),
                                   [obj_name])
        v3d_o = obj_meta['v'][0].detach().numpy()


        obj_mesh = trimesh.Trimesh(vertices=v3d_o, faces=f3d_o)
        hand_mesh_r = trimesh.Trimesh(vertices=v3d_r, faces=f3d_r)
        hand_mesh_l = trimesh.Trimesh(vertices=v3d_l, faces=f3d_l)


        #obj_mesh.export(os.path.join(obj_mesh_path, f"{sid}_{seq_name}_{view_idx}_{image_idx}_obj.obj"))

        os.makedirs(os.path.dirname(obj_mesh_path), exist_ok=True)
        os.makedirs(os.path.dirname(hand_mesh_path_r), exist_ok=True)
        obj_mesh.export(obj_mesh_path)
        hand_mesh_r.export(hand_mesh_path_r)
        hand_mesh_l.export(hand_mesh_path_l)

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(intris_mat[sid][view_idx - 1])

        image_size = image_sizes[sid][view_idx]
        #print(image_size)

        # scale and center in the original image space

        data_bbox = seq_data["bbox"]
        bbox = data_bbox[vidx, view_idx]  # original bbox
        #print(bbox)

        dim=min([image_size[0], image_size[1]])

        k_scale = 1000 / dim  # resized_dim / bbox_size in full image space
        # intrx[0, 0] *= k_scale  # k*fx
        # intrx[1, 1] *= k_scale  # k*fy
        # # intrx[0, 2] = image_size[0]/2 - (bbox[0] - dim / 2.0)
        # # intrx[1, 2] = image_size[1]/2 - (bbox[1] - dim / 2.0)
        # intrx[0, 2] -= (bbox[0] - dim / 2.0)
        # intrx[1, 2] -=  (bbox[1] - dim / 2.0)
        # intrx[0, 2] *= k_scale
        # intrx[1, 2] *= k_scale

        cx, cy, scale = bbox
        s = 200.0 * scale
        crop_size = 1.5 * s  # 原图中裁剪的实际像素大小

        # Step 1: 计算 crop 左上角位置
        crop_x0 = cx - crop_size / 2.0
        crop_y0 = cy - crop_size / 2.0

        # Step 2: 缩放比例（将 crop_size 映射为 cap_dim）
        resize_scale = 1000 / crop_size

        # Step 3: 更新内参
        fx_new = intrx[0, 0] * resize_scale
        fy_new = intrx[1, 1] * resize_scale
        cx_new = (intrx[0, 2] - crop_x0) * resize_scale
        cy_new = (intrx[1, 2] - crop_y0) * resize_scale

        intrx_1 = np.array([
            [fx_new, 0.0, cx_new],
            [0.0, fy_new, cy_new],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        #exit()


        # generate_seg
        # 定义颜色映射（主手红/物体蓝/次手绿）
        COLOR_MAP = {
            1: [1.0, 0.0, 0.0],  # r
            2: [0.0, 1.0, 0.0],  # l
            3: [0.0, 0.0, 1.0]  # o
        }

        # 输入数据格式示例：
        meshes_data = [
            (torch.from_numpy(v3d_r.astype(np.float32)).to(device), torch.from_numpy(f3d_r.astype(np.float32)).to(device), 1),  # r手
            (torch.from_numpy(v3d_l.astype(np.float32)).to(device), torch.from_numpy(f3d_l.astype(np.float32)).to(device), 2),  # l手
            (torch.from_numpy(v3d_o.astype(np.float32)).to(device), torch.from_numpy(f3d_o.astype(np.float32)).to(device), 3),  # 物体

        ]
        #print(intrx_1)

        fx, fy, px, py = intrx_1[0, 0], intrx_1[1, 1], intrx_1[0, 2], intrx_1[1, 2]

        R = torch.eye(3).unsqueeze(0)
        R[:, 0, 0] = -1
        R[:, 1, 1] = -1

        T = torch.zeros(3).unsqueeze(0)

        cameras = PerspectiveCameras(
            focal_length=((fx, fy),),  # (fx, fy)
            principal_point=((px, py),),  # (px, py)
            image_size=((1000, 1000),),  # (imwidth, imheight)
            device="cuda",
            R=R.cuda(), T=T.cuda(),
            in_ndc=False
        )
        raster_settings = RasterizationSettings(
            image_size=(1000, 1000),
            blur_radius=0,
            faces_per_pixel=1,
            bin_size=0
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        # img_h = mesh_uvd2img(mesh_h, face_h, rasterizer)
        # img_o = mesh_uvd2img(mesh_o, face_o, rasterizer)

        # 生成最终分割图
        final_seg = generate_segmentation(meshes_data, COLOR_MAP, rasterizer)

        # 转换为 uint8 格式并保存图像
        seg_img = (final_seg * 255).astype(np.uint8)  # 如果像素值是 [0, 1] 范围，需要乘以 255
        image = Image.fromarray(seg_img)

        os.makedirs(os.path.dirname(seg_path), exist_ok=True)
        image.save(seg_path)
        if idx%10==0:
            pbar.write(seg_path)

        # debug masked img
        # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # mask = (seg_img == [0, 0, 0]).all(axis=-1)  # 得到 H×W 的 bool mask
        # img[mask] = [0, 0, 0]
        # image = Image.fromarray(img)
        # image.save(seg_path)
        #
        # pbar.write(seg_path)
        # exit()
        pbar.update(1)

def mesh_uvd2img(hand_verts, hand_faces,rasterizer):
    #hand_verts = torch.from_numpy(hand_verts).float()
    #hand_faces = torch.from_numpy(hand_faces).float()
    meshes = Meshes(verts=hand_verts, faces=hand_faces)

    fragments = rasterizer(meshes)
    ori_depth = fragments.zbuf
    ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device) * 0, ori_depth)
    resize_depth = ori_depth.permute(0, 3, 1, 2)
    return resize_depth


def generate_segmentation(meshes_list, color_map, rasterizer):
    """
    合并多mesh并生成带深度遮挡的分割图
    :param meshes_list: [(verts, faces, color_id)...]
    :param color_map: {id: [R,G,B]} 如 {1:[1,0,0], 2:[0,0,1], 3:[0,1,0]}
    """
    # 合并所有网格并标记面标签[2,5](@ref)
    combined_verts, combined_faces, face_labels = [], [], []
    vert_offset = 0
    for verts, faces, cid in meshes_list:
        combined_verts.append(verts)
        combined_faces.append(faces + vert_offset)
        face_labels.append(torch.full((len(faces),), cid, device=verts.device))
        vert_offset += len(verts)

    # 构建统一mesh[2](@ref)

    meshes = Meshes(
        verts=torch.cat(combined_verts).unsqueeze(0),
        faces=torch.cat(combined_faces).unsqueeze(0)
    )

    # 渲染获取面索引和深度[5](@ref)
    #fragments = rasterizer(meshes.extend(len(meshes_list)))
    fragments = rasterizer(meshes)
    face_idx = fragments.pix_to_face[0, ..., 0]  # (H,W)
    zbuf = fragments.zbuf[0, ..., 0]  # (H,W)

    # 创建颜色映射表[1,4](@ref)
    color_tensor = torch.zeros((len(color_map) + 1, 3), device=zbuf.device)
    for cid, color in color_map.items():
        color_tensor[cid] = torch.tensor(color, device=zbuf.device)

    # 生成分割图像（根据深度优先选择可见颜色）[5](@ref)
    face_labels = torch.cat(face_labels)
    # print(face_labels.shape) # torch.Size([4552])
    # print(face_idx.shape) # torch.Size([848, 480])
    seg_img = color_tensor[face_labels[face_idx]]
    # print(seg_img.shape) # ([848, 480, 3])
    # print(zbuf.shape) # ([848, 480])
    seg_img[zbuf <= 0] = 0  # 无效深度设为黑色背景
    return seg_img.cpu().numpy()  # HWC格式



def run_subject(subject, imgnames, data_dict):
    print(f"Start processing {subject}")
    preprocess('/mnt/sda2/lxy/dataset/hand/', subject, imgnames, data_dict)
    print(f"Finished processing {subject}")

def main():
    subjects = [f"s{i:02d}" for i in range(1,11)]  # s01~s10
    processes = []

    data = np.load(os.path.join("/mnt/sda2/lxy/arctic/unpack/arctic_data/data/", "splits", "p1_train.npy"),
                   allow_pickle=True).item()

    # data = np.load(os.path.join(data_root, "arctic_seqs", "splits","R9.npy"),allow_pickle=True).item()
    # print(data['data_dict'].keys())

    for subject in subjects:
        imgnames = [n for n in data['imgnames'] if subject in n]
        data_dict = data['data_dict']
        print(len(data['imgnames']))
        p = multiprocessing.Process(target=run_subject, args=(subject, imgnames, data_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()