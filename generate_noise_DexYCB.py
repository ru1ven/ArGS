import os
import torch
import trimesh
import json
import glob
import shutil
import argparse
import re
import pickle
import numpy as np
import yaml

from scipy.spatial.transform import Rotation

from right_hand_model.body_models import MANO

parser = argparse.ArgumentParser(
    description='Preprocessing for HO3D.'
)
parser.add_argument('--source_dir', type=str, default="/home/cyc/pycharm/data/hand/DexYCB/")
parser.add_argument('--output_dir', type=str, default="/mnt/sda1/lxy/DexYCB/")


if __name__ == '__main__':
    args = parser.parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/')
    body_model_pca = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/',use_pca=True,num_pca_comps=48,flat_hand_mean=False)#.cuda()

    faces = np.load("/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/misc/faces.npz")['faces']

    _SERIALS = [
        '836212060125',
        '839512060362',
        '840412060917',
        '841412060263',
        '932122060857',
        '932122060861',
        '932122061900',
        '932122062010',
    ]
    all_cam_params = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # camera para
    for cam in _SERIALS:
        intr_file = os.path.join(os.path.join(source_dir, "calibration"), "intrinsics",
                                 "{}_{}x{}.yml".format(cam, 640, 480))
        with open(intr_file, 'r') as f:
            intr = yaml.load(f, Loader=yaml.FullLoader)
        intr = intr['color']
        D = np.array([0, 0, 0, 0, 0])
        R = np.eye(3)
        R[0, 0] = 1
        R[1, 1] = 1
        R[2, 2] = 1
        T = np.zeros((3, 1))
        K =[[intr['fx'], 0, intr['ppx']], [0, intr['fy'], intr['ppy']], [0, 0, 1]]
        cam_params = {'K': K, 'D': D.tolist(), 'R': R.tolist(), 'T': T.tolist()}
        all_cam_params.update({cam: cam_params})
    # with open(os.path.join(output_dir, 'cam_params.json'), 'w') as f:
    #     json.dump(all_cam_params, f)

    # obj aabb
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
    obj_file = {
        k: os.path.join( os.path.join(source_dir, "models"), v, "textured_simple.obj")
        for k, v in _YCB_CLASSES.items()
    }
    obj_raw_meshes = {}
    obj_corners = {}
    for obj_idx, obj_file in obj_file.items():
        obj_mesh = trimesh.load(obj_file, process=False)
        obj_raw_meshes[obj_idx] = obj_mesh

        # 计算最小值和最大值
        min_coords = np.min(obj_mesh.vertices, axis=0)
        max_coords = np.max(obj_mesh.vertices, axis=0)

        # 生成包围盒的8个角点
        x_min, y_min, z_min = min_coords
        x_max, y_max, z_max = max_coords

        corners = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max]
        ])
        obj_corners[obj_idx] = corners


    #all_cam_params = {'all_cam_params': cam_names}
    for subject_name in sorted(glob.glob(os.path.join(source_dir, '*subject*'))):
        subject_name = os.path.basename(subject_name)
        seq_dir = os.path.join(source_dir, subject_name)
        for seq_name in os.listdir(seq_dir):
            meta_path = os.path.join(seq_dir,seq_name, 'meta.yml')
            with open(meta_path, 'r') as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            ycb_ids = meta['ycb_ids']
            ycb_grasp_ind = meta['ycb_grasp_ind']
            obj_label = ycb_ids[ycb_grasp_ind]
            mano_side= meta['mano_sides'][0]
            if mano_side != 'right':
                continue
            cam_dir = os.path.join(seq_dir, seq_name)
            out_num = 0
            mpjpe = 0
            for cam_name in os.listdir(cam_dir):
                if not os.path.isdir(os.path.join(cam_dir,cam_name)):
                    continue
                params = {}
                camera_intrinsics = None

                out_path = os.path.join(output_dir,subject_name, str(obj_label)+'-'+seq_name,cam_name)
                #print(out_path)
                data_dir = os.path.join(source_dir,subject_name, seq_name,cam_name)

                model_outdir = os.path.join(out_path, "model_noise_root_gt")

                if not os.path.exists(model_outdir):
                    os.makedirs(model_outdir)
                    #print(model_outdir)

                # img_outdir = os.path.join(out_path, "images")
                # if not os.path.exists(img_outdir):
                #     os.makedirs(img_outdir)
                #process annotations
                anno_files = sorted(glob.glob(os.path.join(data_dir, 'labels_*.npz')))
                #print(os.path.join(data_path, 'meta', '*.pkl'))
                for anno_file in anno_files:
                    anno = np.load(anno_file)
                    #print(anno.files)  # ['seg', 'pose_y', 'pose_m', 'joint_3d', 'joint_2d']

                    mask = anno['seg']

                    obj_mask = (mask == obj_label)
                    hand_mask = (mask == 255)

                    if len(np.where(hand_mask)[0]) == 0 or len(np.where(obj_mask)[0])== 0 :
                        continue

                    mano_para = np.array(anno['pose_m']).reshape(-1)
                    mano_calib_file = os.path.join(source_dir, "calibration",
                                                   "mano_{}".format(meta['mano_calib'][0]),
                                                   "mano.yml")
                    with open(mano_calib_file, 'r') as f:
                        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
                    betas = np.array(mano_calib['betas']).astype(np.float32).reshape(1, -1)

                    trans = mano_para[48:].reshape(1, -1)
                    pose = mano_para[3:48].reshape(1, -1)
                    rot = mano_para[:3].reshape(1, -1)
                    rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
                    new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1,3]).astype(np.float32)
                    pose_torch_gt = torch.from_numpy(pose)  # .cuda()
                    betas_torch_gt = torch.from_numpy(betas)  # .cuda()
                    new_trans = trans.reshape([1, 3]).astype(np.float32)
                    new_root_orient_torch_gt = torch.from_numpy(new_root_orient)#.cuda()
                    new_trans_torch_gt = torch.from_numpy(new_trans)#.cuda()
                    body = body_model(betas=betas_torch_gt)
                    minimal_shape_gt = body['v'].detach().cpu().numpy()[0]
                    body = body_model_pca(global_orient=new_root_orient_torch_gt, hand_pose=pose_torch_gt, betas=betas_torch_gt, transl=new_trans_torch_gt)
                    out_filename_gt = os.path.join(model_outdir, os.path.basename(anno_file)[:-4])
                    bone_transforms_gt = body['bone_transforms'].detach().cpu().numpy()
                    Jtr_posed_gt = body['Jtr'].detach().cpu().numpy()
                    # obj
                    transf = anno['pose_y'][ycb_grasp_ind]
                    obj_rot_gt, obj_trans_gt = transf[:3, :3], transf[:, 3:]
                    obj_3DCorners = np.array(obj_corners[obj_label]).astype(np.float32)

                    out_num += 1

                    # generate noise pose
                    #rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
                    new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1, 3]).astype(np.float32)
                    pose_torch = torch.from_numpy(pose)  # .cuda()
                    betas_torch = torch.from_numpy(betas)  # .cuda()
                    new_trans = trans.reshape([1, 3]).astype(np.float32)
                    new_root_orient_torch = torch.from_numpy(new_root_orient)  # .cuda()
                    new_trans_torch = torch.from_numpy(new_trans)  # .cuda()

                    # 给参数添加噪声
                    noise_std = {
                        'global_orient': 0.04,
                        'hand_pose': 0.04,
                        'betas': 0.004,
                        'transl': 0.004
                    }

                    new_root_orient_torch = new_root_orient_torch + torch.randn_like(new_root_orient_torch) * noise_std['global_orient']
                    pose_torch = pose_torch + torch.randn_like(pose_torch) * noise_std['hand_pose']
                    betas_torch = betas_torch + torch.randn_like(betas_torch) * noise_std['betas']
                    if 'root_gt' not in model_outdir:
                        print('Apply root_noise')
                        new_trans_torch = new_trans_torch + torch.randn_like(new_trans_torch) * noise_std['transl']


                    body = body_model(betas=betas_torch)
                    minimal_shape = body['v'].detach().cpu().numpy()[0]
                    body = body_model_pca(global_orient=new_root_orient_torch, hand_pose=pose_torch, betas=betas_torch,
                                          transl=new_trans_torch)
                    out_filename = os.path.join(model_outdir, os.path.basename(anno_file)[:-4])
                    bone_transforms = body['bone_transforms'].detach().cpu().numpy()
                    Jtr_posed = body['Jtr'].detach().cpu().numpy()
                    # print('gt',Jtr_posed_gt)
                    # print(Jtr_posed)
                    # print()
                    errors =np.linalg.norm(Jtr_posed_gt - Jtr_posed, axis=-1)
                    mpjpe += errors.mean()

                    obj_rot_gt = torch.from_numpy(obj_rot_gt).float()
                    obj_trans_gt = torch.from_numpy(obj_trans_gt).float()
                    obj_rot = obj_rot_gt + torch.randn_like(obj_rot_gt) * noise_std['global_orient']*1.5
                    obj_trans = obj_trans_gt + torch.randn_like(obj_trans_gt) * noise_std['transl']*1.5
                    obj_rot = obj_rot.numpy()
                    obj_trans = obj_trans.numpy()

                    np.savez(out_filename,
                             minimal_shape=minimal_shape,
                             betas=betas_torch.numpy(),
                             Jtr_posed=Jtr_posed[0],
                             bone_transforms=bone_transforms[0],
                             trans=new_trans_torch[0].numpy(),
                             root_orient=new_root_orient_torch[0].numpy(),
                             pose=pose_torch[0].numpy(),
                             obj_trans=obj_trans,
                             obj_rot=obj_rot,
                             obj_3DCorners=obj_3DCorners,
                             obj_label=obj_label,
                             seg=anno['seg'],
                             trans_gt=new_trans_torch_gt[0].numpy(),
                             root_orient_gt=new_root_orient_torch_gt[0].numpy(),
                             pose_gt=pose_torch_gt[0].numpy(),
                             betas_gt=betas_torch_gt.numpy(),
                             obj_rot_gt=obj_rot_gt.numpy(),
                             obj_trans_gt=obj_trans_gt.numpy()
                             )


                #process images
                model_files = sorted(glob.glob(os.path.join(model_outdir, '*.npz')))
                image_basenames = sorted([os.path.basename(model_file).replace('npz','jpg').replace('labels','color') for model_file in model_files])
                # for image_basename in image_basenames:
                #     image_file = os.path.join(data_dir,image_basename)
                #     out_filename = os.path.join(img_outdir, os.path.basename(image_file))
                #     shutil.copy(image_file, out_filename)

            print('MPJPE:',mpjpe/out_num)
            #exit()