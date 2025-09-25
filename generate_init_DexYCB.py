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
from manopth_utils.manopth.manolayer import ManoLayer
from utils.general_utils import compute_obj_metrics_ycb, prepare_model_template

parser = argparse.ArgumentParser(
    description='Preprocessing for DexYCB_cut.'
)
parser.add_argument('--source_dir', type=str, default="/home/cyc/pycharm/data/hand/DexYCB/")
parser.add_argument('--output_dir', type=str, default="/mnt/sda1/lxy/3DGS/DexYCB/test/")
parser.add_argument('--pose_dir', type=str, default="/home/cyc/pycharm/lxy/HOISDF/ckpts/dexycb/pred_ho_pose_test_sample_rate_5.json")
parser.add_argument('--annotation_dir', type=str, default="/home/cyc/pycharm/lxy/HOISDF/dataset/DexYCB/annotation/")

_SUBJECTS = [
      '20200709-subject-01',
      '20200813-subject-02',
      '20200820-subject-03',
      '20200903-subject-04',
      '20200908-subject-05',
      '20200918-subject-06',
      '20200928-subject-07',
      '20201002-subject-08',
      '20201015-subject-09',
      '20201022-subject-10',
  ]


def generate_dexycb():
    args = parser.parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    pose_dir = args.pose_dir
    anno_path = os.path.join(args.annotation_dir, "dex_ycb_s0_train_data_cut.json") if 'train' in pose_dir \
        else os.path.join(args.annotation_dir, "dex_ycb_s0_test_data_sample_rate_5.json")
    #body_model_flat = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/', flat_hand_mean=True)
    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/', flat_hand_mean=True)
    body_model_pca = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/', use_pca=True,
                          num_pca_comps=48, flat_hand_mean=False)  # .cuda()

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
        intr_file = os.path.join(os.path.join('/home/cyc/pycharm/data/hand/DexYCB/', "calibration"), "intrinsics",
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
        K = [[intr['fx'], 0, intr['ppx']], [0, intr['fy'], intr['ppy']], [0, 0, 1]]
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
        k: os.path.join(os.path.join('/home/cyc/pycharm/data/hand/DexYCB/', "models"), v, "textured_simple.obj")
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

    out_num = 0
    mpjpe = 0
    e_ADDS =0
    e_MCE =0
    e_OCE =0
    filter_num = 0

    mano_layer = ManoLayer(
        ncomps=45,
        center_idx=0,
        flat_hand_mean=True,
        side="right",
        mano_root="/home/cyc/pycharm/lxy/HOISDF/tool/mano_models/",
        use_pca=False,
    )

    with open(anno_path, 'r') as anno_json:
        with open(pose_dir, 'r') as pose:
            init_pose = json.load(pose)
            anno_json = json.load(anno_json)
            for id, pose_data in init_pose.items():

                #print(os.path.join(source_dir,anno_json[id]['label_file']))
                subject_name = anno_json[id]['label_file'].split('/')[-4]
                seq_dir = os.path.join(source_dir, subject_name)
                seq_name = anno_json[id]['label_file'].split('/')[-3]
                cam_name = anno_json[id]['label_file'].split('/')[-2]
                anno_file = os.path.join(source_dir,anno_json[id]['label_file'])
                anno = np.load(anno_file)

                meta_path = os.path.join(seq_dir, seq_name, 'meta.yml')
                with open(meta_path, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                ycb_ids = meta['ycb_ids']
                ycb_grasp_ind = meta['ycb_grasp_ind']
                obj_label = ycb_ids[ycb_grasp_ind]
                mano_side = meta['mano_sides'][0]
                if mano_side != 'right':
                    print('left!')
                    continue

                mask = anno['seg']
                params = {}
                camera_intrinsics = None

                out_path = os.path.join(output_dir, subject_name, seq_name, cam_name)
                # print(out_path)
                data_dir = os.path.join(source_dir, subject_name, seq_name, cam_name)

                model_outdir = os.path.join(out_path, "model_SDF")
                if not os.path.exists(model_outdir):
                    os.makedirs(model_outdir)

                obj_mask = (mask == obj_label)
                hand_mask = (mask == 255)

                # if len(np.where(hand_mask)[0]) == 0 and len(np.where(obj_mask)[0]) == 0:
                #     print(filter_num)
                #     filter_num += 1
                #     continue

                mano_pose_pca_mean = np.array(anno['pose_m']).reshape(-1)


                mano_para = np.concatenate(
                    (
                        mano_pose_pca_mean[0:3],
                        np.matmul(
                            mano_pose_pca_mean[3:48], np.array(mano_layer.smpl_data["hands_components"]).copy()
                        )
                        +mano_layer.smpl_data["hands_mean"].copy(),
                        mano_pose_pca_mean[48:],
                    ),
                    axis=0,
                ).astype(np.float32)


                mano_calib_file = os.path.join('/home/cyc/pycharm/data/hand/DexYCB/', "calibration",
                                               "mano_{}".format(meta['mano_calib'][0]),
                                               "mano.yml")
                with open(mano_calib_file, 'r') as f:
                    mano_calib = yaml.load(f, Loader=yaml.FullLoader)
                betas = np.array(mano_calib['betas']).astype(np.float32).reshape(1, -1)

                trans = mano_para[48:].reshape(1, -1)
                pose_gt = mano_para[3:48].reshape(1, -1)
                rot_gt = mano_para[:3].reshape(1, -1)
                rot_gt = Rotation.from_rotvec(np.array(rot_gt).reshape([-1])).as_matrix()
                new_root_orient_gt = Rotation.from_matrix(rot_gt).as_rotvec().reshape([1, 3]).astype(np.float32)
                pose_torch_gt = torch.from_numpy(pose_gt)  # .cuda()
                betas_torch_gt = torch.from_numpy(betas)  # .cuda()
                new_trans = trans.reshape([1, 3]).astype(np.float32)
                new_root_orient_torch_gt = torch.from_numpy(new_root_orient_gt)  # .cuda()
                new_trans_torch_gt = torch.from_numpy(new_trans)  # .cuda()
                body = body_model(betas=betas_torch_gt)
                minimal_shape_gt = body['v'].detach().cpu().numpy()[0]
                body = body_model(global_orient=new_root_orient_torch_gt, hand_pose=pose_torch_gt,
                                      betas=betas_torch_gt, transl=new_trans_torch_gt)

                # path = '/home/cyc/pycharm/lxy/3DGS/debug/gt.obj'
                # with open(path, 'w') as fp:
                #     for v in body['v'].detach().cpu().numpy()[0]:
                #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                #     for f in faces + 1:
                #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

                bone_transforms_gt = body['bone_transforms'].detach().cpu().numpy()
                Jtr_posed_gt = body['Jtr'].detach().cpu().numpy()
                joints_gt = body['joints'].detach().cpu().numpy()
                joints_gt = joints_gt[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
                # obj
                transf = anno['pose_y'][ycb_grasp_ind].astype(np.float32)
                obj_rot_gt, obj_trans_gt = transf[:3, :3], transf[:, 3:]
                obj_3DCorners = np.array(obj_corners[obj_label]).astype(np.float32)

                out_num += 1

                #mano_pose_6d = np.array(pose_data['mano_params_gt_out'])[:48]

                mano_pose_6d = np.array(pose_data['mano_pose6d'])
                betas = np.array(pose_data['mano_shape']).reshape(-1,10).astype(np.float32)


                pose = mano_pose_6d[3:48].reshape(1, -1).astype(np.float32)
                rot = mano_pose_6d[:3].reshape(1, -1).astype(np.float32)
                rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
                new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1, 3]).astype(np.float32)
                pose_torch = torch.from_numpy(pose)  # .cuda()
                betas_torch = torch.from_numpy(betas)  # .cuda()

                new_root_orient_torch = torch.from_numpy(new_root_orient)  # .cuda()
                new_trans_torch = new_trans_torch_gt
                body = body_model(betas=betas_torch)
                minimal_shape = body['v'].detach().cpu().numpy()[0]

                noise = False
                if noise:
                    new_root_orient = new_root_orient_gt+np.random.normal(loc=0, scale=np.deg2rad(5), size=new_root_orient_gt.shape)
                    pose = pose_gt+np.random.normal(loc=0, scale=np.deg2rad(5), size=pose_gt.shape)
                    pose_torch = torch.from_numpy(pose).float()
                    new_root_orient_torch = torch.from_numpy(new_root_orient).float()

                body = body_model(global_orient=new_root_orient_torch, hand_pose=pose_torch,
                                      betas=betas_torch, transl=new_trans_torch)

                # path = '/home/cyc/pycharm/lxy/3DGS/debug/init.obj'
                # shutil.copy(os.path.join(source_dir,anno_json[id]['color_file']), '/home/cyc/pycharm/lxy/3DGS/debug/init_flatF.png')
                # with open(path, 'w') as fp:
                #     for v in body['v'].detach().cpu().numpy()[0]:
                #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                #     for f in faces + 1:
                #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

                bone_transforms = body['bone_transforms'].detach().cpu().numpy()
                Jtr_posed = body['Jtr'].detach().cpu().numpy()
                joints = body['joints'].detach().cpu().numpy()

                # pred_verts, pred_joints = mano_layer(th_pose_coeffs=torch.cat([new_root_orient_torch,pose_torch],dim=-1), th_betas=betas_torch, th_trans=new_trans_torch)
                # joints = pred_joints/1000
                # print(joints)

                joints = joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
                joints_3d_gt = np.array(anno_json[id]["joint_3d"], dtype=np.float32)
                #print(joints_3d_gt)
                #print(joints_gt[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]])

                errors = np.linalg.norm(joints - joints_gt, axis=-1)
                print(errors.mean())

                #print('1',np.sqrt(np.sum((Jtr_posed - Jtr_posed_gt) ** 2, axis=-1)).mean())

                mpjpe += errors.mean()
                # obj

                obj_rot, obj_trans = Rotation.from_rotvec(np.array(pose_data['obj_rot'], dtype=np.float32).reshape([-1])).as_matrix(), \
                                     (np.array(pose_data['obj_trans'], dtype=np.float32)+np.array(pose_data['obj_center_cam'], dtype=np.float32)).reshape(3,1)

                # print('init_h', mano_pose_6d[:3])
                #
                # print('anno_json:',np.array(anno_json[id]['pose_m']).reshape(-1)[:3])
                # print('gt_h', mano_para[:3])

                # print(np.array(pose_data['obj_center_cam']))
                # print('init', obj_rot, obj_trans)
                # print('gt', obj_rot_gt, obj_trans_gt)
                if noise:
                    obj_rot = obj_rot_gt+np.random.normal(loc=0, scale=np.deg2rad(10), size=obj_rot_gt.shape)
                    obj_trans = obj_trans_gt + np.random.normal(loc=0, scale=0.01, size=obj_trans_gt.shape)

                ADDS, MCE = compute_obj_metrics_ycb(torch.from_numpy(obj_rot_gt),
                                                       torch.from_numpy(obj_trans_gt),
                                                       torch.from_numpy(obj_rot),
                                                       torch.from_numpy(obj_trans),
                                                       obj_label)


                OCE = np.linalg.norm(obj_trans - obj_trans_gt, axis=0)

                e_ADDS += ADDS.item()
                e_MCE += MCE.item()
                e_OCE += OCE.mean()

                print('ADDS:', ADDS.item())
                print('MCE:',MCE.item())
                print('OCE:', OCE.mean())

                # if out_num == 200:
                #     print('MPJPE:', mpjpe / out_num)
                #     print('ADDS:', e_ADDS / out_num)
                #     print('MCE:', e_MCE / out_num)
                #     print('OCE:', e_OCE / out_num)
                #     exit()


                out_filename = os.path.join(model_outdir, anno_json[id]['color_file'].split('/')[-1][:-4].replace("color","labels"))

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
                         obj_rot_gt=obj_rot_gt,
                         obj_trans_gt=obj_trans_gt,
                         joints_3d_gt=joints_3d_gt
                         )

                image_file = os.path.join(source_dir,anno_json[id]['color_file'])
                os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
                out_filename = os.path.join(out_path,'images',anno_json[id]['color_file'].split('/')[-1])
                #print(out_filename)
                shutil.copy(image_file, out_filename)

                #print('MPJPE:', errors.mean())
                #exit()
            print('all_MPJPE:', mpjpe / out_num)
            print('all_ADDS:', e_ADDS / out_num)
            print('all_MCE:', e_MCE / out_num)
            print('all_OCE:', e_OCE / out_num)
            print('filter_num:', filter_num)


def copy_json():

    args = parser.parse_args()
    anno_path = os.path.join(args.annotation_dir, "dexycb_test_sdf_sample_rate_5.json")
    out_json_path = os.path.join(args.annotation_dir, "dex_ycb_s0_test_data_sample_rate_5.json")

    #24549
    # with open(os.path.join(args.annotation_dir, "dex_ycb_s0_test_data_cut.json"), 'r') as full_anno_json:
    #     full_anno_json = json.load(full_anno_json)
    #     print(len(full_anno_json))

    # miss_num=0
    # out_json = {}
    # with open(anno_path, 'r') as anno_json:
    #     with open(os.path.join(args.annotation_dir, "dex_ycb_s0_test_data.json"), 'r') as full_anno_json:
    #         anno_json = json.load(anno_json)
    #         full_anno_json = json.load(full_anno_json)
    #
    #
    #         print(anno_json['images'][0].keys())
    #         print(anno_json['annotations'][0].keys())
    #         for anno in anno_json['images']:
    #             id = 'id_'+str(anno['id'])
    #         #     #print(anno['file_name'])
    #         #     name_list = anno['file_name'].split('_')
    #         #     subject, action, camera, filename = name_list[-5], name_list[-4]+'_'+name_list[-3], name_list[-2], name_list[-1]
    #         #     model_hoisdf = os.path.join(args.output_dir, _SUBJECTS[int(subject)-1], action, camera, 'model_HOISDF', 'color_{}.npz'.format(format(filename, '>06')))
    #         #     model_sdf = model_hoisdf.replace('model_HOISDF', 'model_SDF')
    #         #     #os.makedirs(os.path.dirname(model_sdf), exist_ok=True)
    #         #     if not os.path.isfile(model_hoisdf):
    #         #         miss_num+=1
    #         #         print(miss_num)
    #         #         print(model_sdf)
    #         #     #shutil.copy(model_hoisdf, model_sdf)
    #
    #             print(full_anno_json[id].keys())
    #             out_json[id] = full_anno_json[id]
    # print(len(out_json.keys()))
    # with open(os.path.join(out_json_path), 'w') as f:
    #     json.dump(out_json, f)






if __name__ == '__main__':

    #generate_dexycb()

    copy_json()

