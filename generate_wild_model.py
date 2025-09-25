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


parser.add_argument('--wild_dir', type=str, default="/home/cyc/pycharm/lxy/HOISDF/ckpts/wild/wood_0")


def generate_dexycb(pose_dir):


    #anno_path = os.path.join(args.annotation_dir, "dex_ycb_s0_train_data_cut.json") if 'train' in pose_dir else os.path.join(args.annotation_dir, "dex_ycb_s0_test_data_cut.json")
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


    with open(pose_dir, 'r') as pose:
        init_pose = json.load(pose)

        for id, pose_data in init_pose.items():

            #print(os.path.join(source_dir,anno_json[id]['label_file']))

            img_file = pose_data['img_file']
            model_file = img_file.replace('rgb', 'model_hamer').replace('png', 'npz')

            #model_outdir = model_file.replace('.npz', '_HOISDF.npz')


            # mano_pose_6d = np.array(pose_data['mano_pose6d'])
            # betas = np.array(pose_data['mano_shape']).reshape(-1,10).astype(np.float32)
            mano_pose_6d = np.array(pose_data['mano_param_hamer'][:48]).reshape(48).astype(np.float32)
            betas = np.array(pose_data['mano_param_hamer'][48:58]).reshape(-1, 10).astype(np.float32)
            #trans = np.array(pose_data['mano_param_hamer'][58:]).reshape(-1, 3).astype(np.float32)


            joint_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'joints_kpfusion',
                                      os.path.basename(img_file)[:-4] + '.npz')

            handJoints3D = np.load(joint_path)['joint_xyz'].astype(np.float32).copy()

            handJoints3D *= np.array([1, -1, -1])
            trans = handJoints3D[0].reshape(-1, 3).astype(np.float32)


            pose = mano_pose_6d[3:48].reshape(1, -1).astype(np.float32)
            rot = mano_pose_6d[:3].reshape(1, -1).astype(np.float32)

            rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
            new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1, 3]).astype(np.float32)
            pose_torch = torch.from_numpy(pose)  # .cuda()
            betas_torch = torch.from_numpy(betas)  # .cuda()

            new_root_orient_torch = torch.from_numpy(new_root_orient)  # .cuda()
            new_trans_torch = torch.from_numpy(trans)
            body = body_model(betas=betas_torch)
            minimal_shape = body['v'].detach().cpu().numpy()[0]

            obj_3DCorners = np.array(obj_corners[int(pose_data['obj_label'])]).astype(np.float32)

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

            #print(joints_3d_gt)
            #print(joints_gt[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]])


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


            out_filename = model_file.replace('model_hamer', 'model_HOISDF')
            os.makedirs(os.path.dirname(out_filename),exist_ok=True)
            #
            # out_filename = os.path.join(model_outdir, frame_id)
            # cv2.imwrite(out_filename+'.png', cv2.cvtColor(np.uint8(rgb), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(out_filename + '_depth.png', depth)

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
                     obj_label=int(pose_data['obj_label']),
                     )

            # image_file = os.path.join(source_dir,anno_json[id]['color_file'])
            # out_path = os.path.join(output_dir, subject_name, str(obj_label) + '-' + seq_name, cam_name)
            # out_filename = os.path.join(out_path,anno_json[id]['color_file'].split('/')[-1])
            # shutil.copy(image_file, out_filename)

            #print('MPJPE:', errors.mean())
            #exit()
        # print('MPJPE:', mpjpe / out_num)
        # print('ADDS:', e_ADDS / out_num)
        # print('MCE:', e_MCE / out_num)
        # print('OCE:', e_OCE / out_num)


if __name__ == '__main__':
    args = parser.parse_args()
    wild_dir = args.wild_dir
    # for cam_name in os.listdir(wild_dir):
    pose_dir = os.path.join(wild_dir,'pred_ho_pose.json')
    #output_dir = os.path.join(args.output_dir)
    generate_dexycb(pose_dir)
    # generate_dexycb(os.path.join(
    #                 cfg.annotation_dir, "dex_ycb_s0_train_data_cut.json"
    #             ))
