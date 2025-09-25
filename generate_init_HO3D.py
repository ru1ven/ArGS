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
    description='Preprocessing for HO3D_cut.'
)
parser.add_argument('--source_dir', type=str, default="/mnt/sda1/lxy/3DGS/HO3D_v3/train/")
parser.add_argument('--pose_dir', type=str, default="/home/cyc/pycharm/lxy/HOISDF/ckpts/ho3d/pred_ho_pose.json")
#parser.add_argument('--annotation_dir', type=str, default="/home/cyc/pycharm/lxy/HOISDF/dataset/DexYCB/annotation/")


_YCB_NAME2ID = {
   '002_master_chef_can':1,
   '003_cracker_box': 2,
   '004_sugar_box': 3,
   '005_tomato_soup_can': 4,
   '006_mustard_bottle' : 5,
   '007_tuna_fish_can' : 6,
   '008_pudding_box' : 7,
   '009_gelatin_box' : 8,
   '010_potted_meat_can' : 9,
   '011_banana' : 10,
   '019_pitcher_base' : 11,
   '021_bleach_cleanser' : 12,
   '024_bowl' : 13,
   '025_mug' : 14,
   '035_power_drill' : 15,
   '036_wood_block' : 16,
   '037_scissors' : 17,
   '040_large_marker' : 18,
   '051_large_clamp' : 19,
   '052_extra_large_clamp' : 20,
   '061_foam_brick' : 21,
}

def generate_dexycb():
    args = parser.parse_args()
    source_dir = args.source_dir
    pose_dir = args.pose_dir

    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/', flat_hand_mean=True)
    body_model_pca = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/', use_pca=True,
                          num_pca_comps=48, flat_hand_mean=False)  # .cuda()

    faces = np.load("/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/misc/faces.npz")['faces']

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

    seqname_to_subject_id = {
    0: ['ABF11', 'ABF13', 'ABF14', 'ABF10', 'ABF12'],
    1: ['BB10', 'BB13', 'BB12', 'BB11', 'BB14'],
    2: ['GSF14', 'GSF12', 'GSF10', 'GSF13', 'GSF11', 'GPMF11', 'GPMF12', 'GPMF10', 'GPMF14', 'GPMF13'],

    3: ['MC6', 'MC2', 'MC4','MC1', 'MC3', 'MC5'],
    4: ['MDF10', 'MDF13', 'MDF11', 'MDF14', 'MDF12'],

    5: ['ND2'],
    6: ['SB10', 'SB12', 'SB14','SM2', 'SM3','SM4','SM5','SS2', 'SS1','SMu42', 'SMu41', 'SMu40','SMu1','SMu42', 'SMu41', 'SMu40','SMu1'],
    7: ['ShSu12', 'ShSu13', 'ShSu14', 'ShSu10'],
    8: ['SiBF10', 'SiBF11', 'SiBF12', 'SiBF13', 'SiBF14','SiS1'],
}


    with open(pose_dir, 'r') as pose:
        init_pose = json.load(pose)
        for img_path, pose_data in init_pose.items():

            #print(os.path.join(source_dir,anno_json[id]['label_file']))
            seq_name = img_path.split('/')[-3]
            frame_name = img_path.split('/')[-1]
            seq_dir = os.path.join(source_dir, seq_name)
            img_file = os.path.join(seq_dir,'rgb',frame_name)
            anno_file = os.path.join(seq_dir,'meta',frame_name.replace('jpg', 'pkl'))

            with open(anno_file, 'rb') as f:
                anno = pickle.load(f, encoding='latin1')
            if anno['handBeta'] is None:
                continue

            #subject_id
            # 提取hand_beta并推断subject_id
            # 获取subject_id
            for key, value in seqname_to_subject_id.items():
                if seq_name in value:
                    subject_id = key
            #print(subject_id)

            trans_gt = np.array(anno['handTrans']).reshape(1, -1)
            #trans_gt*=np.array([1, -1, -1])
            pose_gt = np.array(anno['handPose'][3:]).reshape(1, -1)
            rot_gt = np.array(anno['handPose'][:3]).reshape(1, -1)
            #print(rot_gt)

            betas_gt = np.array(anno['handBeta']).reshape(1, -1)
            rot_gt = Rotation.from_rotvec(np.array(rot_gt).reshape([-1])).as_matrix()

            pose_torch_gt = torch.from_numpy(pose_gt).float()  # .cuda()
            betas_torch_gt = torch.from_numpy(betas_gt).float()  # .cuda()
            new_root_orient_gt = Rotation.from_matrix(rot_gt).as_rotvec().reshape([1, 3]).astype(np.float32)
            new_trans_gt = trans_gt.reshape([1, 3]).astype(np.float32)
            new_root_orient_torch_gt = torch.from_numpy(new_root_orient_gt).float()  # .cuda()
            new_trans_torch_gt = torch.from_numpy(new_trans_gt).float()  # .cuda()

            body_gt = body_model(global_orient=new_root_orient_torch_gt, hand_pose=pose_torch_gt,
                              betas=betas_torch_gt, transl=new_trans_torch_gt)

            joints_gt = body_gt['joints'].detach().cpu().numpy()
            # path = '/home/cyc/pycharm/lxy/3DGS/debug/ho3d/gt.obj'
            # with open(path, 'w') as fp:
            #     for v in body_gt['v'].detach().cpu().numpy()[0]:
            #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            #     for f in faces + 1:
            #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

            #print('gt', joints_gt[0])


            # obj
            obj_trans_gt = np.array(anno['objTrans']).astype(np.float32)
            # R33, _ = cv2.Rodrigues(np.array(anno['handRot']))
            obj_rot_gt = np.array(anno['objRot']).astype(np.float32)
            obj_3DCorners = np.array(anno['objCorners3D']).astype(np.float32)
            objCorners3DRest = np.array(anno['objCorners3DRest']).astype(np.float32)
            objName = anno['objName']

            mano_pose_6d = np.array(pose_data['mano_pose6d'])
            betas = np.array(pose_data['mano_shape']).reshape(-1, 10).astype(np.float32)

            pose = mano_pose_6d[3:48].reshape(1, -1).astype(np.float32)
            rot = mano_pose_6d[:3].reshape(1, -1).astype(np.float32)
            #print(rot)


            rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
            # rot_m = np.eye(3)
            #
            # rot_m[1,1] = -1
            # rot_m[2, 2] = -1
            R_x_180 = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            rot = R_x_180@rot
            new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1, 3]).astype(np.float32)
            #print('rot -1',new_root_orient)
            pose_torch = torch.from_numpy(pose)  # .cuda()
            betas_torch = torch.from_numpy(betas)  # .cuda()

            new_root_orient_torch = torch.from_numpy(new_root_orient)  # .cuda()
            new_trans_torch = new_trans_torch_gt
            body = body_model(betas=betas_torch)
            minimal_shape = body['v'].detach().cpu().numpy()[0]

            body = body_model(global_orient=new_root_orient_torch, hand_pose=pose_torch,
                              betas=betas_torch, transl=new_trans_torch)

            bone_transforms = body['bone_transforms'].detach().cpu().numpy()
            Jtr_posed = body['Jtr'].detach().cpu().numpy()
            joints = body['joints'].detach().cpu().numpy()

            # path = '/home/cyc/pycharm/lxy/3DGS/debug/ho3d/pred.obj'
            # with open(path, 'w') as fp:
            #     for v in body['v'].detach().cpu().numpy()[0]:
            #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            #     for f in faces + 1:
            #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


            #joints = joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
            joints_3d_gt = np.array(anno["handJoints3D"], dtype=np.float32)

            # pred_verts, pred_joints = mano_layer(th_pose_coeffs=torch.cat([new_root_orient_torch, pose_torch], dim=-1), th_betas=betas_torch, th_trans=new_trans_torch)
            # print('pred_mano', pred_joints[0])
            # obj
            obj_rot, obj_trans = Rotation.from_rotvec(
                np.array(pose_data['obj_rot'], dtype=np.float32).reshape([-1])).as_matrix(), \
                                 (np.array(pose_data['obj_trans'], dtype=np.float32) + np.array(
                                     pose_data['obj_center_cam'], dtype=np.float32)).reshape(3, 1)



            os.makedirs(os.path.join(seq_dir,'model_HOISDF'), exist_ok=True)
            out_filename = os.path.join(seq_dir,'model_HOISDF',frame_name.replace('jpg', 'npz'))

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
                     obj_3DCornersRest=objCorners3DRest,
                     subject_id=subject_id,
                     obj_label=_YCB_NAME2ID[objName],
                     # seg=anno['seg'],
                     trans_gt=new_trans_torch_gt[0].numpy(),
                     root_orient_gt=new_root_orient_torch_gt[0].numpy(),
                     pose_gt=pose_torch_gt[0].numpy(),
                     betas_gt=betas_torch_gt.numpy(),
                     obj_rot_gt=obj_rot_gt,
                     obj_trans_gt=obj_trans_gt,
                     joints_3d_gt=joints_3d_gt
                     )

            # image_file = os.path.join(source_dir,anno_json[id]['color_file'])
            # out_path = os.path.join(output_dir, subject_name, str(obj_label) + '-' + seq_name, cam_name)
            # out_filename = os.path.join(out_path,anno_json[id]['color_file'].split('/')[-1])
            # shutil.copy(image_file, out_filename)

            #print('pred',joints)


            errors = np.linalg.norm(joints[0] - joints_gt[0], axis=-1)
            mpjpe += errors.mean()
            print('MPJPE:', errors.mean())
            #exit()
            camera_intrinsics = np.array(anno['camMat'])
            K = camera_intrinsics
            D = np.array([0, 0, 0, 0, 0])
            R = np.eye(3)
            R[0, 0] = 1
            R[1, 1] = -1
            R[2, 2] = -1
            T = np.zeros((3, 1))
            cam_params = {'K': K.tolist(), 'D': D.tolist(), 'R': R.tolist(), 'T': T.tolist()}

            # if not os.path.isfile(os.path.join(seq_dir, 'cam_params.json')):
            #     with open(os.path.join(seq_dir, 'cam_params.json'), 'w') as f:
            #         json.dump(cam_params, f)



if __name__ == '__main__':

    generate_dexycb()
    # generate_dexycb(os.path.join(
    #                 cfg.annotation_dir, "dex_ycb_s0_train_data_cut.json"
    #             ))
