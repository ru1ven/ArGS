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

from scipy.spatial.transform import Rotation

from right_hand_model.body_models import MANO

parser = argparse.ArgumentParser(
    description='Preprocessing for HO3D.'
)
parser.add_argument('--source_dir', type=str, default="/mnt/sda1/lxy/3DGS/HO3D_v3/train/", help='Directory that contains raw HO3D data.')
source_dir = "/mnt/sda1/lxy/3DGS/HO3D_v3/train/"

if __name__ == '__main__':
    args = parser.parse_args()
    source_dir = args.source_dir
    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/')#.cuda()

    faces = np.load("/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/misc/faces.npz")['faces']



    #all_cam_params = {'all_cam_params': cam_names}
    for cam_name in os.listdir(source_dir):

        params = {}
        camera_intrinsics = None
        
        data_dir = os.path.join(source_dir, cam_name)


        model_outdir = os.path.join(data_dir, "model")
        if not os.path.exists(model_outdir):
            os.makedirs(model_outdir)
        # img_outdir = os.path.join(data_dir, "images")
        # if not os.path.exists(img_outdir):
        #     os.makedirs(img_outdir)
        #process annotations
        anno_files = sorted(glob.glob(os.path.join(data_dir, 'meta', '*.pkl')))
        #print(os.path.join(data_path, 'meta', '*.pkl'))
        for anno_file in anno_files:
            with open(anno_file, 'rb') as f:
                anno = pickle.load(f, encoding='latin1')
            if(anno['handBeta'] is None):
                continue
            trans = np.array(anno['handTrans']).reshape(1, -1)
            pose = np.array(anno['handPose'][3:]).reshape(1, -1)
            rot = np.array(anno['handPose'][:3]).reshape(1, -1)
            betas = np.array(anno['handBeta']).reshape(1, -1)
            rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
            pose_torch = torch.from_numpy(pose)#.cuda()
            betas_torch = torch.from_numpy(betas)#.cuda()
            new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1,3]).astype(np.float32)
            new_trans = trans.reshape([1, 3]).astype(np.float32)
            new_root_orient_torch = torch.from_numpy(new_root_orient)#.cuda()
            new_trans_torch = torch.from_numpy(new_trans)#.cuda()

            body = body_model(betas=betas_torch)
            minimal_shape = body['v'].detach().cpu().numpy()[0]

            body = body_model(global_orient=new_root_orient_torch, hand_pose=pose_torch, betas=betas_torch, transl=new_trans_torch)

            out_filename = os.path.join(model_outdir, os.path.basename(anno_file)[:-4])
            bone_transforms = body['bone_transforms'].detach().cpu().numpy()
            Jtr_posed = body['Jtr'].detach().cpu().numpy()

            #obj
            obj_trans = np.array(anno['objTrans']).astype(np.float32)
            #R33, _ = cv2.Rodrigues(np.array(anno['handRot']))
            obj_rot = np.array(anno['objRot']).astype(np.float32)

            obj_3DCorners = np.array(anno['objCorners3D']).astype(np.float32)

            obj_label = np.array(anno['objLabel'])

            np.savez(out_filename, 
                     minimal_shape=minimal_shape,
                     betas=betas,
                     Jtr_posed=Jtr_posed[0],
                     bone_transforms=bone_transforms[0],
                     trans=new_trans[0],
                     root_orient=new_root_orient[0],
                     pose=pose[0],
                     obj_trans=obj_trans,
                     obj_rot=obj_rot,
                     obj_3DCorners=obj_3DCorners,
                     obj_label=obj_label,
                     )
            camera_intrinsics = np.array(anno['camMat'])
            print(camera_intrinsics)

        K = camera_intrinsics
        D = np.array([0, 0, 0, 0, 0])
        R = np.eye(3)
        R[0, 0] = 1
        R[1, 1] = -1
        R[2, 2] = -1
        T = np.zeros((3, 1))
        cam_params = {'K': K.tolist(), 'D': D.tolist(), 'R': R.tolist(), 'T': T.tolist()}
        #all_cam_params.update({cam_name: cam_params})
        
        #process images
        model_files = sorted(glob.glob(os.path.join(model_outdir, '*.npz')))
        model_basenames = sorted([os.path.basename(model_file)[:-4] for model_file in model_files])
        # for model_basename in model_basenames:
        #     image_file = os.path.join(data_path, 'rgb', model_basename+'.jpg')
        #     out_filename = os.path.join(img_outdir, os.path.basename(image_file))
        #     shutil.copy(image_file, out_filename)
        with open(os.path.join(data_dir, 'cam_params.json'), 'w') as f:
            json.dump(cam_params, f)