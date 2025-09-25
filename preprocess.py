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
    description='Preprocessing for ZJU-MoCap.'
)
parser.add_argument('--source_dir', type=str, default="C:\\Users\\HuiZhong\\Desktop\\HO3d", help='Directory that contains raw HO3D data.')
parser.add_argument('--out_dir', type=str, default="C:\\Users\\HuiZhong\\Desktop\\data_clean\\output", help='Directory where preprocessed data is saved.')
parser.add_argument('--seq_name', type=str, default='ABF1', help='Sequence to process.')
source_dir = "E:\\dataset\\HO3D_v3\\HO3D_v3\\HO3D_v3"
cam_num = [0, 1, 2, 3, 4]
output_dir = "C:\\Users\\HuiZhong\\Desktop\\data_clean\\output"
if __name__ == '__main__':
    args = parser.parse_args()
    out_dir = args.out_dir
    data_dir = os.path.join(args.source_dir, "train")
    cam_dir = os.path.join(args.source_dir, 'calibration\\{}\\calibration'.format(args.seq_name))
    body_model = MANO(model_path='C:\\Users\\HuiZhong\\Desktop\\data_clean\\hand_models\\mano')#.cuda()

    cam_names = cam_num
    faces = np.load("C:\\Users\\HuiZhong\\Desktop\\data_clean\\hand_models\\misc\\faces.npz")['faces']

    all_cam_params = {'all_cam_params': cam_names}
    for cam_idx, cam_name in enumerate(cam_names):
        cam_intri = os.path.join(cam_dir, 'cam_{}_intrinsics.txt'.format(cam_name))
        cam_trans = os.path.join(cam_dir, 'trans_{}.txt'.format(cam_name))
        params = {}
        with open(cam_intri, 'r') as f:
            intri = f.readlines()
            matches = re.findall(r'(\w+): ([\[\]0-9., ]+)', intri[0])
            for match in matches:
                param_name, param_value = match
                if param_value.startswith('[') and param_value.endswith(']'):
                    param_value = list(map(float, param_value[1:-1].split(',')))
                elif ',' in param_value:
                    param_value = float(param_value.split(', ')[0])
                else:
                    param_value = float(param_value)
                params[param_name] = param_value
        K = np.array([[params["fx"], 0, params["ppx"]],
                     [0, params["fy"], params["ppy"]],
                     [0, 0, 1]])
        D = np.array([0, 0, 0, 0, 0])
        R = np.eye(3)
        T = np.zeros((3, 1))
        cam_params = {'K': K.tolist(), 'D': D.tolist(), 'R': R.tolist(), 'T': T.tolist()}
        all_cam_params.update({cam_name: cam_params})
        
        sub_out_dir = os.path.join(output_dir, args.seq_name+'{}'.format(cam_name))
        if not os.path.exists(sub_out_dir):
            os.makedirs(sub_out_dir)
        seq = args.seq_name+str(cam_name)
        data_path = os.path.join(data_dir, seq)
        model_outdir = os.path.join(sub_out_dir, "model")
        if not os.path.exists(model_outdir):
            os.makedirs(model_outdir)
        img_outdir = os.path.join(sub_out_dir, "images")
        if not os.path.exists(img_outdir):
            os.makedirs(img_outdir)
        #process annotations
        anno_files = sorted(glob.glob(os.path.join(data_path, 'meta', '*.pkl')))
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
            np.savez(out_filename, 
                     minimal_shape=minimal_shape,
                     betas=betas,
                     Jtr_posed=Jtr_posed[0],
                     bone_transforms=bone_transforms[0],
                     trans=new_trans[0],
                     root_orient=new_root_orient[0],
                     pose=pose[0])
        
        #process images
        model_files = sorted(glob.glob(os.path.join(model_outdir, '*.npz')))
        model_basenames = sorted([os.path.basename(model_file)[:-4] for model_file in model_files])
        for model_basename in model_basenames:
            image_file = os.path.join(data_path, 'rgb', model_basename+'.jpg')
            out_filename = os.path.join(img_outdir, os.path.basename(image_file))
            shutil.copy(image_file, out_filename)
    with open(os.path.join(output_dir, 'cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)