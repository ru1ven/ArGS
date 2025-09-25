import os
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera
from utils.camera_utils import freeview_camera


import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh

class HO3DDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg 
        self.split = split
        self.root_dir = cfg.root_dir

        self.subject = cfg.subject
        self.white_bg = cfg.white_background
        self.H, self.W = 480, 640
        self.h, self.w = cfg.img_hw

        self.faces = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/faces.npz')['faces']
        self.skinning_weights = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/skinning_weights_all.npz')['rightHand']
        self.posedirs = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/posedirs_all.npz')['rightHand']
        self.J_regressor = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/J_regressors.npz')['rightHand']

        if split == 'train':
            cam_names = cfg.train_view
            sample_rate = cfg.train_sample_rate
        elif split == 'val':
            cam_names = cfg.val_view
            sample_rate = cfg.val_sample_rate
        elif split == 'test':
            cam_names = cfg.test_view
            sample_rate = cfg.test_sample_rate
        elif split == 'predict':
            cam_names = cfg.predict_view
            sample_rate = cfg.predict_sample_rate
        else:
            return ValueError
    
        with open(os.path.join(self.root_dir, 'cam_params.json'), 'r') as f:
            self.cam_params = json.load(f)
        
        if len(cam_names) == 0:
            cam_names = self.cam_params['all_cam_params']
        
        self.data = []
        if split == 'predict':
           for cam_idx, cam_name in enumerate(cam_names):
                data_dir = os.path.join(self.root_dir, self.subject+str(cam_name))
                model_files = sorted(glob.glob(os.path.join(data_dir, 'model', '*.npz')))
                self.model_files = model_files[::sample_rate]
                img_files = []
                mask_files = []
                for model_file in self.model_files:
                    img_file = os.path.join(data_dir, 'rgb', os.path.basename(model_file)[:-4]+'.png')
                    mask_file = os.path.join(data_dir, 'hand_mask', os.path.basename(model_file)[:-4] + '.png')
                    img_files.append(img_file)
                    mask_files.append(mask_file)

                for d_idx, value in enumerate(self.model_files):
                    self.data.append(
                        {
                            'cam_idx': cam_idx,
                            'cam_name': cam_name,
                            'data_idx': d_idx,
                            'frame_idx': d_idx,
                            'img_file': img_files[d_idx],
                            'mask_file': mask_files[d_idx],
                            'model_file': value
                        }
                    )
        else:
            for cam_idx, cam_name in enumerate(cam_names):
                data_dir = os.path.join(self.root_dir, self.subject+str(cam_name))
                model_files = sorted(glob.glob(os.path.join(data_dir, 'model', '*.npz')))
                self.model_files = model_files[::sample_rate]
                img_files = []
                mask_files = []
                for model_file in self.model_files:
                    img_file = os.path.join(data_dir, 'rgb', os.path.basename(model_file)[:-4]+'.png')
                    mask_file = os.path.join(data_dir, 'hand_mask', os.path.basename(model_file)[:-4] + '.png')
                    img_files.append(img_file)
                    mask_files.append(mask_file)

                for d_idx, value in enumerate(self.model_files):
                    self.data.append(
                        {
                            'cam_idx': cam_idx,
                            'cam_name': cam_name,
                            'data_idx': d_idx,
                            'frame_idx': d_idx,
                            'img_file': img_files[d_idx],
                            'mask_file': mask_files[d_idx],
                            'model_file': value
                        }
                    )
        self.frames = list(range(len(self.model_files)))
        self.model_file_list = model_files

        self.get_metadata()
        self.get_obj_data()

        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]
    
    def get_metadata(self):
        data_paths = self.model_files
        data_path = data_paths[0]

        cano_data = self.get_cano_mano_verts(data_path)
        if self.split != 'train':
            self.metadata = cano_data
            return

        frame_dict = {
            frame : i for i, frame in enumerate(self.frames)
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797,
            'frame_dict': frame_dict
        }
        self.metadata.update(cano_data)
        if(self.cfg.train_smpl):
            self.metadata.update(self.get_mano_data())

    def get_obj_data(self):
        data_path = self.model_files
        data_path = data_path[0]
        model_dict = np.load(data_path)
        obj_3DCorners = model_dict['obj_3DCorners']
        obj_trans = model_dict['obj_trans']
        obj_rots = model_dict['obj_rot']
        obj_rot, _ = cv2.Rodrigues(np.array(obj_rots))
        # 计算反变换矩阵
        R_inv = np.linalg.inv(obj_rot)
        T_inv = -np.dot(R_inv, obj_trans)
        # 进行反变换
        obj3DCornerRest = np.dot(obj_3DCorners, R_inv.T) + T_inv
        self.objdata = {
            'obj3DCorners': obj3DCornerRest,
            'obj_aabb': AABB(obj3DCornerRest.min(axis=0), obj3DCornerRest.max(axis=0))
        }
        self.metadata.update(self.objdata)

    def get_cano_mano_verts(self, data_path):
         # compute scale from Mano
        model_dict = np.load(data_path)
        # 3D models and points
        minimal_shape = model_dict['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else: 
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights
        
        #bone_transforms = model_dict['bone_transforms']
        bone_transforms = np.repeat(np.eye(4)[np.newaxis, ...], 16, axis=0)
        T = np.matmul(skinning_weights, bone_transforms.reshape([-1,16])).reshape([-1,4,4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]                
    
        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding    

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)

        return {
            'smpl_verts':vertices.astype(np.float32),
            'minimal_shape':minimal_shape,
            'Jtr':Jtr,
            'skinning_weights':skinning_weights,
            'bone_transforms':bone_transforms,
            'cano_mesh':cano_mesh,

            'coord_max':coord_max,
            'coord_min':coord_min,
            'aabb':AABB(coord_min, coord_max),
        }
    
    def get_mano_data(self):
        if self.split != 'train':
            return {}
        
        from collections import defaultdict
        mano_data = defaultdict(list)

        for idx, (frame,model_file) in enumerate(zip(self.frames, self.model_file_list)):
            model_dict = np.load(model_file)
            if idx == 0:
                mano_data['betas'] = model_dict['betas'].astype(np.float32)
            
            mano_data['frames'].append(frame)
            mano_data['root_orient'].append(model_dict['root_orient'].astype(np.float32))
            mano_data['pose_hand'].append(model_dict['pose'].astype(np.float32))
            mano_data['trans'].append(model_dict['trans'].astype(np.float32))

        return mano_data
    
    def __len__(self):
        return len(self.data)
    
    def getitem(self,idx,data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]
        
        cam_idx = data_dict['cam_idx']
        cam_name = data_dict['cam_name']
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        model_file = data_dict['model_file']

        K = np.array(self.cam_params[cam_name]['K'], dtype=np.float32).copy()
        dist = np.array(self.cam_params[cam_name]['D'], dtype=np.float32).ravel()
        R = np.array(self.cam_params[cam_name]['R'], np.float32)
        R[0,0]= 1
        R[1,1]= -1
        #R[2,2]= 1

        T = np.array(self.cam_params[cam_name]['T'], np.float32)

        
        image = cv2.cvtColor(cv2.imread(img_file),cv2.COLOR_BGR2RGB)
        #hand 65-85
        #obj 20-35
        image = cv2.resize(image,(self.w,self.h))
        hand_image = image.copy()
        hand_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        #_, hand_mask = cv2.threshold(hand_mask, 1, 1, cv2.THRESH_BINARY)
        hand_image[hand_mask != 162] = 255. if self.white_bg else 0.
        hand_image = hand_image / 255.
        hand_image = torch.from_numpy(hand_image).permute(2, 0, 1).float()

        obj_image = image.copy()
        obj_mask = cv2.imread(mask_file.replace('hand_mask', 'obj_mask'), cv2.IMREAD_GRAYSCALE)
        _, obj_mask = cv2.threshold(obj_mask, 1, 1, cv2.THRESH_BINARY)
        obj_image[obj_mask == 0] = 255. if self.white_bg else 0
        obj_image = obj_image /255.
        obj_image = torch.from_numpy(obj_image).permute(2, 0, 1).float()

        
        full_image = image.copy()
        full_mask = hand_mask | obj_mask
        hand_mask = torch.from_numpy(hand_mask).unsqueeze(0).float()
        obj_mask = torch.from_numpy(obj_mask).unsqueeze(0).float()
        #full_mask = cv2.resize(full_mask.astype(np.uint8),(self.w,self.h))
        full_image[full_mask == 0] = 255. if self.white_bg else 0.
        full_image = full_image / 255.
        full_image = torch.from_numpy(full_image).permute(2, 0, 1).float()
        full_mask = torch.from_numpy(full_mask).unsqueeze(0).float()
        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        #compute posed mano hand
        minimal_shape = self.metadata['minimal_shape']
        model_dict = np.load(model_file)
        n_mano_points = minimal_shape.shape[0]
        trans = model_dict['trans'].astype(np.float32)
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)
        pose = model_dict['pose'].astype(np.float32)
        pose = np.concatenate([root_orient, pose], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))
        pose_mat_full = pose.as_matrix()
        pose_mat = pose_mat_full[1:, ...].copy()
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])
        ###
        Jtr = self.metadata['Jtr']
        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05
       
        # compute pose condition
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        bone_transforms1 = self.metadata['bone_transforms']
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms1)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:,:3, 3] += trans

        obj_rot = model_dict["obj_rot"]
        obj_trans = model_dict["obj_trans"]
        
        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=np.squeeze(T),
            FoVx=FovX,
            FoVy=FovY,
            image=hand_image,
            mask=hand_mask,
            obj_image = obj_image,
            obj_mask = obj_mask,
            full_image = full_image,
            full_mask = full_mask,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            # human params
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms),
            # obj params
            obj_rots = torch.from_numpy(obj_rot).float().unsqueeze(0),
            obj_trans = torch.from_numpy(obj_trans).float().unsqueeze(0),
        )
    def __getitem__(self,idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)
    
    def readPointCloud(self):
        if self.cfg.get('random_init', False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        else:
            ply_path = os.path.join(self.root_dir, 'cano_mano.ply')
            try:
                pcd = fetchPly(ply_path)
            except:
                verts = self.metadata['smpl_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 5000

                xyz = mesh.sample(n_points)
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)

                pcd = fetchPly(ply_path)

        return pcd
    
    def randomPointCloud(self):
        ply_path = os.path.join(self.root_dir, 'random_objpc.ply')
        n_points = 2000
        objcoords = self.metadata['obj3DCorners']
        coord_min = objcoords.min(axis=0)
        coord_max = objcoords.max(axis=0)
        xyz = np.random.uniform(coord_min, coord_max, (n_points, 3))
        rgb = np.ones_like(xyz) * 255
        storePly(ply_path, xyz, rgb)

        pcd = fetchPly(ply_path)

        return pcd