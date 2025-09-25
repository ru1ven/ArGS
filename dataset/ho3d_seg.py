import os
import sys
import glob
import cv2

from right_hand_model import MANO
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera, Camera_multi_batch
from utils.camera_utils import freeview_camera

import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh


class HO3DDataset(Dataset):
    def __init__(self, cfg, split='train', test_split=None, multi_batch=False):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.root_dir = cfg.root_dir
        self.multi_batch = multi_batch

        self.white_bg = cfg.white_background
        self.H, self.W = 480, 640
        self.h, self.w = cfg.img_hw

        self.faces = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/faces.npz')['faces']
        self.skinning_weights = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/skinning_weights_all.npz')[
            'rightHand']
        self.posedirs = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/posedirs_all.npz')['rightHand']
        self.J_regressor = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/J_regressors.npz')['rightHand']

        self.cam_params = {}
        self.body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/', flat_hand_mean=True)

        #self.filter_num = 0

        if split == 'train':
            seq_names = cfg.train_view

        elif split == 'val':
            seq_names = cfg.val_view

        elif split == 'test':
            seq_names = cfg.test_view

        elif split == 'predict':
            seq_names = cfg.predict_view

        else:
            return ValueError

        self.data = []
        self.model_file_list = []
        for seq_idx, seq_name in enumerate(seq_names):
            data_dir = os.path.join(self.root_dir, seq_name)

            with open(os.path.join(data_dir, 'cam_params.json'), 'r') as f:
                self.cam_params[seq_name] = json.load(f)

            if split == 'train':
                model_files = sorted(glob.glob(os.path.join(data_dir, 'model_HOISDF', '*.npz')))
            else:
                model_files = []
                with open(os.path.join(data_dir, 'corres.txt'), 'r') as f:
                    filenames = f.readlines()

                for filename in filenames:
                    if os.path.isfile(os.path.join(data_dir, 'model_HOISDF',filename.strip().replace('jpg','npz'))):
                        model_files.append(os.path.join(data_dir, 'model_HOISDF',filename.strip().replace('jpg','npz')))


            self.model_files = model_files

            img_files = []
            mask_files = []
            sam_files = []
            for model_file in self.model_files:
                img_file = os.path.join(data_dir, 'rgb', os.path.basename(model_file)[:-4] + '.jpg')
                mask_file = os.path.join(data_dir, 'seg', os.path.basename(model_file)[:-4] + '.png')
                sam_file = os.path.join(data_dir, 'hand_mask', os.path.basename(model_file)[:-4] + '.png')
                img_files.append(img_file)
                mask_files.append(mask_file)
                sam_files.append(sam_file)

            for d_idx, value in enumerate(self.model_files):
                self.data.append(
                    {
                        'seq_idx': seq_idx,
                        'seq_name': seq_name,
                        'data_idx': d_idx,
                        'frame_idx': d_idx,
                        'img_file': img_files[d_idx],
                        'mask_file': mask_files[d_idx],
                        'sam_file': sam_files[d_idx],
                        'model_file': value
                    }
                )


            self.model_file_list.extend(self.model_files)
        self.frames = list(range(len(self.model_files)))

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
            frame: i for i, frame in enumerate(self.frames)
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 1.0395,
            'frame_dict': frame_dict
        }
        self.metadata.update(cano_data)
        if (self.cfg.train_smpl):
            self.metadata.update(self.get_mano_data())

    def get_obj_data(self):
        self.metadata_obj = {}
        for data_path in self.model_file_list:
            model_dict = np.load(data_path)
            obj_label = model_dict['obj_label']
            obj3DCornerRest = model_dict['obj_3DCornersRest']

            # obj_trans = model_dict['obj_trans_gt']
            # obj_rots = model_dict['obj_rot_gt']
            # obj_rot, _ = cv2.Rodrigues(np.array(obj_rots))
            #
            # # 计算反变换矩阵
            # R_inv = np.linalg.inv(obj_rot)
            # T_inv = -np.dot(R_inv, obj_trans)
            # # 进行反变换
            # obj3DCornerRest = np.dot(obj_3DCorners, R_inv.T) + T_inv
            self.objdata = {int(obj_label): {
                'obj3DCorners': obj3DCornerRest,
                'obj_aabb': AABB(obj3DCornerRest.min(axis=0), obj3DCornerRest.max(axis=0))
            }}

            self.metadata_obj.update(self.objdata)

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

        # bone_transforms = model_dict['bone_transforms']
        bone_transforms = np.repeat(np.eye(4)[np.newaxis, ...], 16, axis=0)
        T = np.matmul(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])
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
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights,
            'bone_transforms': bone_transforms,
            'cano_mesh': cano_mesh,

            'coord_max': coord_max,
            'coord_min': coord_min,
            'aabb': AABB(coord_min, coord_max),
        }

    def get_mano_data(self):
        if self.split != 'train':
            return {}

        from collections import defaultdict
        mano_data = defaultdict(list)

        for idx, (frame, model_file) in enumerate(zip(self.frames, self.model_file_list)):
            model_dict = np.load(model_file)
            if idx == 0:
                mano_data['betas'] = model_dict['betas'].astype(np.float32)

            mano_data['frames'].append(frame)
            mano_data['root_orient'].append(model_dict['root_orient'].astype(np.float32))
            mano_data['pose_hand'].append(model_dict['pose'].astype(np.float32))
            mano_data['trans'].append(model_dict['trans'].astype(np.float32))

        return mano_data

    def process_bbox(self, bbox, img_width, img_height, expansion_factor=1.25):
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
        if w * h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])
        else:
            return None

        # aspect ratio preserving bbox
        w = bbox[2]
        h = bbox[3]
        c_x = bbox[0] + w / 2.
        c_y = bbox[1] + h / 2.
        aspect_ratio = 1
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bbox[2] = w * expansion_factor
        bbox[3] = h * expansion_factor
        bbox[0] = c_x - bbox[2] / 2.
        bbox[1] = c_y - bbox[3] / 2.

        return bbox

    def generate_patch_image(self, cvimg, bbox, input_shape, do_flip, scale, rot):
        """
        @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      generate the patch image from the bounding box and other parameters.
        ---------
        @param: input image, bbox(x1, y1, h, w), dest image shape, do_flip, scale factor, rotation degrees.
        -------
        @Returns: processed image, affine_transform matrix to get the processed image.
        -------
        """

        img = cvimg.copy()
        img_height, img_width, _ = img.shape

        bb_c_x = float(bbox[0] + 0.5 * bbox[2])
        bb_c_y = float(bbox[1] + 0.5 * bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

        if do_flip:
            img = img[:, ::-1, :]
            bb_c_x = img_width - bb_c_x - 1

        trans = self.gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale,
                                             rot, inv=False)
        img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
        new_trans = np.zeros((3, 3), dtype=np.float32)
        new_trans[:2, :] = trans
        new_trans[2, 2] = 1

        return img_patch, new_trans

    def gen_trans_from_patch_cv(self, c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
        """
        @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      get affine transform matrix
        ---------
        @param: image center, original image size, desired image size, scale factor, rotation degree, whether to get inverse transformation.
        -------
        @Returns: affine transformation matrix
        -------
        """

        def rotate_2d(pt_2d, rot_rad):
            x = pt_2d[0]
            y = pt_2d[1]
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            xx = x * cs - y * sn
            yy = x * sn + y * cs
            return np.array([xx, yy], dtype=np.float32)

        # augment size with scale
        src_w = src_width * scale
        src_h = src_height * scale
        src_center = np.array([c_x, c_y], dtype=np.float32)

        # augment rotation
        rot_rad = np.pi * rot / 180
        src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
        src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

        dst_w = dst_width
        dst_h = dst_height
        dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
        dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = src_center
        src[1, :] = src_center + src_downdir
        src[2, :] = src_center + src_rightdir

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_downdir
        dst[2, :] = dst_center + dst_rightdir

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]

        seq_name = data_dict['seq_name']

        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        model_file = data_dict['model_file']

        K = np.array(self.cam_params[seq_name]['K'], dtype=np.float32).copy()
        dist = np.array(self.cam_params[seq_name]['D'], dtype=np.float32).ravel()
        R = np.array(self.cam_params[seq_name]['R'], np.float32)
        R[0, 0] = 1
        R[1, 1] = -1
        R[2, 2] = -1

        T = np.array(self.cam_params[seq_name]['T'], np.float32)

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # hand 65-85
        # obj 20-35
        image = cv2.resize(image, (self.w, self.h))

        ori_image = image.copy()
        ori_image=ori_image /255.
        ori_image = torch.from_numpy(ori_image).permute(2, 0, 1).float()

        obj_image = image.copy()
        obj_mask = (mask == 149)
        obj_mask = cv2.resize(obj_mask.astype(np.uint8), (self.w, self.h))
        obj_image[obj_mask == 0] = 255. if self.white_bg else 0
        obj_image = obj_image / 255.
        obj_image = torch.from_numpy(obj_image).permute(2, 0, 1).float()

        hand_image = image.copy()
        hand_mask = (mask == 29)
        # hand_mask = cv2.resize(hand_mask.astype(np.uint8),(self.w,self.h),interpolation=cv2.INTER_NEAREST)
        hand_mask = cv2.resize(hand_mask.astype(np.uint8), (self.w, self.h))
        hand_image[hand_mask == 0] = 255. if self.white_bg else 0.
        hand_image = hand_image / 255.
        hand_image = torch.from_numpy(hand_image).permute(2, 0, 1).float()
        # hand_mask = torch.from_numpy(hand_mask).unsqueeze(0).float()
        full_mask =  (mask != 0)
        full_mask = cv2.resize(full_mask.astype(np.uint8), (self.w, self.h))

        # hand_image = image.copy()
        # hand_mask = cv2.imread(sam_file, cv2.IMREAD_GRAYSCALE)
        # # _, hand_mask = cv2.threshold(hand_mask, 1, 1, cv2.THRESH_BINARY)
        # hand_image[hand_mask != 162] = 255. if self.white_bg else 0.
        # hand_image = hand_image / 255.
        # hand_image = torch.from_numpy(hand_image).permute(2, 0, 1).float()
        # hand_mask[hand_mask!=162]=0
        # _, hand_mask = cv2.threshold(hand_mask, 1, 1, cv2.THRESH_BINARY)
        # full_mask = hand_mask|obj_mask

        obj_mask = torch.from_numpy(obj_mask).unsqueeze(0).float()
        hand_mask = torch.from_numpy(hand_mask).unsqueeze(0).float()

        full_image = image.copy()

        min_y, max_y = np.where(full_mask)[0].min(), np.where(full_mask)[0].max() + 1
        min_x, max_x = np.where(full_mask)[1].min(), np.where(full_mask)[1].max() + 1
        # bbox = [x1, y1, max(y2 - y1, x2 - x1), max(y2 - y1, x2 - x1)]

        c_x = int((max_x + min_x) / 2)
        c_y = int((max_y + min_y) / 2)
        bbox_delta_x = (max_x - min_x)  / 2
        bbox_delta_y = (max_y - min_y)  / 2
        bbox_delta = max(bbox_delta_x,bbox_delta_y)
        bbox = [c_x-bbox_delta, c_y-bbox_delta, bbox_delta*2, bbox_delta*2]

        bbox = self.process_bbox(bbox, self.w, self.h, 1.2)
        trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]
        # bbox[0] = bbox[0] + bbox[2] * trans[0]
        # bbox[1] = bbox[1] + bbox[3] * trans[1]
        roi_size = self.cfg.get('roi_size', 224)
        img_ROI, trans_img2roi = self.generate_patch_image(full_image, bbox, [roi_size, roi_size], do_flip, scale, rot)

        # cv2.imwrite("/home/cyc/pycharm/lxy/visual/img_ROI.png",cv2.cvtColor(np.uint8(img_ROI), cv2.COLOR_BGR2RGB))

        img_ROI = img_ROI / 255.
        img_ROI = torch.from_numpy(img_ROI).permute(2, 0, 1).float()

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

        # compute posed mano hand
        minimal_shape = self.metadata['minimal_shape']
        model_dict = np.load(model_file)
        obj_id = model_dict['obj_label']
        #subject_id = seq_name.split('-')[0]
        subject_id =  model_dict['subject_id']
        # print(seq_name)
        #
        # print(subject_id)
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

        root_orient_gt = torch.from_numpy(model_dict['root_orient_gt']).float().reshape(-1, 3)
        pose_gt = torch.from_numpy(model_dict['pose_gt']).float().reshape(-1, 45)
        betas_gt = torch.from_numpy(model_dict['betas_gt']).float().reshape(-1, 10)
        trans_gt = torch.from_numpy(model_dict['trans_gt']).float().reshape(-1, 3)
        hand_param_gt = torch.cat([root_orient_gt, pose_gt, betas_gt, trans_gt], dim=-1)
        obj_rot_gt = model_dict['obj_rot_gt']
        obj_rot_gt, _ = cv2.Rodrigues(np.array(obj_rot_gt))
        obj_trans_gt = model_dict['obj_trans_gt']

        # bone_transforms gt
        body = self.body_model(global_orient=root_orient_gt, hand_pose=pose_gt, betas=betas_gt,
                              transl=trans_gt)
        bone_transforms = body['bone_transforms'][0].detach().cpu().numpy()


        bone_transforms1 = self.metadata['bone_transforms']
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms1)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:, :3, 3] += trans

        obj_rot = model_dict["obj_rot"].reshape(3,3)
        #obj_rot, _ = cv2.Rodrigues(np.array(obj_rot))
        obj_trans = model_dict["obj_trans"]



        root_orient = model_dict['root_orient'].astype(np.float32)
        pose = model_dict['pose'].astype(np.float32)
        betas = model_dict["betas"]
        root_orient = torch.from_numpy(root_orient).float().reshape(-1, 3)
        pose = torch.from_numpy(pose).float().reshape(-1, 45)
        trans = torch.from_numpy(trans).float().reshape(-1, 3)
        betas = torch.from_numpy(betas).float().reshape(-1, 10)
        hand_param = torch.cat([root_orient, pose, betas, trans], dim=-1)
        obj_trans = obj_trans.reshape(3)* np.array([1, -1, -1])

        R_x_180 = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        obj_rot = R_x_180@obj_rot

        def generate_alpha_gt(seed=None) -> float:
            """
            生成单个符合概率分布的alpha_gt标量
            概率分布：
                20% 为 1.0
                60% 为 0.9
                15% 为 0.5
                5% 为 0.0
            参数:
                seed: 随机种子（确保可重复性）
            返回:
                alpha_gt: Python浮点标量（非numpy类型）
            """
            if seed is not None:
                np.random.seed(seed)

            rand_val = np.random.rand()  # 生成[0,1)的随机数

            if rand_val < 0.6:
                return 1.0
            elif rand_val < 0.8:
                return 0.95
            elif rand_val < 0.95:
                return 0.8
            else:  # 最后5%
                return 0.6

        alpha_gt = generate_alpha_gt()
        obj_rot_noise = obj_rot_gt*alpha_gt+obj_rot*(1-alpha_gt)
        alpha_gt = generate_alpha_gt()
        obj_trans_noise = obj_trans_gt * alpha_gt+obj_trans * (1-alpha_gt)

        if self.multi_batch:
            camera = Camera_multi_batch(
                frame_id=frame_idx,
                cam_id=seq_name,
                subject_id=torch.tensor(int(subject_id)),
                obj_id=torch.tensor(int(obj_id)),
                K=K, R=R, T=np.squeeze(T),
                bbox=torch.from_numpy(bbox),
                FoVx=FovX,
                FoVy=FovY,
                image=hand_image,
                mask=hand_mask,
                obj_image=obj_image,
                obj_mask=obj_mask,
                full_image=full_image,
                full_mask=full_mask,
                img_ROI=img_ROI,
                trans_img2roi=trans_img2roi,

                image_name=f"c{seq_name}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
                data_device=self.cfg.data_device,
                # human params
                rots=torch.from_numpy(pose_rot).float(),
                Jtrs=torch.from_numpy(Jtr_norm).float(),
                bone_transforms=torch.from_numpy(bone_transforms),
                # obj params
                obj_rots=torch.from_numpy(obj_rot_noise).float().view(3, 3),
                obj_trans=torch.from_numpy(obj_trans_noise).float().view(3),
                hand_param=hand_param_gt.reshape(61),
                hand_param_gt=hand_param_gt.reshape(61),
                obj_rots_gt=torch.from_numpy(obj_rot_gt).float().view(3, 3),
                obj_trans_gt=torch.from_numpy(obj_trans_gt).float().view(3),
                joints_gt=torch.from_numpy(model_dict['joints_3d_gt']).float().view(21, 3),
                # Jtrs_gt=torch.from_numpy(Jtr_gt).float().unsqueeze(0)
                hand_root=torch.from_numpy(Jtr[0]).float()
            )
            return camera.data
        else:
            return Camera(
                frame_id=frame_idx,
                cam_id=seq_name,
                subject_id=int(subject_id),
                obj_id=int(obj_id),
                K=K, R=R, T=np.squeeze(T),
                FoVx=FovX,
                FoVy=FovY,
                bbox=torch.from_numpy(bbox),
                image=hand_image,
                ori_img=ori_image,
                mask=hand_mask,
                obj_image=obj_image,
                obj_mask=obj_mask,
                full_image=full_image,
                full_mask=full_mask,
                img_ROI=img_ROI,
                trans_img2roi=trans_img2roi,
                gt_alpha_mask=None,
                image_name=f"c{seq_name}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
                data_device=self.cfg.data_device,
                # human params
                rots=torch.from_numpy(pose_rot).float(),
                Jtrs=torch.from_numpy(Jtr_norm).float(),
                bone_transforms=torch.from_numpy(bone_transforms),
                # obj params
                obj_rots=torch.from_numpy(obj_rot_noise).float().view(3, 3),
                obj_trans=torch.from_numpy(obj_trans_noise).float().view(3),

                hand_param=hand_param_gt.view(1, 61),
                hand_param_gt=hand_param_gt.view(1, 61),
                obj_rots_gt=torch.from_numpy(obj_rot_gt).float().view(3, 3),
                obj_trans_gt=torch.from_numpy(obj_trans_gt).float().view(3),
                joints_gt=torch.from_numpy(model_dict['joints_3d_gt']).float().squeeze(0).view(21, 3),
                # Jtrs_gt=torch.from_numpy(Jtr_gt).float().unsqueeze(0)
                hand_root=torch.from_numpy(Jtr[0]).float()
            )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    def readPointCloud(self, sub_id):

        ply_path = os.path.join(self.root_dir, 'cano_mano_{}.ply'.format(sub_id))
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

    def randomPointCloud(self, obj_id):
        ply_path = os.path.join(self.root_dir, 'random_pc_obj_{}.ply'.format(obj_id))
        n_points = 5000

        objcoords = self.metadata_obj[obj_id]['obj3DCorners']
        coord_min = objcoords.min(axis=0)
        coord_max = objcoords.max(axis=0)
        xyz = np.random.uniform(coord_min, coord_max, (n_points, 3))
        rgb = np.ones_like(xyz) * 255
        storePly(ply_path, xyz, rgb)

        pcd = fetchPly(ply_path)

        return pcd
