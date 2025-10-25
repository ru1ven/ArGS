import os
import random
import sys
import glob
import cv2
import common.transforms as tf
from common import data_utils
from common.data_utils import read_img
from common.mesh import Mesh
from common.object_tensors import ObjectTensors
from right_hand_model import MANO
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB, get_valid, pad_jts2d, \
    transform_2d_for_speedup, apply_w2c_pose_numpy, update_K_after_bbox_crop_resize
from scene.cameras import Camera
from utils.camera_utils import freeview_camera

import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh


class ArcticDataset(Dataset):
    def __init__(self, cfg, split='train', test_split='SDF', multi_batch=False):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self._load_data(cfg)
        #self.root_dir = os.path.join(cfg.root_dir, split)
        self.root_dir = cfg.root_dir
        self.multi_batch = multi_batch

        self.white_bg = cfg.white_background
        self.H, self.W = 1000, 1000
        self.h, self.w = cfg.img_hw


        #self.faces = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/faces.npz')['faces']
        # self.skinning_weights = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/skinning_weights_all.npz')[
        #     'rightHand']
        # self.posedirs = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/posedirs_all.npz')['rightHand']
        # self.J_regressor = np.load('/home/cyc/pycharm/lxy/gs/ho_gs/hand_models/misc/J_regressors.npz')['rightHand']
        self.skinning_weights = np.load('./hand_models/misc/skinning_weights_all.npz')
        self.posedirs = np.load('./hand_models/misc/posedirs_all.npz')
        self.J_regressor = np.load('./hand_models/misc/J_regressors.npz')


        self.cam_params = {}

        self.body_model_r = MANO(model_path='/mnt/sda2/lxy/arctic/unpack/body_models/mano/', flat_hand_mean=False)  # .cuda()
        self.body_model_l = MANO(model_path='/mnt/sda2/lxy/arctic/unpack/body_models/mano/',
                            is_rhand=False, flat_hand_mean=False)
        self.faces = {'right': self.body_model_r.faces, 'left':self.body_model_l.faces}
        #self.color_scales = np.load("../lib/color_scales_25k.npy")

        self.metadata = {}
        self.get_metadata('right')
        self.get_metadata('left')
        self.get_obj_data()

    def _load_data(self, cfg):
        self.data = {}
        self.imgnames = {}
        if self.split == 'train':
            data_p = os.path.join(
                f"/mnt/sda2/lxy/dataset/hand/arctic_seqs/splits/train/{cfg.split_train}.npy"
            )
        else:
            self.train_view_id = int(cfg.split_train.split('_')[-1])
            data_p = os.path.join(
                f"/mnt/sda2/lxy/dataset/hand/arctic_seqs/splits/test/{cfg.split_test}.npy"
            )

        # view_list  = cfg.train_views if self.split == 'train' else cfg.test_views
        #
        # for view_name in view_list:
        #     data = np.load(f"/mnt/sda2/lxy/dataset/hand/arctic_seqs/splits/NR_seqs/{view_name}.npy", allow_pickle=True).item()
        #     self.data.update(data["data_dict"])
        #     self.imgnames.update({view_name : data["imgnames"]})

        data = np.load(data_p, allow_pickle=True).item()
        self.data = data["data_dict"]
        imgnames = data["imgnames"]

        with open("/mnt/sda2/lxy/dataset/hand/arctic/meta/misc.json", "r") as f:
            misc = json.load(f)

        # unpack
        subjects = list(misc.keys())
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam
        self.intris_mat = intris_mat
        self.image_sizes = image_sizes
        self.ioi_offset = ioi_offset

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.egocam_k = None

        frames = []
        self.imgnames = []
        for imgname in imgnames:
            # 解析路径
            sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
            # 减去偏移得到真实帧索引
            vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
            if vidx % 2 == 1:
                frames.append(vidx)
                self.imgnames.append(imgname)

        #frames = list(range(len(self.imgnames)))

        self.frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }


    def get_metadata(self, hand_side='right'):

        cano_data = self.get_cano_mano_verts(hand_side)
        if self.split != 'train':
            self.metadata[hand_side] = cano_data
            return

        self.metadata[hand_side]={
            'faces': self.faces[hand_side],
            'posedirs': self.posedirs[hand_side+'Hand'],
            'J_regressor': self.J_regressor[hand_side+'Hand'],
            'cameras_extent': 1.0395,
            'frame_dict': self.frame_dict,
        }
        self.metadata[hand_side].update(cano_data)


    def get_obj_data(self):
        self.metadata_obj = {}
        for obj_name in self.cfg._YCB_CLASSES:

            path = f"/mnt/sda2/lxy/dataset/hand/arctic/meta/object_vtemplates/{obj_name}/mesh.obj"

            # 加载 mesh
            mesh = trimesh.load_mesh(path, process=False)
            verts = np.asarray(mesh.vertices)/1000
            faces = np.asarray(mesh.faces)

            # 构建角点 AABB
            aabb_min = verts.min(axis=0)
            aabb_max = verts.max(axis=0)

            if obj_name == 'phone':
                aabb_max[0] = 2 * aabb_max[0] - aabb_min[0]

            obj_aabb = AABB(aabb_min, aabb_max)

            # 8个角点（AABB）作为 obj3DCorners
            obj3DCorners = np.array([
                [aabb_min[0], aabb_min[1], aabb_min[2]],
                [aabb_min[0], aabb_min[1], aabb_max[2]],
                [aabb_min[0], aabb_max[1], aabb_min[2]],
                [aabb_min[0], aabb_max[1], aabb_max[2]],
                [aabb_max[0], aabb_min[1], aabb_min[2]],
                [aabb_max[0], aabb_min[1], aabb_max[2]],
                [aabb_max[0], aabb_max[1], aabb_min[2]],
                [aabb_max[0], aabb_max[1], aabb_max[2]],
            ], dtype=np.float32)

            self.metadata_obj[obj_name] = {
                'obj3DCorners': obj3DCorners,  # 8×3
                'obj_aabb': obj_aabb,
                'obj_triangles': faces,
                'obj_points': verts,
                'frame_dict': self.frame_dict
                #'obj_mesh': mesh
            }


    def get_cano_mano_verts(self, hand_side='right'):
        # compute scale from Mano
        if hand_side=='right':
            body = self.body_model_r()
        else:
            body = self.body_model_l()
        # 3D models and points
        minimal_shape = body['v'][0].detach().numpy()
        #print('minimal_shape',minimal_shape[0])
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[hand_side+'Hand']
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[hand_side+'Hand']

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

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces[hand_side])

        return {
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights,
            'bone_transforms': bone_transforms,
            'cano_mesh': cano_mesh,
            'faces': self.faces[hand_side],
            'coord_max': coord_max,
            'coord_min': coord_min,
            'aabb': AABB(coord_min, coord_max),
        }


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

    def generate_patch_image(self, cvimg, bbox, input_shape, do_flip=False, scale=1, rot=0):
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

        bb_c_x = float(bbox[0])
        bb_c_y = float(bbox[1])
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
        return len(self.imgnames)

    def __getitem__(self, idx):
        imgname = self.imgnames[idx]
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]
        data_cam = seq_data["cam_coord"]
        data_2d = seq_data["2d"]
        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]
        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        vidx, is_valid, right_valid, left_valid = get_valid(
            data_2d, data_cam, vidx, view_idx, imgname
        )

        if view_idx == 0:
            K = data_params["K_ego"][vidx].copy()
        else:
            K = np.array(self.intris_mat[sid][view_idx - 1])

        # hands
        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()

        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

        pose_r = data_params["pose_r"][vidx].copy()
        trans_r = data_params["trans_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        rot_r = data_params["rot_r"][vidx].copy()

        pose_l = data_params["pose_l"][vidx].copy()
        trans_l = data_params["trans_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()
        rot_l = data_params["rot_l"][vidx].copy()

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()

        # objects
        bbox2d = pad_jts2d(data_2d["bbox3d"][vidx, view_idx].copy())
        bbox3d = data_cam["bbox3d"][vidx, view_idx].copy()
        bbox2d_t = bbox2d[:8]
        bbox2d_b = bbox2d[8:]
        bbox3d_t = bbox3d[:8]
        bbox3d_b = bbox3d[8:]

        kp2d = pad_jts2d(data_2d["kp3d"][vidx, view_idx].copy())
        kp3d = data_cam["kp3d"][vidx, view_idx].copy()
        kp2d_t = kp2d[:16]
        kp2d_b = kp2d[16:]
        kp3d_t = kp3d[:16]
        kp3d_b = kp3d[16:]

        obj_radian = data_params["obj_arti"][vidx].copy()
        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        # scale and center in the original image space

        bbox = data_bbox[vidx, view_idx]  # original bbox
        intrx = update_K_after_bbox_crop_resize(K, bbox, cap_dim=1000, crop_ratio=1.5)


        is_egocam = "/0/" in imgname

        (
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
        ) = transform_2d_for_speedup(
            self.cfg.speedup,
            is_egocam,
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
            self.cfg.ego_image_scale,
        )
        imgname = imgname.replace('./arctic_data/data/images/', 'cropped_images/')
        img_name = os.path.join(self.root_dir, imgname)
        seg_path = img_name.replace('arctic/cropped_images/', 'arctic_seqs/seg/').replace('jpg', 'png')
        part_seg_path = seg_path.replace("seg","masks").replace("png","npy")

        image, img_status = read_img(img_name, (1000, 1000, 3))
        mask = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)
        

        if self.split == 'train':
            R = np.eye(3).astype(np.float32)
            T = np.zeros(3).astype(np.float32)

        else:
            world2cam1 =  np.array(self.world2cam[sid][self.train_view_id - 1])
            world2cam = np.array(self.world2cam[sid][view_idx - 1])
            # cam2相对于cam1的外参
            cam2_from_cam1 = world2cam @ np.linalg.inv(world2cam1)
            cam1_from_cam2 = world2cam1 @ np.linalg.inv(world2cam)

            # 提取 R 和 t
            R = cam1_from_cam2[:3, :3]
            T = cam2_from_cam1[:3, 3]


        color_jitting = False
        color_factor = 0.3
        if color_jitting:
            if self.split == 'train':
                c_up = 1.0 + color_factor
                c_low = 1.0 - color_factor
                color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
            else:
                # pre defined color_jitting for testset
                color_scale = self.color_scales[idx].astype(np.float32)
            for i in range(3):
               image[:, :, i] = np.clip(image[:, :, i] * color_scale[i], 0, 255)


        #image = cv2.resize(image, (self.w, self.h))

        obj_image = image.copy()
        full_image = image.copy()
        hand_image = image.copy()

        #trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]
        scale, rot, do_flip, color_scale = 1, 0, False, [1.0, 1.0, 1.0]

        # bbox[0] = bbox[0] + bbox[2] * trans[0]
        # bbox[1] = bbox[1] + bbox[3] * trans[1]
        roi_size = self.cfg.get('roi_size', 224)

        img_ROI, trans_img2roi = self.generate_patch_image(full_image, [bbox[0],bbox[1],200.0 *bbox[2],200.0 *bbox[2]], [roi_size, roi_size], do_flip, scale, rot)


        # cv2.imwrite("/home/cyc/pycharm/lxy/visual/img_ROI.png",cv2.cvtColor(np.uint8(img_ROI), cv2.COLOR_BGR2RGB))
        mask_r = (mask == [255, 0, 0]).all(axis=-1)
        mask_l = (mask == [0, 255, 0]).all(axis=-1)
        mask_o = (mask == [0, 0, 255]).all(axis=-1)

        hand_mask = mask_r | mask_l
        obj_mask = mask_o
        full_mask = hand_mask | obj_mask

        mask_hand = np.logical_or(mask_r, mask_l)
        mask_obj = mask_o
        mask_full = np.logical_or(mask_hand, mask_obj)

        img_ROI = img_ROI / 255.
        img_ROI = torch.from_numpy(img_ROI).permute(2, 0, 1).float()

        hand_image[hand_mask == 0] = 255. if self.white_bg else 0.
        hand_image = hand_image / 255.
        hand_image = torch.from_numpy(hand_image).permute(2, 0, 1).float()

        obj_image[obj_mask == 0] = 255. if self.white_bg else 0.
        obj_image = obj_image / 255.
        obj_image = torch.from_numpy(obj_image).permute(2, 0, 1).float()

        full_image_ori = full_image.copy()
        full_image_ori = full_image_ori/255.
        full_image_ori = torch.from_numpy(full_image_ori).permute(2, 0, 1).float()

        full_image[full_mask == 0] = 255. if self.white_bg else 0.
        full_image = full_image / 255.
        full_image = torch.from_numpy(full_image).permute(2, 0, 1).float()

        full_mask = torch.from_numpy(full_mask.astype(np.float32)).unsqueeze(0).float()
        hand_mask = torch.from_numpy(hand_mask.astype(np.float32)).unsqueeze(0).float()
        obj_mask = torch.from_numpy(obj_mask.astype(np.float32)).unsqueeze(0).float()

        if self.split == 'train':
            part_mask = np.load(part_seg_path)
            mask_static = (part_mask == 1)
            mask_dynamic = (part_mask == 2)
            mask_static = torch.from_numpy(mask_static.astype(np.float32)).float()
            mask_dynamic = torch.from_numpy(mask_dynamic.astype(np.float32)).float()
        else:
            mask_static = torch.zeros_like(obj_mask).squeeze(0)
            mask_dynamic = torch.zeros_like(obj_mask).squeeze(0)


        focal_length_x = intrx[0, 0]
        focal_length_y = intrx[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        #obj_idx = self.obj_names.index(obj_name)
        #kp3d_cano = self.kp3d_cano[obj_idx] / 1000  # meter
        # kp3d_cano = kp3d_cano.numpy()
        # kp3d_target = kp3d_b[:, :3].float()

        # # rotate canonical kp3d to match original image
        # R, obj_trans = tf.solve_rigid_tf_np(kp3d_cano, kp3d_target)
        # obj_rot = (
        #     rot.batch_rot2aa(torch.from_numpy(R).float().view(1, 3, 3)).view(3).numpy()
        # )
        obj_rot = data_params["obj_rot"][vidx].copy()
        obj_trans = data_params["obj_trans"][vidx].copy()
        if view_idx != 0:
            world2cam = np.array(self.world2cam[sid][view_idx - 1])
            obj_rot, obj_trans = apply_w2c_pose_numpy(obj_rot, obj_trans / 1000, world2cam)
            #rot_r, trans_r = apply_w2c_pose_numpy(rot_r, trans_r, world2cam)
            #rot_l, trans_l = apply_w2c_pose_numpy(rot_l, trans_l, world2cam)
        # if self.split == 'train':
        #     print('train:', obj_trans)

        obj_rot, _ = cv2.Rodrigues(obj_rot)

        body_r = self.body_model_r(betas=torch.from_numpy(betas_r).float().reshape(-1, 10))
        minimal_shape_r = body_r['v'][0].detach().cpu().numpy()
        body_l = self.body_model_l(betas=torch.from_numpy(betas_l).float().reshape(-1, 10))
        minimal_shape_l = body_l['v'][0].detach().cpu().numpy()

        body_r_world = self.body_model_r(global_orient=torch.from_numpy(rot_r).float().reshape(-1, 3),
                                   hand_pose=torch.from_numpy(pose_r).float().reshape(-1, 45),
                                   betas=torch.from_numpy(betas_r).float().reshape(-1, 10),
                                   trans_l=torch.from_numpy(trans_r).float().reshape(-1, 3),
                                   )

        body_l_world = self.body_model_l(global_orient=torch.from_numpy(rot_l).float().reshape(-1, 3),
                                   hand_pose=torch.from_numpy(pose_l).float().reshape(-1, 45),
                                   betas=torch.from_numpy(betas_l).float().reshape(-1, 10),
                                   trans_l=torch.from_numpy(trans_l).float().reshape(-1, 3),
                                   )

        bone_transforms_r, Jtr_r, Jtr_norm_r, pose_rot_r = compute_posed_mano_hand(body_r_world, minimal_shape_r, rot_r, pose_r, trans_r, world2cam)
        bone_transforms_l, Jtr_l, Jtr_norm_l, pose_rot_l = compute_posed_mano_hand(body_l_world, minimal_shape_l, rot_l, pose_l, trans_l, world2cam)

        # obj_rot = data_params["obj_rot"][vidx].copy()
        # obj_rot, _ = cv2.Rodrigues(obj_rot)
        #obj_trans = data_params["obj_trans"][vidx].copy()/1000

        #world2cam = np.array(self.world2cam[sid][view_idx - 1]).astype(dtype=np.float32)
        #cam2world = np.linalg.inv(world2cam)

        # 不行
        # R = cam2world[:3, :3]
        # T = world2cam[:3, 3]

        # 可以
        # T = world2cam[:3, 3]
        # obj_trans = world2cam[:3, :3] @ obj_trans
        # print(obj_trans)



        camera = Camera(
            offaxis=True,
            frame_id=int(vidx),
            cam_id=view_idx,
            subject_id=sid,
            obj_id=obj_name,
            K=intrx, R=R, T=np.squeeze(T),
            bbox=torch.from_numpy(bbox),
            FoVx=FovX,
            FoVy=FovY,
            image=hand_image,
            mask=hand_mask,
            obj_image=obj_image,
            obj_mask=obj_mask,
            mask_static=mask_static,
            mask_dynamic=mask_dynamic,
            full_image=full_image,
            full_image_ori=full_image_ori,
            full_mask=full_mask,
            img_ROI=img_ROI,
            trans_img2roi=trans_img2roi,
            #camera_center=torch.from_numpy(cam_center),
            image_name=imgname.replace('cropped_images/', ''),
            data_device=self.cfg.data_device,
            # human params
            rots_r=torch.from_numpy(pose_rot_r).float(),
            Jtrs_r=torch.from_numpy(Jtr_norm_r).float(),
            bone_transforms_r=torch.from_numpy(bone_transforms_r),
            rots_l=torch.from_numpy(pose_rot_l).float(),
            Jtrs_l=torch.from_numpy(Jtr_norm_l).float(),
            bone_transforms_l=torch.from_numpy(bone_transforms_l),
            # obj params
            obj_rots=torch.from_numpy(obj_rot).float().view(3,3),
            obj_trans=torch.from_numpy(obj_trans).float().view(3),
            Jtrs_l_3d=torch.from_numpy(Jtr_l).float(),
            Jtrs_r_3d=torch.from_numpy(Jtr_r).float(),
            # hand_root_r=torch.from_numpy(joints3d_r[0]).float(),
            # hand_root_l=torch.from_numpy(joints3d_l[0]).float()
        )
        return camera


    # def __getitem__(self, idx):
    #     if self.preload:
    #         return self.cameras[idx]
    #     else:
    #         return self.getitem(idx)

    def readPointCloud(self, sub_id, mano_side='right'):

        ply_path = os.path.join(self.root_dir.replace('arctic', 'arctic_seqs/'),'canonical', 'cano_mano_{}_{}.ply'.format(sub_id, mano_side))
        try:
            pcd = fetchPly(ply_path)
        except:
            verts = self.metadata[mano_side]['smpl_verts']
            faces = self.faces[mano_side]
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            n_points = 5000

            xyz = mesh.sample(n_points)
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)

        return pcd

    def randomPointCloud(self, obj_id):
        ply_path = os.path.join(self.root_dir.replace('arctic', 'arctic_seqs/'),'canonical','random_pc_obj_{}.ply'.format(obj_id))
        n_points = 100

        objcoords = self.metadata_obj[obj_id]['obj3DCorners']

        coord_min = objcoords.min(axis=0)
        coord_max = objcoords.max(axis=0)
        xyz = np.random.uniform(coord_min, coord_max, (n_points, 3))
        rgb = np.ones_like(xyz) * 255
        storePly(ply_path, xyz, rgb)

        pcd = fetchPly(ply_path)

        return pcd


def compute_posed_mano_hand(body, minimal_shape, rot, pose, trans, w2c):
    # compute posed mano hand


    Jtr = body['Jtr'][0].detach().numpy()
    bone_transforms = body['bone_transforms'][0].detach().numpy()

    Jtr = (w2c[:3, :3] @ Jtr.T).T + w2c[:3, 3]

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

    bone_transforms1 = np.repeat(np.eye(4)[np.newaxis, ...], 16, axis=0)
    bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms1)
    bone_transforms = bone_transforms.astype(np.float32)
    bone_transforms[:, :3, 3] += trans

    pose6d = np.concatenate([rot, pose], axis=-1)
    pose6d = Rotation.from_rotvec(pose6d.reshape([-1, 3]))
    pose_mat_full = pose6d.as_matrix()
    pose_mat = pose_mat_full[1:, ...].copy()
    pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
        [-1, 9])

    bone_transforms = w2c @ bone_transforms

    return bone_transforms.astype(np.float32), Jtr, Jtr_norm, pose_rot


def construct_hand_meshes(cam_data, layers, view_idx, distort):
    if view_idx == 0 and distort:
        view_idx = 9
    v3d_r = cam_data["verts.right"][:, view_idx]
    v3d_l = cam_data["verts.left"][:, view_idx]

    right = {
        "v3d": v3d_r,
        "f3d": layers["right"].faces,
        "vc": None,
        "name": "right",
        "color": "white",
    }
    left = {
        "v3d": v3d_l,
        "f3d": layers["left"].faces,
        "vc": None,
        "name": "left",
        "color": "white",
    }
    return right, left


def construct_object_meshes(cam_data, obj_name, layers, view_idx, distort):
    if view_idx == 0 and distort:
        view_idx = 9
    v3d_o = cam_data["verts.object"][:, view_idx]
    f3d_o = Mesh(
        filename=f"/mnt/sda2/lxy/arctic/meta/object_vtemplates/{obj_name}/mesh.obj"
    ).faces

    obj = {
        "v3d": v3d_o,
        "f3d": f3d_o,
        "vc": None,
        "name": "object",
        "color": "light-blue",
    }
    return obj

def construct_smplx_meshes(cam_data, layers, view_idx, distort):
    assert not distort, "Distortion rendering not supported for SMPL-X"
    # We use the following algorithm to render meshes with distortion effects:
    # VR Distortion Correction Using Vertex Displacement
    # https://stackoverflow.com/questions/44489686/camera-lens-distortion-in-opengl
    # However, this method creates artifacts when vertices are too close to the camera.

    if view_idx == 0 and distort:
        view_idx = 9

    v3d_s = cam_data["verts.smplx"][:, view_idx]

    smplx_mesh = {
        "v3d": v3d_s,
        "f3d": layers["smplx"].faces,
        "vc": None,
        "name": "smplx",
        "color": "rice",
    }

    return smplx_mesh

