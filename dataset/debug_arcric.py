import os

import cv2
import common.transforms as tf
from common.data_utils import read_img
from common.mesh import Mesh
from right_hand_model import MANO
from utils.graphics_utils import focal2fov
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB, get_valid, pad_jts2d, \
    transform_2d_for_speedup, apply_w2c_pose_numpy, update_K_after_bbox_crop_resize, load_K_Rt_from_P
from scene.cameras import Camera

import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh


class RigidArcticDataset(Dataset):
    def __init__(self):
        super().__init__()
        
        self.root_dir = '/home/cyc/pycharm/lxy/hold/unpack/arctic_data/arctic_s03_box_grab_01_1/'
        self._load_data()
        #self.root_dir = os.path.join(cfg.root_dir, split)

        self.SEGM_IDS = {"bg": 0, "object": 50, "right": 150, "left": 250}
        self.white_bg = True

        self.w, self.h = 1384, 1277


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

    def _load_data(self):
        f_num = 300
        self.data = {}
        data_p = os.path.join(self.root_dir, "build/image")
        self.imgnames = [os.path.join(data_p, f) for f in os.listdir(data_p)]

        np_data = np.load(os.path.join(self.root_dir,"build/data.npy"), allow_pickle=True).item()

        camera_data = np_data['cameras']
        cached_data = np_data['entities']

        hand = ['left', 'right']

        # load hand/object poses
        for side in hand:
            self.data[side] = {
                'global_orient': cached_data[side]['hand_poses'][:, :3],
                'hand_pose': cached_data[side]['hand_poses'][:, 3:],
                'betas': cached_data[side]['mean_shape'][None, :].repeat(f_num, 0),
                'transl': cached_data[side]['hand_trans'],
                'scale': np.ones([1, 1]).repeat(f_num, 0),
            }
        self.data['object'] = {
            'global_orient': cached_data['object']['object_poses'][:, :3],
            'hand_pose': None,
            'betas': None,
            'transl': cached_data['object']['object_poses'][:, 3:],
            'scale': np.ones([1, 1]).repeat(f_num, 0),
        }

        # load cam params
        self.projection_mat = (camera_data['world_mat_0'] @ camera_data['scale_mat_0'])
        P = self.projection_mat[:3, :4]
        intrinsics, extrinsics = load_K_Rt_from_P(None, P)
        self.intris_mat = intrinsics
        self.world2cam = extrinsics

        # set extrinsics
        #self.world2cam = np.eye(4)

        frames = list(range(f_num))
        self.frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        self.image_sizes = [1384,1277]
        self.obj_name = 'box'

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
            center = (aabb_min+aabb_max)/2

            #verts = verts-center

            aabb_min = aabb_min-center
            aabb_max = aabb_max - center

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
        bbox[0] = c_x
        bbox[1] = c_y

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

    def get(self, idx=0):
        img_name = self.imgnames[idx]

        intris_mat = np.array(self.intris_mat)

        pose_r = self.data['right']["hand_pose"][idx].copy()
        trans_r = self.data['right']["transl"][idx].copy()

        betas_r = self.data['right']["betas"][idx].copy()
        rot_r = self.data['right']["global_orient"][idx].copy()

        pose_l = self.data['left']["hand_pose"][idx].copy()
        trans_l = self.data['left']["transl"][idx].copy()
        betas_l = self.data['left']["betas"][idx].copy()
        rot_l = self.data['left']["global_orient"][idx].copy()
        # print()
        # print(self.data['right']["transl"][idx].copy())
        # print(self.data['left']["transl"][idx].copy())
        # print(self.data['object']["transl"][idx].copy())

        image_size = self.image_sizes
        image_size = {"width": image_size[0], "height": image_size[1]}

        # scale and center in the original image space
        seg_path = img_name.replace('image', 'mask')
        #print(seg_path)
        # imgname = imgname.replace("/arctic_data/", "/data/arctic_data/")
        image, img_status = read_img(img_name, (image_size['width'], image_size['height'], 3))
        mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)


        R = np.eye(3).astype(np.float32)
        #R[0, 0] = -1
        #R[1, 1] = -1
        # R[2, 2] = 1

        T = np.zeros(3).astype(np.float32)


        obj_image = image.copy()
        full_image = image.copy()
        hand_image = image.copy()

        #trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]
        scale, rot, do_flip, color_scale = 1, 0, False, [1.0, 1.0, 1.0]

        # bbox[0] = bbox[0] + bbox[2] * trans[0]
        # bbox[1] = bbox[1] + bbox[3] * trans[1]
        roi_size = self.cfg.get('roi_size', 224)

        # cv2.imwrite("/home/cyc/pycharm/lxy/visual/img_ROI.png",cv2.cvtColor(np.uint8(img_ROI), cv2.COLOR_BGR2RGB))
        mask_r = (mask == self.SEGM_IDS['right'])
        mask_l = (mask == self.SEGM_IDS['left'])
        mask_o = (mask == self.SEGM_IDS['object'])

        hand_mask = mask_r | mask_l
        obj_mask = mask_o
        full_mask = hand_mask | obj_mask

        mask_hand = np.logical_or(mask_r, mask_l)
        mask_obj = mask_o
        mask_full = np.logical_or(mask_hand, mask_obj)

        min_y, max_y = np.where(obj_mask)[0].min(), np.where(obj_mask)[0].max() + 1
        min_x, max_x = np.where(obj_mask)[1].min(), np.where(obj_mask)[1].max() + 1
        c_x = int((max_x + min_x) / 2)
        c_y = int((max_y + min_y) / 2)
        bbox_delta_x = (max_x - min_x) / 2
        bbox_delta_y = (max_y - min_y) / 2
        bbox_delta = max(bbox_delta_x, bbox_delta_y)
        bbox = [c_x - bbox_delta, c_y - bbox_delta, bbox_delta * 2, bbox_delta * 2]
        bbox = self.process_bbox(bbox, self.w, self.h, 1.5)

        img_ROI, trans_img2roi = self.generate_patch_image(full_image, [bbox[0], bbox[1], bbox[2], bbox[2]],
                                                           [roi_size, roi_size], do_flip, scale, rot)

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


        focal_length_x = intris_mat[0, 0]
        focal_length_y = intris_mat[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)


        obj_rot = self.data['object']["global_orient"][idx].copy()
        obj_trans = self.data['object']["transl"][idx].copy()

        #obj_rot, obj_trans = apply_w2c_pose_numpy(obj_rot, obj_trans, self.world2cam)

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

        bone_transforms_r, Jtr_norm_r, pose_rot_r = compute_posed_mano_hand(body_r_world, minimal_shape_r, rot_r, pose_r, trans_r, self.world2cam)
        bone_transforms_l, Jtr_norm_l, pose_rot_l = compute_posed_mano_hand(body_l_world, minimal_shape_l, rot_l, pose_l, trans_l, self.world2cam)

        print(obj_trans)
        print(bone_transforms_r[0])
        print(bone_transforms_l[0])
        print()

        camera = Camera(
            offaxis=False,
            frame_id=int(idx),
            cam_id=0,
            subject_id='s03',
            obj_id=self.obj_name,
            K=intris_mat, R=R, T=np.squeeze(T),
            bbox=torch.from_numpy(bbox),
            FoVx=FovX,
            FoVy=FovY,
            image=hand_image,
            mask=hand_mask,
            obj_image=obj_image,
            obj_mask=obj_mask,
            full_image=full_image,
            full_image_ori=full_image_ori,
            full_mask=full_mask,
            img_ROI=img_ROI,
            trans_img2roi=trans_img2roi,

            image_name=img_name.split('/')[-1],
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

        ply_path = os.path.join(self.root_dir,'canonical', 'cano_mano_{}_{}.ply'.format(sub_id, mano_side))
        try:
            pcd = fetchPly(ply_path)
        except:
            verts = self.metadata[mano_side]['smpl_verts']
            faces = self.faces[mano_side]
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            n_points = 5000

            xyz = mesh.sample(n_points)
            rgb = np.ones_like(xyz) * 255
            os.makedirs(os.path.join(self.root_dir, 'canonical'), exist_ok=True)
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)

        return pcd

    def randomPointCloud(self, obj_id):
        ply_path = os.path.join(self.root_dir,'canonical','random_pc_obj_{}.ply'.format(obj_id))
        n_points = 5000

        objcoords = self.metadata_obj[obj_id]['obj3DCorners']
        coord_min = objcoords.min(axis=0)
        coord_max = objcoords.max(axis=0)
        xyz = np.random.uniform(coord_min, coord_max, (n_points, 3))
        rgb = np.ones_like(xyz) * 255
        os.makedirs(os.path.join(self.root_dir,'canonical'),exist_ok=True)
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

    return bone_transforms.astype(np.float32), Jtr_norm, pose_rot



if __name__=='__main__':
    dataset = RigidArcticDataset()
    cam = dataset.get(0)
    # debug
    # ====== 1. 生成右手 mesh ======
    verts_r = cam.bone_transforms_r.numpy()[:, :3, 3]  # 这里可以用 body_model_r 重新forward得到verts
    faces_r = dataset.faces['right']
    hand_mesh_r = trimesh.Trimesh(vertices=verts_r, faces=faces_r)
    hand_mesh_r.export(os.path.join(save_dir, "right_hand.obj"))

    # ====== 2. 生成左手 mesh ======
    verts_l = cam.bone_transforms_l.numpy()[:, :3, 3]
    faces_l = dataset.faces['left']
    hand_mesh_l = trimesh.Trimesh(vertices=verts_l, faces=faces_l)
    hand_mesh_l.export(os.path.join(save_dir, "left_hand.obj"))

    # ====== 3. 生成物体 mesh ======
    obj_verts = dataset.metadata_obj[dataset.obj_name]['obj_points']
    obj_faces = dataset.metadata_obj[dataset.obj_name]['obj_triangles']
    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
    obj_mesh.export(os.path.join(save_dir, "object.obj"))

    # ====== 4. 投影到2D并保存图像 ======
    K = cam.K
    R = cam.R
    T = cam.T.reshape(3, 1)

    def project(verts, K, R, T):
        verts_cam = (R @ verts.T + T).T
        verts_proj = (K @ verts_cam.T).T
        verts_proj = verts_proj[:, :2] / verts_proj[:, 2:]
        return verts_proj

    img = (cam.full_image_ori.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    for mesh, color in [(hand_mesh_r, (0,0,255)), (hand_mesh_l, (0,255,0)), (obj_mesh, (255,0,0))]:
        verts = np.array(mesh.vertices)
        pts_2d = project(verts, K, R, T).astype(int)
        for p in pts_2d:
            if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                cv2.circle(img, tuple(p), 1, color, -1)

    cv2.imwrite(os.path.join(save_dir, "proj_debug.png"), img)
