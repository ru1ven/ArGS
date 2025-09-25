import numpy as np
import torch

from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
from pytorch3d.renderer import look_at_view_transform


class QueryCamera:
    def __init__(self, camera=None, **kwargs):
        if camera is not None:
            self.data = camera.data.copy()
            return

        self.data = kwargs
        self.data['trans'] = np.array([0.0, 0.0, 0.0])
        self.data['scale'] = 1.0
        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01
        self.data['world_view_transform'] = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                             fovY=self.FoVy).transpose(0, 1).cuda()
        self.data['full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).cuda()
        self.data['camera_center'] = self.world_view_transform.inverse()[3, :3].cuda()
        self.data['R'] = torch.tensor(self.R).cuda()
        self.data['T'] = torch.tensor(self.T).cuda()
        self.data['K'] = torch.tensor(self.K).cuda()


    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = QueryCamera(camera=self)
        return new_cam


class QueryCamerasLoader:
    def __init__(self, aabb, cam_num=64):
        coord_min = aabb.coord_min
        coord_max = aabb.coord_max
        self.cam_num = cam_num
        self._cam = []
        self.K = np.array([[615.7017822265625, 0, 321.1631774902344], [0, 614.4563598632812, 239.0236053466797], [0, 0, 1]])
        self.h, self.w = 480, 640
        focal_length_x = self.K[0][0]
        focal_length_y = self.K[1][1]
        self.FoVy = focal2fov(focal_length_y, self.h)
        self.FoVx = focal2fov(focal_length_x, self.w)
        extrinsics = self.get_camera_extrinsics(coord_min, coord_max, cam_num)
        for R, T in extrinsics:
            camera = QueryCamera(K=self.K, R=np.array(R), T=np.array(T), focal_x=focal_length_x, focal_y=focal_length_y,
                                 FoVx=self.FoVx, FoVy=self.FoVy,image_height=self.h, image_width=self.w)
            self._cam.append(camera)

    @property
    def get_cam(self):
        return self._cam

    def generate_fibonacci_sphere(self, n_points, radius, center_tensor):
        """生成均匀分布在球面上的点"""
        device = center_tensor.device
        # 黄金比
        phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2  # golden ratio
        # 使用torch张量生成点
        z = torch.linspace(-1, 1, n_points, device=device)  # 映射到 [-1, 1]
        theta = 2 * torch.pi * torch.arange(n_points, device=device) / phi
        # 计算x, y, z
        x = torch.sqrt(1 - z ** 2) * torch.cos(theta)
        y = torch.sqrt(1 - z ** 2) * torch.sin(theta)
        # 将坐标移到指定的中心并缩放到给定的半径
        x = x * radius + center_tensor[0]
        y = y * radius + center_tensor[1]
        z = z * radius + center_tensor[2]
        # 将x, y, z 合并为一个点，并将所有点添加到列表中
        points = torch.stack((x, y, z), dim=-1)
        return points

    def compute_camera_extrinsics(self, camera_pos, bbox_center):
        """计算每个相机的外参，包括旋转矩阵和平移向量，支持CUDA加速，但输出为list和numpy格式"""

        rotation_matrix, translation_vector = look_at_view_transform(eye=camera_pos.unsqueeze(0), at=bbox_center.unsqueeze(0))

        # 转换为numpy并返回
        rotation_matrix = rotation_matrix.squeeze(0).cpu().numpy()  # 将tensor转换为numpy数组
        translation_vector = translation_vector.squeeze(0).cpu().numpy()  # 转换为numpy数组

        return rotation_matrix, translation_vector

    def get_radius(self, r, extension=1.1):
        """计算最小半径"""
        # BBox 尺寸加上扩展系数
        r = r * extension

        # 距离计算：取水平和垂直的最大约束
        d_h = r / np.sin(self.FoVx / 2)  # 根据水平视角计算距离
        d_v = r / np.sin(self.FoVy / 2)  # 根据垂直视角计算距离

        return max(d_h, d_v)

    def get_camera_extrinsics(self, c_min, c_max, n_cameras=64, extension=1.25):
        """
        输入 BBox 的最小点、最大点、相机数量和扩展系数，生成所有相机的外参
        """
        # 计算 BBox 的中心和尺寸
        bbox_center = (c_min + c_max) / 2
        r = torch.norm(c_max - c_min) / 2.0

        # 计算最小安全距离
        radius = self.get_radius(r, extension)

        # 生成球面上均匀分布的相机位置
        camera_positions = self.generate_fibonacci_sphere(n_cameras, radius, bbox_center)

        # 计算每个相机的外参
        extrinsics = []
        path = '/home/cyc/pycharm/lxy/3DGS/debug/mesh/camera.obj'
        with open(path, 'w') as fp:
            for cam_pos in camera_positions:
                v = cam_pos.detach().cpu().numpy()
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

                R, t = self.compute_camera_extrinsics(cam_pos, bbox_center)
                extrinsics.append((R, t))
        return extrinsics