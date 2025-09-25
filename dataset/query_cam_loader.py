import numpy as np
import torch

from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix


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
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.data['camera_center'] = self.world_view_transform.inverse()[3, :3]
        self.data['R'] = torch.tensor(self.R).to(self.data_device)
        self.data['T'] = torch.tensor(self.T).to(self.data_device)
        self.data['K'] = torch.tensor(self.K).to(self.data_device)

    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = QueryCamera(camera=self)
        return new_cam


class QueryCamerasLoader:
    def __init__(self, aabb):
        coord_min = aabb.coord_min
        coord_max = aabb.coord_max
        self._cam = []
        self.K = [[615.7017822265625, 0, 321.1631774902344], [0, 614.4563598632812, 239.0236053466797], [0, 0, 1]]
        self.h, self.w = 480, 640
        focal_length_x = self.K[0, 0]
        focal_length_y = self.K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)
        extrinsics = self.get_camera_extrinsics(coord_min, coord_max)
        for R, T in extrinsics:
            camera = QueryCamera(K=self.K, R=R, T=T, FoVx=FovX, FoVy=FovY)
            self._cam.append(camera)

    @property
    def get_cam(self):
        return self._cam

    def generate_fibonacci_sphere(self, n_points, radius=1.0, center=(0, 0, 0)):
        """生成均匀分布在球面上的点"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n_points):
            z = -1 + (2 * i) / (n_points - 1)  # 映射到 [-1, 1]
            theta = 2 * np.pi * i / phi
            x = np.sqrt(1 - z ** 2) * np.cos(theta)
            y = np.sqrt(1 - z ** 2) * np.sin(theta)
            points.append((x * radius + center[0],
                           y * radius + center[1],
                           z * radius + center[2]))
        return np.array(points)

    def compute_camera_extrinsics(self, camera_pos, bbox_center):
        """计算每个相机的外参，包括旋转矩阵和平移向量"""
        direction = bbox_center - camera_pos
        direction = direction / np.linalg.norm(direction)  # 归一化方向向量
        up = np.array([0, 1, 0])
        # 检查 up 是否与方向平行，避免奇异情况
        if np.allclose(direction, up) or np.allclose(direction, -up):
            up = np.array([0, 0, 1])  # 调整 up 为另一个向量
        x_axis = np.cross(up, direction)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(direction, x_axis)
        rotation_matrix = np.stack((x_axis, y_axis, direction), axis=-1)
        translation_vector = camera_pos
        return rotation_matrix, translation_vector

    def get_radius(self, bbox_size, extension=1.25):
        """计算最小半径"""
        f_x, f_y = self.K[0, 0], self.K[1, 1]  # 提取焦距
        c_x, c_y = self.K[0, 2], self.K[1, 2]  # 提取主点

        # BBox 尺寸加上扩展系数
        bbox_size = bbox_size * extension
        x_max = bbox_size[0] / 2
        y_max = bbox_size[1] / 2

        # 计算安全距离
        Z_min_x = x_max * f_x / c_x
        Z_min_y = y_max * f_y / c_y
        return max(Z_min_x, Z_min_y)

    def get_camera_extrinsics(self, c_min, c_max, n_cameras=64, extension=1.25):
        """
        输入 BBox 的最小点、最大点、相机数量和扩展系数，生成所有相机的外参
        """
        # 计算 BBox 的中心和尺寸
        bbox_center = (c_min + c_max) / 2
        bbox_size = c_max - c_min

        # 计算最小安全距离
        radius = self.get_radius(bbox_size, extension)

        # 生成球面上均匀分布的相机位置
        camera_positions = self.generate_fibonacci_sphere(n_cameras, radius=radius, center=bbox_center)

        # 计算每个相机的外参
        extrinsics = []
        for cam_pos in camera_positions:
            R, t = self.compute_camera_extrinsics(cam_pos, bbox_center)
            extrinsics.append((R, t))
        return extrinsics