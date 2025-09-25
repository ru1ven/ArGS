import torch
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes, Pointclouds
# 初始化相机。
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras, OpenGLPerspectiveCameras
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)

T = torch.tensor([[-4.228051957326409149e-01, -2.078562115810418665e-01, -8.820609739517225600e-01, 4.637252480950287414e-01],
                                [9.057441941962964815e-01, -6.536944939351106709e-02, -4.187532564239835331e-01, 1.535094236870509776e-01],
                                [2.938062526878559150e-02, -9.759726586299259932e-01, 2.159030997129234852e-01, -1.448621926159473529e-02],
                                [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
T = torch.tensor([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
device = torch.device("cpu")

# 加载 PLY 文件。
#verts  = load_ply("/home/pfren/3dgsAvatar/HO3D/ABF1/cano_mano.ply")
model_dict = np.load("/home/pfren/3dgsAvatar/HO3D/ABF1/ABF10/model/0000.npz")
verts = torch.tensor(model_dict['minimal_shape']).unsqueeze(0)
faces  = np.load("/home/pfren/3dgsAvatar/3dgs-avatar-release/hand_models/misc/faces.npz")['faces']
# 创建 3D 网格。
texture = TexturesVertex(verts_features=torch.zeros_like(verts[0])[None])  # 使用白色纹理。
mesh = Meshes(verts=[verts[0]], faces=[torch.from_numpy(faces)],textures=texture)


# 假设你的相机内参是这样的：
fx, fy = torch.tensor(614.627), torch.tensor(614.101) # 焦距
cx, cy = 320.262, 238.469  # 主点
width, height = torch.tensor(640), torch.tensor(480)  # 图像的宽度和高度
K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]])
# 计算视场角（Field of View，FOV）。
fov_x = 2 * torch.atan(width / (2 * fx))
fov_y = 2 * torch.atan(height / (2 * fy))
fov = torch.tensor([[fov_x, fov_y]])
R= T[:3,:3].unsqueeze(0)
T= T[:3,3].unsqueeze(0)
# 创建相机。
#cameras = FoVPerspectiveCameras(device=device, R=R, T=T,fov=fov)
#cameras = PerspectiveCameras(R=R,T=T,focal_length=torch.tensor([fx,fy]).reshape([1,2]), principal_point=torch.tensor([cx,cy]).reshape([1,2]))
cameras = OpenGLPerspectiveCameras(R=R, T=T)
# 创建相机。
#cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=(fx, fy), principal_point=(cx, cy), image_size=(width, height))
# 初始化光源。
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# 初始化渲染器。
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=RasterizationSettings(image_size=512)
    ),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# 渲染图像。
images = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
# 保存图像。
plt.savefig("/home/pfren/3dgsAvatar/renderimg/output2.png")
'''
import open3d as o3d

def load_points(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    return torch.Tensor(np.asarray(pcd.points))

# 加载点云数据
points = load_points("/home/pfren/3dgsAvatar/HO3D/ABF1/cano_mano.ply")
points = Pointclouds(points.unsqueeze(0), features=torch.zeros_like(points).unsqueeze(0))
# 设置相机参数
R= T[:3,:3].unsqueeze(0)
T= T[:3,3].unsqueeze(0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# 设置渲染参数
raster_settings = PointsRasterizationSettings(
    image_size=512, 
    radius = 0.003,
    points_per_pixel = 10
)

# 创建渲染器
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
compositor = NormWeightedCompositor()
renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

# 渲染点云
images = renderer(points)

# 使用 matplotlib 保存图片
import matplotlib.pyplot as plt
plt.imsave('/home/cyc/pycharm/lxy/gs/renderimg/output2.png', images[0, ..., :3].cpu().numpy())
'''