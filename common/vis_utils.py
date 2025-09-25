import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# connection between the 8 points of 3d bbox
import torch

BONES_3D_BBOX = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
]


def plot_2d_bbox(bbox_2d, bones, color, ax):
    if ax is None:
        axx = plt
    else:
        axx = ax
    colors = cm.rainbow(np.linspace(0, 1, len(bbox_2d)))
    for pt, c in zip(bbox_2d, colors):
        axx.scatter(pt[0], pt[1], color=c, s=50)

    if bones is None:
        bones = BONES_3D_BBOX
    for bone in bones:
        sidx, eidx = bone
        # bottom of bbox is white
        if min(sidx, eidx) >= 4:
            color = "w"
        axx.plot(
            [bbox_2d[sidx][0], bbox_2d[eidx][0]],
            [bbox_2d[sidx][1], bbox_2d[eidx][1]],
            color,
        )
    return axx


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D
    numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image
    in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tobytes())


def concat_pil_images(images):
    """
    Put a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def stack_pil_images(images):
    """
    Stack a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def im_list_to_plt(image_list, figsize, title_list=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(image_list), figsize=figsize)
    for idx, (ax, im) in enumerate(zip(axes, image_list)):
        ax.imshow(im)
        ax.set_title(title_list[idx])
    fig.tight_layout()
    im = fig2img(fig)
    plt.close()
    return im

import open3d as o3d
def save_bone_contributions_with_joints(
    xyz,          # [N, 3] 点坐标
    pts_W,        # [N, B] 每个点对每个骨骼的权重
    joint_pos,    # [B, 3] 每个骨骼的位置
    out_dir="bone_plys"
):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    if isinstance(pts_W, torch.Tensor):
        pts_W = pts_W.detach().cpu().numpy()
    if isinstance(joint_pos, torch.Tensor):
        joint_pos = joint_pos.detach().cpu().numpy()

    print(xyz.shape)
    print(joint_pos.shape)

    os.makedirs(out_dir, exist_ok=True)
    N, B = pts_W.shape

    for b in range(B):
        weights = pts_W[:, b]  # [N]
        # 颜色映射：红(权重大), 蓝(权重小)
        colors = np.stack([
            weights,                   # R
            np.zeros_like(weights),   # G
            1 - weights               # B
        ], axis=1)
        colors = np.clip(colors, 0, 1)

        # 拼接骨骼位置（绿色）
        joint = joint_pos[b:b+1]  # [1, 3]
        joint_color = np.array([[0, 1, 0]])  # 绿色
        all_points = np.concatenate([xyz, joint], axis=0)
        all_colors = np.concatenate([colors, joint_color], axis=0)

        # 构造 PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # 保存
        filename = os.path.join(out_dir, f"bone_{b:02d}.ply")
        o3d.io.write_point_cloud(filename, pcd)

    print(f"[✔] 保存带骨骼位置的 {B} 个骨骼 ply 到目录 {out_dir}/")
