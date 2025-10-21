#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

import numpy as np
from pytorch3d.ops.knn import knn_points

import trimesh
import pickle
from functools import lru_cache

from utils.graphics_utils import axis_angle_to_matrix


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def full_aiap_loss(gs_can, gs_obs, n_neighbors=5, articulated=False):
    xyz_can = gs_can.get_xyz
    xyz_obs = gs_obs.get_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_obs.get_covariance()

    if articulated:
        ismovable = gs_can.get_dynamic.detach() > 0.5
        if ismovable.sum() > 10:

            xyz_can = xyz_can[ismovable.squeeze(1)]
            xyz_obs = xyz_obs[ismovable.squeeze(1)]
            cov_can = cov_can[ismovable.squeeze(1)]
            cov_obs = cov_obs[ismovable.squeeze(1)]


    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0),
                             xyz_can.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return loss_xyz, loss_cov

def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                                 x_canonical.unsqueeze(0),
                                 K=n_neighbors + 1,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:,0,1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:,0,1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss


@lru_cache(maxsize=128)
def load_contacts(save_contact_paths="assets/contact_zones.pkl", display=False):
    with open(save_contact_paths, "rb") as p_f:
        contact_data = pickle.load(p_f)
    hand_verts = contact_data["verts"]
    return hand_verts, contact_data["contact_zones"]

def batch_mesh_contains_points(
    ray_origins,
    obj_triangles,
    direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]).cuda(),
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh

    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001
    # ray_origins.requires_grad = False
    # obj_triangles.requires_grad = False
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)

    dets = torch.bmm(
        v0v1.contiguous().view(batch_points_size, 1, 3), pvec.contiguous().view(batch_points_size, 3, 1)
    ).view(batch_size, triangle_nb)

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0

    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    not_parallel = parallel.logical_not()
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior


def batch_index_select(inp, dim, index):
    views = [inp.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(inp.shape))
    ]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def thresh_ious(gt_dists, pred_dists, thresh):
    """
    Computes the contact intersection over union for a given threshold
    """
    gt_contacts = gt_dists <= thresh
    pred_contacts = pred_dists <= thresh
    inter = (gt_contacts * pred_contacts).sum(1).float()
    union = union = (gt_contacts | pred_contacts).sum(1).float()
    iou = torch.zeros_like(union)
    iou[union != 0] = inter[union != 0] / union[union != 0]
    return iou


def meshiou(gt_dists, pred_dists, threshs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    For each thresholds, computes thresh_ious and averages accross batch dim
    """
    all_ious = []
    for thresh in threshs:
        ious = thresh_ious(gt_dists, pred_dists, thresh)
        all_ious.append(ious)
    iou_auc = np.mean(
        np.trapz(torch.stack(all_ious).cpu().numpy(), axis=0, x=threshs)
    )
    batch_ious = torch.stack(all_ious).mean(1)
    return batch_ious, iou_auc


def masked_mean_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    if valid_vals > 0:
        loss = (mask * dists).sum() / valid_vals
    else:
        loss = torch.Tensor(0).cuda()
    return loss


def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def thres_loss(vals, thres=25):
    """
    Args:
        vals: positive values !
    """
    thres_mask = (vals < thres).float()
    loss = masked_mean_loss(vals, thres_mask)
    return loss


def compute_naive_contact_loss(points_1, points_2, contact_threshold=25):
    dists = batch_pairwise_dist(points_1, points_2)
    mins12, _ = torch.min(dists, 1)
    mins21, _ = torch.min(dists, 2)
    loss_1 = thres_loss(mins12, contact_threshold)
    loss_2 = thres_loss(mins21, contact_threshold)
    loss = torch.mean((loss_1 + loss_2) / 2)
    return loss


def mesh_vert_int_exts(obj1_mesh, obj2_verts, result_distance, tol=0.1):
    nonzero = result_distance > tol
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    penetrating = [sign == 1][0] & nonzero
    exterior = [sign == -1][0] & nonzero
    return penetrating, exterior



def get_depth_info(obj_mesh, hand_verts):
    result_close, result_distance, _ = trimesh.proximity.closest_point(
        obj_mesh, hand_verts
    )
    penetrating, exterior = mesh_vert_int_exts(
        obj_mesh, hand_verts, result_distance
    )
    return result_close, result_distance, penetrating


def compute_contact_loss(
    hand_verts_pt,
    obj_verts_pt,
    obj_triangles,
    contact_thresh=15 / 1000,
    contact_mode="dist_tanh",
    collision_thresh=25 / 1000,
    collision_mode="dist_tanh",
    contact_target="all",
    contact_sym=False,
    contact_zones="zones",
):
    # obj_verts_pt = obj_verts_pt.detach()
    # hand_verts_pt = hand_verts_pt.detach()
    dists = batch_pairwise_dist(hand_verts_pt, obj_verts_pt)
    mins12, min12idxs = torch.min(dists, 1)
    mins21, min21idxs = torch.min(dists, 2)

    # Get obj triangle positions
    #obj_triangles = obj_verts_pt[:, obj_faces]
    exterior = batch_mesh_contains_points(
        hand_verts_pt.detach(), obj_triangles.detach()
    )
    penetr_mask = ~exterior
    results_close = batch_index_select(obj_verts_pt, 1, min21idxs)

    if contact_target == "all":
        anchor_dists = torch.norm(results_close - hand_verts_pt, 2, 2)
    elif contact_target == "obj":
        anchor_dists = torch.norm(results_close - hand_verts_pt.detach(), 2, 2)
    elif contact_target == "hand":
        anchor_dists = torch.norm(results_close.detach() - hand_verts_pt, 2, 2)
    else:
        raise ValueError(
            "contact_target {} not in [all|obj|hand]".format(contact_target)
        )
    if contact_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            contact_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            contact_vals = ((results_close - hand_verts_pt.detach()) ** 2).sum(
                2
            )
        elif contact_target == "hand":
            contact_vals = ((results_close.detach() - hand_verts_pt) ** 2).sum(
                2
            )
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
        below_dist = mins21 < (contact_thresh ** 2)
    elif contact_mode == "dist":
        # Use distance to penalize contact
        contact_vals = anchor_dists
        below_dist = mins21 < contact_thresh
    elif contact_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        contact_vals = contact_thresh * torch.tanh(
            anchor_dists / contact_thresh
        )
        # All points are taken into account
        below_dist = torch.ones_like(mins21).byte()
    else:
        raise ValueError(
            "contact_mode {} not in [dist_sq|dist|dist_tanh]".format(
                contact_mode
            )
        )
    if collision_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            collision_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            collision_vals = (
                (results_close - hand_verts_pt.detach()) ** 2
            ).sum(2)
        elif contact_target == "hand":
            collision_vals = (
                (results_close.detach() - hand_verts_pt) ** 2
            ).sum(2)
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
    elif collision_mode == "dist":
        # Use distance to penalize collision
        collision_vals = anchor_dists
    elif collision_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        collision_vals = collision_thresh * torch.tanh(
            anchor_dists / collision_thresh
        )
    else:
        raise ValueError(
            "collision_mode {} not in "
            "[dist_sq|dist|dist_tanh]".format(collision_mode)
        )

    missed_mask = below_dist & exterior
    if contact_zones == "tips":
        tip_idxs = [745, 317, 444, 556, 673]
        tips = torch.zeros_like(missed_mask)
        tips[:, tip_idxs] = 1
        missed_mask = missed_mask & tips
    elif contact_zones == "zones":
        _, contact_zones = load_contacts(
            "../lib/contact_zones.pkl"
        )
        contact_matching = torch.zeros_like(missed_mask)
        for zone_idx, zone_idxs in contact_zones.items():
            min_zone_vals, min_zone_idxs = mins21[:, zone_idxs].min(1)
            cont_idxs = mins12.new(zone_idxs)[min_zone_idxs]
            # For each batch keep the closest point from the contact zone
            contact_matching[
                [torch.range(0, len(cont_idxs) - 1).long(), cont_idxs.long()]
            ] = 1
        missed_mask = missed_mask & contact_matching
    elif contact_zones == "all":
        missed_mask = missed_mask
    else:
        raise ValueError(
            "contact_zones {} not in [tips|zones|all]".format(contact_zones)
        )

    # Apply losses with correct mask
    missed_loss = masked_mean_loss(contact_vals, missed_mask)
    penetr_loss = masked_mean_loss(collision_vals, penetr_mask)
    if contact_sym:
        obj2hand_dists = torch.sqrt(mins12)
        sym_below_dist = mins12 < contact_thresh
        sym_loss = masked_mean_loss(obj2hand_dists, sym_below_dist)
        missed_loss = missed_loss + sym_loss
    # print('penetr_nb: {}'.format(penetr_mask.sum()))
    # print('missed_nb: {}'.format(missed_mask.sum()))
    max_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).max(1)[0].mean()
    )
    mean_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).mean(1).mean()
    )
    contact_info = {
        "attraction_masks": missed_mask,
        "repulsion_masks": penetr_mask,
        "contact_points": results_close,
        "min_dists": mins21,
    }
    metrics = {
        "max_penetr": max_penetr_depth,
        "mean_penetr": mean_penetr_depth,
    }
    return missed_loss, penetr_loss, contact_info, metrics


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = True, threshold: float = 0.5) -> torch.Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

from torch_cluster import radius
def anti_penetration_loss(A_xyz, A_scaling, B_xyz, B_scaling, r_search=0.05):
    if A_xyz.numel() == 0 or B_xyz.numel() == 0:
        print("numel_0")
        return torch.tensor(0.0, device=A_xyz.device)

    # 转 float32 且 contiguous
    A_xyz = A_xyz.float().contiguous()
    B_xyz = B_xyz.float().contiguous()

    A_radius = A_scaling.max(dim=-1).values
    B_radius = B_scaling.max(dim=-1).values

    try:
        idx_i, idx_j = radius(A_xyz, B_xyz, r=r_search, max_num_neighbors=64)
    except RuntimeError as e:
        print("Radius search error:", e)
        return torch.tensor(0.0, device=A_xyz.device, requires_grad=True)

    if idx_i.numel() == 0:
        print("Radius search 0")
        return torch.tensor(0.0, device=A_xyz.device, requires_grad=True)

    # 确保索引不越界
    idx_i = idx_i.clamp(0, A_xyz.shape[0]-1)
    idx_j = idx_j.clamp(0, B_xyz.shape[0]-1)

    dist = (A_xyz[idx_i] - B_xyz[idx_j]).norm(dim=-1)
    r_sum = A_radius[idx_i] + B_radius[idx_j]
    penetration = F.relu(r_sum - dist)
    #
    # print("A_radius min/max:", A_radius.min().item(), A_radius.max().item())
    # print("B_radius min/max:", B_radius.min().item(), B_radius.max().item())
    # print("dist min/max:", dist.min().item(), dist.max().item())
    # print("penetration min/max:", penetration.min().item(), penetration.max().item())

    return penetration.mean()

def build_occupancy_grid(B_xyz, B_scaling, grid_res=128, sigma_scale=2.0, device="cuda"):
    # 1. 定义边界
    min_xyz = B_xyz.min(dim=0).values
    max_xyz = B_xyz.max(dim=0).values
    center = (min_xyz + max_xyz) / 2
    extent = (max_xyz - min_xyz).max() / 2 * 1.1  # 稍微放大
    grid_coords = torch.linspace(-1, 1, grid_res, device=device)

    # 2. 生成体素中心
    xx, yy, zz = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    grid_points = torch.stack([xx, yy, zz], dim=-1)  # [res, res, res, 3]
    grid_points = grid_points.reshape(-1, 3)  # [N, 3]

    # 映射到真实坐标系
    grid_points_world = grid_points * extent + center

    # 3. 计算 occupancy
    occ = torch.zeros(grid_points.shape[0], device=device)

    for i in range(B_xyz.shape[0]):
        center_i = B_xyz[i]
        radius_i = B_scaling[i].max()  # 粗略半径
        d2 = ((grid_points_world - center_i) ** 2).sum(-1)
        occ += torch.exp(-d2 / (2 * (radius_i * sigma_scale) ** 2))

    occ = 1 - torch.exp(-occ)  # 映射到 [0,1]
    occ = occ.reshape(grid_res, grid_res, grid_res).unsqueeze(0).unsqueeze(0)  # [1,1,res,res,res]
    return occ, center, extent


def occupancy_loss(A_xyz, occ_grid, center, extent):
    # A_xyz: [N,3]
    # occ_grid: [1,1,res,res,res]

    # 归一化到 [-1,1]
    norm_xyz = (A_xyz - center) / extent
    norm_xyz = norm_xyz.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,N,3]

    # grid_sample 输入需要 [B,C,D,H,W], 采样 [B,D,H,W,3]
    norm_xyz = norm_xyz.permute(0, 3, 1, 2, 4)  # [1,N,1,1,3]

    occ_val = F.grid_sample(occ_grid, norm_xyz, align_corners=True)
    return occ_val.mean()
