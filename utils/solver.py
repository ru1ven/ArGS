#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import copy

import numpy as np
import trimesh
from sklearn.neighbors import KDTree
from tqdm import tqdm


def register(source, target, type='icp_common', scale=False, **kwargs):
    if type == 'icp_common':
        from trimesh.registration import mesh_other
    elif type == 'icp_constrained':
        from tools.icp import mesh_other
    else:
        raise ValueError('Registration Type Should Be in {icp_common} and {icp_constrained}.')

    # register
    source2target, cost = mesh_other(source, target, scale=scale, **kwargs)
    # source2target, cost = mesh_other(source, target, scale=False)

    # transform
    source.apply_transform(source2target)

    return source, target, cost

class icp_ts():
    """
    @description:
    icp solver which only aligns translation and scale
    """
    def __init__(self, mesh_source, mesh_target):
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

        self.points_source = self.mesh_source.vertices.copy()
        self.points_target = self.mesh_target.vertices.copy()

    def sample_mesh(self, n=30000, mesh_id='both'):
        if mesh_id == 'source' or mesh_id == 'both':
            self.points_source, _ = trimesh.sample.sample_surface(self.mesh_source, n)
        if mesh_id == 'target' or mesh_id == 'both':
            self.points_target, _ = trimesh.sample.sample_surface(self.mesh_target, n)

        self.offset_source = self.points_source.mean(0)
        self.scale_source = np.sqrt(((self.points_source - self.offset_source)**2).sum() / len(self.points_source))
        self.offset_target = self.points_target.mean(0)
        self.scale_target = np.sqrt(((self.points_target - self.offset_target)**2).sum() / len(self.points_target))

        self.points_source = (self.points_source - self.offset_source) / self.scale_source * self.scale_target + self.offset_target

    def run_icp_f(self, max_iter = 10, stop_error = 1e-3, stop_improvement = 1e-5, verbose=0):
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1,3), dtype = np.float32)
        self.scale = 1.0
        self.A_c123 = []

        error = 1e8
        previous_error = error
        for i in range(0, max_iter):
            
            # Find closest target point for each source point:
            query_source_points = self.points_source * self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)

            closest_target_points = self.points_target[closest_target_points_index[:, 0], :]

            # Find closest source point for each target point:
            query_target_points = (self.points_target - self.trans)/self.scale
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index[:, 0], :]
            closest_source_points = closest_source_points * self.scale + self.trans
            query_target_points = self.points_target

            # Compute current error:
            error = (((query_source_points - closest_target_points)**2).sum() + ((query_target_points - closest_source_points)**2).sum()) / (query_source_points.shape[0] + query_target_points.shape[0])
            error = error ** 0.5
            if verbose >= 1:
                print(i, "th iter, error: ", error)

            if previous_error - error < stop_improvement:
                break
            else:
                previous_error = error

            if error < stop_error:
                break

            ''' 
            Build lsq linear system:
            / x1 1 0 0 \  / scale \     / x_t1 \
            | y1 0 1 0 |  |  t_x  |  =  | y_t1 |
            | z1 0 0 1 |  |  t_y  |     | z_t1 | 
            | x2 1 0 0 |  \  t_z  /     | x_t2 |
            | ...      |                | .... |
            \ zn 0 0 1 /                \ z_tn /
            '''
            A_c0 = np.vstack([self.points_source.reshape(-1, 1), self.points_source[closest_source_points_index[:, 0], :].reshape(-1, 1)])
            if i == 0:
                A_c1 = np.zeros((self.points_source.shape[0] + self.points_target.shape[0], 3), dtype=np.float32) + np.array([1.0, 0.0, 0.0])
                A_c1 = A_c1.reshape(-1, 1)
                A_c2 = np.zeros_like(A_c1)
                A_c2[1:,0] = A_c1[0:-1, 0]
                A_c3 = np.zeros_like(A_c1)
                A_c3[2:,0] = A_c1[0:-2, 0]

                self.A_c123 = np.hstack([A_c1, A_c2, A_c3])

            A = np.hstack([A_c0, self.A_c123])
            b = np.vstack([closest_target_points.reshape(-1, 1), query_target_points.reshape(-1, 1)])
            x = np.linalg.lstsq(A, b, rcond=None)
            self.scale = x[0][0]
            self.trans = (x[0][1:]).transpose()

    def get_trans_scale(self):
        all_scale = self.scale_target * self.scale / self.scale_source 
        all_trans = self.trans + self.offset_target * self.scale - self.offset_source * self.scale_target * self.scale / self.scale_source
        return all_trans, all_scale

    def export_source_mesh(self, output_name):
        self.mesh_source.vertices = (self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        self.mesh_source.export(output_name)

    def get_source_mesh(self):
        self.mesh_source.vertices = (self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        return self.mesh_source


from scipy.spatial.transform import Rotation as R


class icp_rts():
    """
    @description:
    ICP solver which aligns translation, rotation, and scale
    """

    def __init__(self, mesh_source, mesh_target):
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

        self.points_source = self.mesh_source.vertices.copy()
        self.points_target = self.mesh_target.vertices.copy()

    def sample_mesh(self, n=30000, mesh_id='both'):
        if mesh_id == 'source' or mesh_id == 'both':
            self.points_source, _ = trimesh.sample.sample_surface(self.mesh_source, n)
        if mesh_id == 'target' or mesh_id == 'both':
            self.points_target, _ = trimesh.sample.sample_surface(self.mesh_target, n)

        self.offset_source = self.points_source.mean(0)
        self.scale_source = np.sqrt(((self.points_source - self.offset_source) ** 2).sum() / len(self.points_source))
        self.offset_target = self.points_target.mean(0)
        self.scale_target = np.sqrt(((self.points_target - self.offset_target) ** 2).sum() / len(self.points_target))

        self.points_source = (self.points_source - self.offset_source) / self.scale_source * self.scale_target + self.offset_target

    def run_icp_f(self, max_iter=10, stop_error=1e-3, stop_improvement=1e-5, verbose=0):
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1, 3), dtype=np.float32)
        self.scale = 1.0
        self.rotation = R.from_euler('xyz', [0, 0, 0], degrees=False)

        error = 1e8
        previous_error = error
        # 假设前面已构建 KDTree、初始化 self.rotation/self.scale/self.trans 等
        for i in range(max_iter):
            # 1) 用当前变换把 source 投到 target 空间，查最近邻
            query_source_points = self.rotation.apply(self.points_source) * self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)
            closest_target_points = self.points_target[closest_target_points_index[:, 0], :]

            # 2) 为对称性也寻找反向对应（可选）
            query_target_points = (self.points_target - self.trans) / self.scale
            query_target_points = self.rotation.apply(query_target_points)
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index[:, 0], :]
            closest_source_points = self.rotation.apply(closest_source_points) * self.scale + self.trans

            # 3) 计算误差（保持你原有的定义）
            error = (((query_source_points - closest_target_points) ** 2).sum() + ((query_target_points - closest_source_points) ** 2).sum())
            denom = query_source_points.shape[0] + query_target_points.shape[0]
            error = (error / denom) ** 0.5

            # 停止判断...
            # (省略，还使用你原代码的 stop_improvement/stop_error 检查)

            # 4) 关键：用匹配对的质心中心化并做 SVD
            # 这里用 P = source_matches (in source coord BEFORE applying rotation/scale/trans)
            # and Q = corresponding target points (in target coord)
            # 但更直接：使用 transformed source matches (query_source_points) and closest_target_points
            P = query_source_points      # shape (N,3)
            Q = closest_target_points    # shape (N,3)

            centroid_P = P.mean(axis=0)
            centroid_Q = Q.mean(axis=0)

            P_centered = P - centroid_P
            Q_centered = Q - centroid_Q

            H = P_centered.T @ Q_centered
            U, S, Vt = np.linalg.svd(H)
            R_mat = Vt.T @ U.T
            if np.linalg.det(R_mat) < 0:
                Vt[2, :] *= -1
                R_mat = Vt.T @ U.T

            # 5) 计算 scale（推荐用基于内积的稳定公式）
            num = np.sum(S)  # trace(U S Vt) simplification
            den = np.sum((P_centered ** 2))
            if den <= 1e-12:
                scale_factor = 1.0
            else:
                scale_factor = num / den

            # 更新整体变换（累乘）
            # 新的 rotation 应当复合到现有 rotation： R_new = R_mat @ self.rotation.as_matrix()
            self.rotation = R.from_matrix(R_mat @ self.rotation.as_matrix())
            self.scale = self.scale * scale_factor

            # 6) 计算 translation，使 centroid 对齐： t = centroid_Q - s * R_mat @ centroid_P
            self.trans = centroid_Q - scale_factor * R_mat @ centroid_P


    def get_trans_scale_rotation(self):
        return self.trans, self.scale, self.rotation.as_euler('xyz', degrees=False)

    def export_source_mesh(self, output_name):
        # Apply the computed transformation to the source mesh
        self.mesh_source.vertices = self.rotation.apply((self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target)
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        self.mesh_source.export(output_name)

    def get_source_mesh(self):
        # Apply the computed transformation to the source mesh
        self.mesh_source.vertices = self.rotation.apply((self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target)
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        return self.mesh_source

import open3d as o3d



def calculate_metrics(aligned_mesh, target_mesh, is_sqrt=False):
    vertices_source = np.asarray(aligned_mesh.vertices) * 100
    vertices_target = np.asarray(target_mesh.vertices) * 100

    # dist_bidirectional = chamferDist(vertices_source, vertices_target, bidirectional=True,point_reduction = "mean") #* 0.001
    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)

    if is_sqrt:  # square-root chamfer
        gt_to_gen_chamfer = np.mean(one_distances)
    else:  # squared chamfer
        gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)

    if is_sqrt:
        gen_to_gt_chamfer = np.mean(two_distances)
    else:
        gen_to_gt_chamfer = np.mean(np.square(two_distances))

    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10

def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = 0.01  # 1#voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source, target, source_fpfh, target_fpfh, voxel_size, init_alignment
):
    distance_threshold = 0.01  # voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)

    # result_ransac.transformation
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_alignment,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
    )
    return result


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh

def compute_icp_metrics(
    target_mesh, source_mesh, num_iters=60, no_tqdm=False, is_sqrt=False
):
    voxel_size = 0.005
    # print(":: Load two meshes")
    # exp_id = "42465ff7e"

    # assert os.path.exists(target_deform_p)
    # assert os.path.exists(pred_deform_p)
    # target_mesh = o3d.io.read_triangle_mesh(target_deform_p)
    # source_mesh = o3d.io.read_triangle_mesh(pred_deform_p)

    center_mass = source_mesh.get_center()
    source_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(source_mesh.vertices) - center_mass
    )
    source_copy = copy.deepcopy(source_mesh)
    center_mass = target_mesh.get_center()
    target_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(target_mesh.vertices) - center_mass
    )

    # o3d.io.write_triangle_mesh("target.obj", target_mesh)
    # o3d.io.write_triangle_mesh("source.obj", source_mesh)

    # print(":: Sample mesh to point cloud")
    target = target_mesh.sample_points_uniformly(1000)
    source = source_mesh.sample_points_uniformly(1000)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    trans_init = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result_icp_ra = refine_registration(
        source, target, source_fpfh, target_fpfh, voxel_size, trans_init
    )

    # Apply the transformation to align the source mesh with the target
    aligned_source_mesh = source_copy.transform(result_icp_ra.transformation)
    cd_ra, f5_ra, f10_ra = calculate_metrics(aligned_source_mesh, target_mesh, is_sqrt)
    best_cd = cd_ra
    best_f5 = f5_ra
    best_f10 = f10_ra

    if no_tqdm:
        pbar = range(num_iters)
    else:
        pbar = tqdm(range(num_iters))
    # for iter in tqdm(range(num_iters)):
    for iter in pbar:
        try:
            result_ransac = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size
            )

            result_icp_nra = refine_registration(
                source,
                target,
                source_fpfh,
                target_fpfh,
                voxel_size,
                result_ransac.transformation,
            )
        except:
            print("Error in ICP: Skipping")
        aligned_source_mesh_ransac = copy.deepcopy(source_mesh).transform(
            result_icp_nra.transformation
        )

        cd_nra, f5_nra, f10_nra = calculate_metrics(
            aligned_source_mesh_ransac, target_mesh
        )
        # print(cd_nra)
        if cd_nra < best_cd:
            best_cd, best_f5, best_f10 = cd_nra, f5_nra, f10_nra

    # print("CD F5 F10", best_cd, best_f5, best_f10 )
    return best_cd, best_f5, best_f10


from open3d.geometry import TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector

def eval_icp_HOLD(pred_mesh, gt_mesh, metric_dict):
    faces = pred_mesh.faces
    faces_gt = gt_mesh.faces

    v3d_o_ra = Vector3dVector(np.asarray(pred_mesh.vertices))
    faces_o = Vector3iVector(np.asarray(faces))
    v3d_o_ra_gt = Vector3dVector(np.asarray(gt_mesh.vertices))
    faces_o_gt = Vector3iVector(np.asarray(faces_gt))
    source_mesh = TriangleMesh(v3d_o_ra, faces_o)
    target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)
    best_cd, best_f5, best_f10 = compute_icp_metrics(
        target_mesh, source_mesh, num_iters=10, no_tqdm=True
    )
    metric_dict["cd_icp"] = best_cd
    metric_dict["f5_icp"] = best_f5 * 100.0
    metric_dict["f10_icp"] = best_f10 * 100.0
    return metric_dict