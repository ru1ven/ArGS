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
import itertools

import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import strip_symmetric, build_scaling_rotation, points3DToImg
from utils.graphics_utils import getWorld2View2


def render(data,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=False,
           pose_refine=False,
           delay=False,
           white_bg=False,
           save=False,
           novel_data=None,
           prev_data=None,
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc_hand_r, pc_hand_l,pc_obj, refined_pc_obj, loss_reg, colors_precomp_r,colors_precomp_l, obj_colors_precomp, updated_camera,\
    movable_prob, pc_articulated, pivot, axis = scene.convert_gaussians(data, iteration, compute_loss, delay, prev_data=prev_data)
    if pose_refine:
        data = updated_camera
    if iteration <= 4 or iteration % 1001 == 0:
        pc_hand_r.save_ply('output/output_{}_r.ply'.format(iteration))
        #pc_hand_l.save_ply('output/output_{}_l.ply'.format(iteration))
        pc_obj.save_ply('output/output_obj_{}.ply'.format(iteration))
    if save:
        pc_hand_r.save_ply('/home/cyc/pycharm/lxy/3DGS/debug/pcl/pcl_r_{}_noise_0.ply'.format(int(data.subject_id)))
        #pc_hand_l.save_ply('/home/cyc/pycharm/lxy/3DGS/debug/pcl/pcl_l_{}_noise_0.ply'.format(int(data.subject_id)))
        pc_obj.save_ply('/home/cyc/pycharm/lxy/3DGS/debug/pcl/pcl_obj_{}_noise_0.ply'.format(int(data.subject_id)))
        print('save')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points_hand_r = torch.zeros_like(pc_hand_r.get_xyz, dtype=pc_hand_r.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_hand_l = None
    if pc_hand_l is not None:
        screenspace_points_hand_l = torch.zeros_like(pc_hand_l.get_xyz, dtype=pc_hand_l.get_xyz.dtype, requires_grad=True,
                                               device="cuda") + 0
    screenspace_points_obj = torch.zeros_like(pc_obj.get_xyz, dtype=pc_obj.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points_hand_r.retain_grad()
        if pc_hand_l is not None:
            screenspace_points_hand_l.retain_grad()
        screenspace_points_obj.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        #kernel_size=kernel_size,
        #subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=pc_hand_r.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D_r = pc_hand_r.get_xyz
    npt_hand_r = means3D_r.shape[0]
    obj_means3D = pc_obj.get_xyz
    if pc_hand_l is not None:
        means3D_l = pc_hand_l.get_xyz
        npt_hand_l = means3D_l.shape[0]
        means2D = torch.cat([screenspace_points_hand_r, screenspace_points_hand_l,screenspace_points_obj], dim=0)
    else:
        npt_hand_l = 0
        means2D = torch.cat([screenspace_points_hand_r,screenspace_points_obj], dim=0)

    opacity_r = pc_hand_r.get_opacity
    obj_opacity = pc_obj.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None
    shs = None

    if pc_hand_l is not None:
        opacity_l = pc_hand_l.get_opacity
        full_means3D = torch.cat([means3D_r, means3D_l,obj_means3D], dim=0)
        full_opacity = torch.cat([opacity_r, opacity_l, obj_opacity], dim=0)
        if pipe.compute_cov3D_python:
            cov3D_precomp = torch.cat([pc_hand_r.get_covariance(scaling_modifier),
                                   pc_hand_l.get_covariance(scaling_modifier),
                                   pc_obj.get_covariance(scaling_modifier)],dim=0)
        else:
            scales = torch.cat([pc_hand_r.get_scaling, pc_hand_l.get_scaling,pc_obj.get_scaling], dim=0)
            rotations = torch.cat([pc_hand_r.get_rotation, pc_hand_l.get_rotation,pc_obj.get_rotation], dim=0)
            # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if white_bg:
            colors_hand = torch.cat([colors_precomp_r,colors_precomp_l,torch.ones(obj_opacity.shape[0], 3, device=opacity_r.device)], dim=0)
        else:
            colors_hand = torch.cat([colors_precomp_r,colors_precomp_l,torch.zeros(obj_opacity.shape[0], 3, device=opacity_r.device)], dim=0)

    else:
        full_means3D = torch.cat([means3D_r,obj_means3D], dim=0)
        full_opacity = torch.cat([opacity_r, obj_opacity], dim=0)

    
        if pipe.compute_cov3D_python:
            cov3D_precomp = torch.cat([pc_hand_r.get_covariance(scaling_modifier),
                                   pc_obj.get_covariance(scaling_modifier)],dim=0)
        else:
            scales = torch.cat([pc_hand_r.get_scaling, pc_obj.get_scaling], dim=0)
            rotations = torch.cat([pc_hand_r.get_rotation, pc_obj.get_rotation], dim=0)

   

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if white_bg:
            colors_hand = torch.cat([colors_precomp_r,torch.ones(obj_opacity.shape[0], 3, device=opacity_r.device)], dim=0)
        else:
            colors_hand = torch.cat([colors_precomp_r,torch.zeros(obj_opacity.shape[0], 3, device=opacity_r.device)], dim=0)


    rendered_image, radii = rasterizer(
        means3D = full_means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_hand,
        opacities = full_opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    if pc_hand_l is not None:
        if white_bg:
            colors_obj = torch.cat([torch.ones(opacity_r.shape[0], 3, device=opacity_r.device),
                                    torch.ones(opacity_l.shape[0], 3, device=opacity_l.device),
                                    obj_colors_precomp], dim=0)
        else:
            colors_obj = torch.cat([torch.zeros(opacity_r.shape[0], 3, device=opacity_r.device),
                                    torch.zeros(opacity_l.shape[0], 3, device=opacity_l.device),
                                    obj_colors_precomp], dim=0)
        color_full = torch.cat([colors_precomp_r,colors_precomp_l, obj_colors_precomp], dim=0)
    else:
        if white_bg:
            colors_obj = torch.cat([torch.ones(opacity_r.shape[0], 3, device=opacity_r.device),
                                    obj_colors_precomp], dim=0)
        else:
            colors_obj = torch.cat([torch.zeros(opacity_r.shape[0], 3, device=opacity_r.device),
                                    obj_colors_precomp], dim=0)
        color_full = torch.cat([colors_precomp_r, obj_colors_precomp], dim=0)


    obj_rendered_image, obj_radii = rasterizer(
        means3D=full_means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_obj,
        opacities=full_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    full_render_image, full_radii = rasterizer(
        means3D = full_means3D,
        means2D = means2D,
        shs = shs,
        opacities = full_opacity,
        scales = scales,
        rotations = rotations,
        colors_precomp = color_full,
        cov3D_precomp = cov3D_precomp
    )

    #print(full_render_image.shape)

    opacity_hand = None
    opacity_obj = None
    opacity_ho = None

    if return_opacity:
        if white_bg:
            mask_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            opacity_raster_settings = GaussianRasterizationSettings(
                image_height=int(data.image_height),
                image_width=int(data.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=mask_color,
                scale_modifier=scaling_modifier,
                viewmatrix=data.world_view_transform,
                projmatrix=data.full_proj_transform,
                sh_degree=pc_hand_r.active_sh_degree,
                campos=data.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )
            rasterizer = GaussianRasterizer(raster_settings=opacity_raster_settings)
        if pc_hand_l is not None:
            colors_opacity_hand = torch.cat([torch.ones(opacity_r.shape[0], 3, device=opacity_r.device),
                                         torch.ones(opacity_l.shape[0], 3, device=opacity_l.device),
                                         torch.zeros(obj_opacity.shape[0], 3, device=obj_opacity.device)], dim=0)
            colors_opacity_obj = torch.cat([torch.zeros(opacity_r.shape[0], 3, device=opacity_r.device),
                                        torch.zeros(opacity_l.shape[0], 3, device=opacity_l.device),
                                        torch.ones(obj_opacity.shape[0], 3, device=obj_opacity.device)], dim=0)
        else:
            colors_opacity_hand = torch.cat([torch.ones(opacity_r.shape[0], 3, device=opacity_r.device),
                                         torch.zeros(obj_opacity.shape[0], 3, device=obj_opacity.device)], dim=0)
            colors_opacity_obj = torch.cat([torch.zeros(opacity_r.shape[0], 3, device=opacity_r.device),
                                        torch.ones(obj_opacity.shape[0], 3, device=obj_opacity.device)], dim=0)
        opacity_hand, _ = rasterizer(
            means3D=full_means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_opacity_hand,
            opacities=full_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_hand = opacity_hand[:1]

       
        opacity_obj, _ = rasterizer(
            means3D=full_means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_opacity_obj,
            opacities=full_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_obj = opacity_obj[:1]

        opacity_ho, _ = rasterizer(
            means3D = full_means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = torch.ones(full_opacity.shape[0], 3, device=full_opacity.device),
            opacities=full_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_ho = opacity_ho[:1]

        opacity_part = pc_obj.get_dynamic
        color_part = torch.cat([opacity_part, 1 - opacity_part, torch.zeros_like(opacity_part)], dim=-1)  # (n, 3)
        if pc_hand_l is not None:
            colors_opacity_part = torch.cat([torch.zeros(opacity_r.shape[0], 3, device=opacity_r.device),
                                        torch.zeros(opacity_l.shape[0], 3, device=opacity_l.device),
                                        color_part], dim=0)
        else:
            colors_opacity_part = torch.cat([torch.zeros(opacity_r.shape[0], 3, device=opacity_r.device),
                                        color_part], dim=0)
        opacity_part, _ = rasterizer(
            means3D=full_means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_opacity_part,
            opacities=full_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_static = opacity_part[1]
        opacity_dynamic = opacity_part[0]

        if novel_data is not None:

            raster_settings_novel = GaussianRasterizationSettings(
                image_height=int(novel_data.image_height),
                image_width=int(novel_data.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=novel_data.world_view_transform,
                projmatrix=novel_data.full_proj_transform,
                sh_degree=pc_hand_r.active_sh_degree,
                campos=novel_data.camera_center,
                #campo=torch.tensor([0, 0, 0], device=full_means3D.device).float(),
                prefiltered=False,
                debug=pipe.debug
            )
            rasterizer_novel = GaussianRasterizer(raster_settings=raster_settings_novel)
            novel_render, _ = rasterizer_novel(
                means3D=full_means3D,
                means2D=means2D,
                shs=shs,
                opacities=full_opacity,
                scales=scales,
                rotations=rotations,
                colors_precomp=color_full,
                cov3D_precomp=cov3D_precomp
            )

        else:
            novel_render = None


    return {"deformed_gaussian_r": pc_hand_r,
            "deformed_gaussian_l": pc_hand_l,
            "obj_deformed_gaussian": pc_obj,
            "obj_refined_gaussian": refined_pc_obj,
            "render": rendered_image,
            "obj_render": obj_rendered_image,
            "full_render": full_render_image,
            "novel_render": novel_render,
            "movable_prob": movable_prob,
            'pc_articulated':pc_articulated,
            "pivot": pivot,
            "axis": axis,

            "viewspace_points_r": screenspace_points_hand_r,
            "viewspace_points_l": screenspace_points_hand_l,
            "obj_viewspace_points": screenspace_points_obj,
            "full_viewspace_points": means2D,

            "visibility_filter_r" : radii[:npt_hand_r] > 0,
            "visibility_filter_l": radii[npt_hand_r:npt_hand_r+npt_hand_l] > 0,
            "obj_visibility_filter" : obj_radii[npt_hand_r+npt_hand_l:] > 0,
            "full_visibility_filter" : full_radii > 0,

            "radii_r": radii[:npt_hand_r],
            "radii_l": radii[npt_hand_r:npt_hand_r+npt_hand_l],
            "obj_radii": obj_radii[npt_hand_r+npt_hand_l:],
            "full_radii": full_radii,

            "loss_reg": loss_reg,

            "opacity_render": opacity_hand,
            "obj_opacity_render": opacity_obj,
            "full_opacity_render": opacity_ho,
            "opacity_static": opacity_static,
            "opacity_dynamic": opacity_dynamic,

            "updated_camera": updated_camera
            }

#import gof_rasterization as gof
def integrate(points3D, viewpoint_camera, pc, pipe, bg_color: torch.Tensor, kernel_size: float,
              scaling_modifier=1.0, override_color=None, subpixel_offset=None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = gof.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = gof.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz

    means2D = screenspace_points
    #opacity = pc.get_opacity_with_3D_filter
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    #compute_cov3D_python = False
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        #scales = pc.get_scaling_with_3D_filter
        scales = pc.get_scaling
        rotations = pc.get_rotation_precomp

    view2gaussian_precomp = None
    pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    colors_opacity_hand = torch.ones(opacity.shape[0], 3, device=opacity.device)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).


    rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
        points3D=points3D,
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_opacity_hand,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        #view2gaussian_precomp=view2gaussian_precomp
        )
    #print('integrate',rendered_image)


    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_opacity_hand,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        #view2gaussian_precomp=view2gaussian_precomp
        )

    #print('render',rendered_image.shape)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
           }
