import os
import torch
import torch.nn as nn
import numpy as np
import time
import json
from datetime import datetime
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    FoVPerspectiveCameras,
    MeshRasterizer,
    SoftPhongShader,
    DirectionalLights,
    PerspectiveCameras,
    TexturesVertex
)
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import axis_angle_to_matrix
from smplx.body_models import SMPLLayer
import matplotlib.pyplot as plt
import trimesh



class TShirtPoseEstimator(nn.Module):
    def __init__(self, device="cuda", image_size=512):
        super().__init__()
        self.device = device
        self.image_size = image_size
        
        # Initialize SMPL model using SMPLLayer
        self.smpl = SMPLLayer(
            model_path='D:/Code/differentiable_rendering/smpl/models',
            gender='male',
            use_pose_blendshape=True,
            use_rotation_matrices=True
        ).to(device)
        
        # Initialize renderer
        self.renderer = self._setup_renderer()
        
        # Learnable pose parameters in axis-angle format
        self.global_orient = nn.Parameter(torch.zeros(1, 3).to(device))
        self.body_pose = nn.Parameter(torch.zeros(1, 23*3).to(device))
        self.betas = nn.Parameter(torch.zeros(1, 10).to(device))
        
    def _setup_renderer(self):
        # Use FoVPerspectiveCameras consistently
        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=60,
        )
        
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        
        lights = DirectionalLights(
            device=self.device,
            direction=((0, 1, 0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.3, 0.3, 0.3),),
            specular_color=((0.2, 0.2, 0.2),),
        )
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        )
        
        return renderer
    
    def _create_mesh_with_textures(self, vertices, faces):
        """Create mesh with simple vertex colors"""
        batch_size = vertices.shape[0]
        num_verts = vertices.shape[1]
        
        # Create a uniform color for all vertices (light gray)
        verts_rgb = torch.ones_like(vertices)[..., :3] * 0.7  # Light gray color
        
        # Create TexturesVertex
        textures = TexturesVertex(verts_features=verts_rgb)
        
        # Create mesh with textures
        mesh = Meshes(
            verts=vertices,
            faces=faces,
            textures=textures
        )
        return mesh
    
    def _compute_silhouette(self, rendered_images):
        """Convert rendered images to binary silhouettes"""
        # Use RGB channels to compute silhouette
        rgb = rendered_images[..., :3]
        return (rgb.mean(dim=-1) > 0).float()
    
    def _silhouette_loss(self, pred_silhouette, target_silhouette):
        """Compute IoU loss between silhouettes"""
        intersection = (pred_silhouette * target_silhouette).sum()
        union = (pred_silhouette + target_silhouette).clamp(0, 1).sum()
        return 1 - (intersection + 1e-6) / (union + 1e-6)
    
    def _surface_loss(self, pred_vertices, target_vertices):
        """Compute bi-directional Chamfer distance between surfaces"""
        pred_points = sample_points_from_meshes(pred_vertices, 5000)
        target_points = sample_points_from_meshes(target_vertices, 5000)
        loss, _ = chamfer_distance(pred_points, target_points)
        return loss
    
    def _convert_poses_to_rotation_matrices(self):
        """Convert axis-angle poses to rotation matrices"""
        global_orient_mat = axis_angle_to_matrix(self.global_orient)  # Shape: (1, 3, 3)
        
        body_pose_reshaped = self.body_pose.reshape(-1, 3)
        body_pose_mats = axis_angle_to_matrix(body_pose_reshaped)  # Shape: (23, 3, 3)
        body_pose_mats = body_pose_mats.reshape(1, 23, 3, 3)  # Shape: (1, 23, 3, 3)
        
        return global_orient_mat, body_pose_mats
    
    def forward(self, target_mesh):
        # Convert poses to rotation matrices
        global_orient_mat, body_pose_mats = self._convert_poses_to_rotation_matrices()
        
        # Get SMPL vertices using rotation matrices
        smpl_output = self.smpl(
            betas=self.betas,
            body_pose=body_pose_mats,
            global_orient=global_orient_mat
        )
        
        vertices = smpl_output.vertices
        faces = self.smpl.faces_tensor.to(self.device)[None]
        
        # Create meshes with textures
        smpl_mesh = self._create_mesh_with_textures(vertices, faces)
        
        # Ensure target mesh has textures
        if not target_mesh.textures:
            target_mesh = self._create_mesh_with_textures(
                target_mesh.verts_list(),
                target_mesh.faces_list()
            )
        
        # Render both meshes
        smpl_rendered = self.renderer(smpl_mesh)
        target_rendered = self.renderer(target_mesh)
        
        # Compute silhouettes
        smpl_silhouette = self._compute_silhouette(smpl_rendered)
        target_silhouette = self._compute_silhouette(target_rendered)
        
        # Compute losses
        sil_loss = self._silhouette_loss(smpl_silhouette, target_silhouette)
        surf_loss = self._surface_loss(smpl_mesh, target_mesh)
        
        # Optional: Add pose prior loss to prevent unrealistic poses
        pose_prior_loss = (torch.sum(self.global_orient ** 2) + 
                          torch.sum(self.body_pose ** 2))
        
        total_loss = 2.0* sil_loss + 1.0 * surf_loss + 0.001 * pose_prior_loss
        
        return total_loss, {
            'silhouette_loss': sil_loss.item(),
            'surface_loss': surf_loss.item(),
            'pose_prior_loss': pose_prior_loss.item()
        }
