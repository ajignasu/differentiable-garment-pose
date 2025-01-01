# conda activate pytorch3d

import os
import torch
import torch.nn as nn
import numpy as np
import time
import json
from utils import create_run_directory
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

class MeshComparisonVisualizer:
    def __init__(self, smpl_model_path, device="cuda", image_size=512):
        self.device = device
        self.image_size = image_size
        
        # Initialize SMPL model
        self.smpl = SMPLLayer(
            model_path=smpl_model_path,
            gender='male',
            use_pose_blendshape=True,
            use_rotation_matrices=True
        ).to(device)
        
        # Setup renderer
        self.renderer = self._setup_renderer()
    
    def _setup_renderer(self):
        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=60,
        )
        
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
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
    
    def _get_initial_smpl(self):
        """Get SMPL mesh in T-pose"""
        initial_pose = torch.zeros((1, 69)).to(self.device)  # Changed from 23*3
        initial_orient = torch.zeros((1, 3)).to(self.device)
        initial_betas = torch.zeros((1, 10)).to(self.device)
        
        # Convert to rotation matrices first
        global_orient_mat = axis_angle_to_matrix(initial_orient)  # Shape: (1, 3, 3)
        body_pose_reshaped = initial_pose.reshape(-1, 3)
        body_pose_mats = axis_angle_to_matrix(body_pose_reshaped)  # Shape: (23, 3, 3)
        body_pose_mats = body_pose_mats.reshape(1, 23, 3, 3)
        
        smpl_output = self.smpl(
            body_pose=body_pose_mats,
            global_orient=global_orient_mat,
            betas=initial_betas
        )
        
        vertices = smpl_output.vertices
        faces = self.smpl.faces_tensor.to(self.device)[None]
        
        # Create red-tinted SMPL mesh
        verts_rgb = torch.ones_like(vertices)[..., :3] * torch.tensor([0.8, 0.2, 0.2]).to(self.device)
        textures = TexturesVertex(verts_features=verts_rgb)
        
        return Meshes(
            verts=vertices,
            faces=faces,
            textures=textures
        )
    
    def load_tshirt_mesh(self, mesh_path):
        """Load and normalize t-shirt mesh"""
        mesh = trimesh.load(mesh_path)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)
        
        # Center the mesh
        verts = verts - verts.mean(dim=0, keepdim=True)
        
        # Get SMPL t-pose height for reference
        initial_pose = torch.zeros((1, 69)).to(self.device)
        initial_orient = torch.zeros((1, 3)).to(self.device)
        initial_betas = torch.zeros((1, 10)).to(self.device)
        
        # Convert to rotation matrices
        global_orient_mat = axis_angle_to_matrix(initial_orient)
        body_pose_reshaped = initial_pose.reshape(-1, 3)
        body_pose_mats = axis_angle_to_matrix(body_pose_reshaped)
        body_pose_mats = body_pose_mats.reshape(1, 23, 3, 3)
        
        smpl_output = self.smpl(
            body_pose=body_pose_mats,
            global_orient=global_orient_mat,
            betas=initial_betas
        )
        smpl_verts = smpl_output.vertices[0]
        smpl_height = smpl_verts[:, 1].max() - smpl_verts[:, 1].min()
        
        # Scale t-shirt to match SMPL height
        tshirt_height = verts[:, 1].max() - verts[:, 1].min()
        height_scale = smpl_height / tshirt_height
        verts = verts * height_scale
        
        # Apply rotations to align with SMPL
        # First rotate 90 degrees around X-axis to get upright
        angle_x = torch.tensor([0.5*np.pi], device=self.device)
        Rx = torch.tensor([[1., 0., 0.],
                        [0., torch.cos(angle_x), -torch.sin(angle_x)],
                        [0., torch.sin(angle_x), torch.cos(angle_x)]], device=self.device)
        
        # Then rotate 90 degrees around Y-axis to face front
        # angle_y = torch.tensor([np.pi/2], device=self.device)
        # Ry = torch.tensor([[torch.cos(angle_y), 0., torch.sin(angle_y)],
        #                 [0., 1., 0.],
        #                 [-torch.sin(angle_y), 0., torch.cos(angle_y)]], device=self.device)
        
        # Apply rotations
        # verts = verts @ Rx.T  # First X rotation
        # verts = verts @ Ry.T  # Then Y rotation
        
        # Translate to match SMPL center
        smpl_center = smpl_verts.mean(dim=0)
        verts = verts + smpl_center
        
        # Create mesh with textures
        verts_rgb = torch.ones_like(verts)[..., :3] * torch.tensor([0.2, 0.5, 0.8]).to(self.device)
        textures = TexturesVertex(verts_features=verts_rgb[None])
        
        return Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
    
    def visualize_initial_state(self, tshirt_mesh_path, output_dir):
        """Create visualization of initial SMPL and t-shirt meshes"""
        # Get meshes
        smpl_mesh = self._get_initial_smpl()
        tshirt_mesh = self.load_tshirt_mesh(tshirt_mesh_path)
        
        # Create multi-view visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        view_angles = [(0, 0), (90, 0), (0, 90)]  # (elevation, azimuth)
        titles = ['Front View', 'Side View', 'Top View']
        
        for idx, (elev, azim) in enumerate(view_angles):
            # Set camera position
            R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
            self.renderer.rasterizer.cameras = FoVPerspectiveCameras(
                device=self.device,
                R=R,
                T=T,
                fov=60
            )
            
            # Render SMPL
            image_smpl = self.renderer(smpl_mesh)
            axes[0, idx].imshow(image_smpl[0, ..., :3].cpu().numpy())
            axes[0, idx].axis('off')
            axes[0, idx].set_title(f'SMPL: {titles[idx]}')
            
            # Render t-shirt
            image_tshirt = self.renderer(tshirt_mesh)
            axes[1, idx].imshow(image_tshirt[0, ..., :3].cpu().numpy())
            axes[1, idx].axis('off')
            axes[1, idx].set_title(f'T-shirt: {titles[idx]}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/initial_state_comparison.png')
        print(f"Saved comparison to {output_dir}/initial_state_comparison.png")
        
        # Print scale information
        smpl_verts = smpl_mesh.verts_packed()
        tshirt_verts = tshirt_mesh.verts_packed()
        
        print("\nScale Analysis:")
        print(f"SMPL height range: {smpl_verts[:, 1].min():.3f} to {smpl_verts[:, 1].max():.3f}")
        print(f"SMPL width range: {smpl_verts[:, 0].min():.3f} to {smpl_verts[:, 0].max():.3f}")
        print(f"SMPL depth range: {smpl_verts[:, 2].min():.3f} to {smpl_verts[:, 2].max():.3f}")
        print(f"\nT-shirt height range: {tshirt_verts[:, 1].min():.3f} to {tshirt_verts[:, 1].max():.3f}")
        print(f"T-shirt width range: {tshirt_verts[:, 0].min():.3f} to {tshirt_verts[:, 0].max():.3f}")
        print(f"T-shirt depth range: {tshirt_verts[:, 2].min():.3f} to {tshirt_verts[:, 2].max():.3f}")


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

    def load_tshirt_mesh(self, mesh_path, debug_vis=False):
        """Load and normalize t-shirt mesh with debug visualization"""
        mesh = trimesh.load(mesh_path)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)
        
        if debug_vis:
            self._debug_visualize(verts, faces, "Initial")
        
        # Center the mesh
        verts = verts - verts.mean(dim=0, keepdim=True)
        if debug_vis:
            self._debug_visualize(verts, faces, "Centered")
        
        # Scale to match SMPL
        smpl_output = self.smpl()
        smpl_verts = smpl_output.vertices[0]
        # smpl_height = smpl_verts[:, 1].max() - smpl_verts[:, 1].min()
        # tshirt_height = verts[:, 1].max() - verts[:, 1].min()
        # height_scale = smpl_height / tshirt_height
        # verts = verts * height_scale
        # if debug_vis:
        #     self._debug_visualize(verts, faces, "Scaled")
        
        # First rotation: get upright
        # angle_x = torch.tensor([0.5*np.pi], device=self.device)
        # Rx = torch.tensor([[1., 0., 0.],
        #                 [0., torch.cos(angle_x), -torch.sin(angle_x)],
        #                 [0., torch.sin(angle_x), torch.cos(angle_x)]], device=self.device)
        # verts = verts @ Rx.T
        # if debug_vis:
        #     self._debug_visualize(verts, faces, "After X rotation")
        
        # Second rotation: face front
        # angle_y = torch.tensor([np.pi/2], device=self.device)
        # Ry = torch.tensor([[torch.cos(angle_y), 0., torch.sin(angle_y)],
        #                 [0., 1., 0.],
        #                 [-torch.sin(angle_y), 0., torch.cos(angle_y)]], device=self.device)
        # verts = verts @ Ry.T
        # if debug_vis:
            # self._debug_visualize(verts, faces, "After Y rotation")
        
        # Translate to SMPL center
        smpl_center = smpl_verts.mean(dim=0)
        verts = verts + smpl_center
        if debug_vis:
            self._debug_visualize(verts, faces, "Final Position", show_smpl=True)
        
        # Create mesh with textures
        verts_rgb = torch.ones_like(verts)[..., :3] * torch.tensor([0.2, 0.5, 0.8]).to(self.device)
        textures = TexturesVertex(verts_features=verts_rgb[None])
        
        return Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )

    def _debug_visualize(self, verts, faces, title, show_smpl=False):
        """Helper function to visualize mesh state during alignment"""
        fig = plt.figure(figsize=(15, 5))
        view_angles = [(0, 0), (90, 0), (0, 90)]
        titles = ['Front View', 'Side View', 'Top View']
        
        for idx, (elev, azim) in enumerate(view_angles):
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')
            
            # Plot t-shirt mesh
            ax.scatter(verts[:, 0].cpu(), verts[:, 1].cpu(), verts[:, 2].cpu(), 
                    c='b', alpha=0.1, s=1)
            
            if show_smpl:
                # Plot SMPL mesh in red for reference
                smpl_output = self.smpl()
                smpl_verts = smpl_output.vertices[0]
                ax.scatter(smpl_verts[:, 0].cpu(), smpl_verts[:, 1].cpu(), 
                        smpl_verts[:, 2].cpu(), c='r', alpha=0.1, s=1)
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'{title}: {titles[idx]}')
        
        plt.tight_layout()
        plt.show()
    


def train_pose_estimator(tshirt_mesh_path, run_dir, num_iterations=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and optimizer
    model = TShirtPoseEstimator(device=device)
    
    # Load and normalize target t-shirt mesh
    target_mesh = model.load_tshirt_mesh(tshirt_mesh_path, debug_vis=True).to(device)
    
    # Save initial t-shirt mesh state
    torch.save({
        'vertices': target_mesh.verts_list()[0].cpu(),
        'faces': target_mesh.faces_list()[0].cpu(),
    }, os.path.join(run_dir, 'initial_tshirt.pth'))
    
    optimizer = torch.optim.Adam([
        {'params': model.global_orient, 'lr': 0.001},
        {'params': model.body_pose, 'lr': 0.001},
        {'params': model.betas, 'lr': 0.0005}
    ])
    
    # Store loss history
    loss_history = []
    
    # Training loop
    start_time = time.time()
    best_loss = float('inf')
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        loss, metrics = model(target_mesh)
        loss.backward()
        optimizer.step()
        
        loss_history.append({
            'iteration': iteration,
            'total_loss': loss.item(),
            **metrics
        })
        
        if iteration % 50 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration}/{num_iterations}")
            print(f"Time taken: {elapsed_time:.2f} s")
            print(f"Time per iteration: {elapsed_time/(iteration+1):.2f} s")
            print(f"Total Loss: {loss.item():.4f}")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            print("---")
            
            # Save checkpoint
            checkpoint_path = os.path.join(run_dir, 'checkpoints', f'checkpoint_{iteration:04d}.pth')
            torch.save({
                'iteration': iteration,
                'global_orient': model.global_orient.detach().cpu(),
                'body_pose': model.body_pose.detach().cpu(),
                'betas': model.betas.detach().cpu(),
                'loss': loss.item(),
                'metrics': metrics,
                'optimizer_state': optimizer.state_dict()
            }, checkpoint_path)
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'iteration': iteration,
                    'global_orient': model.global_orient.detach().cpu(),
                    'body_pose': model.body_pose.detach().cpu(),
                    'betas': model.betas.detach().cpu(),
                    'loss': loss.item(),
                    'metrics': metrics
                }, os.path.join(run_dir, 'best_model.pth'))

    print(f"Training complete. Total time: {time.time() - start_time:.2f} s")
    
    # Save final results
    torch.save({
        'global_orient': model.global_orient.detach().cpu(),
        'body_pose': model.body_pose.detach().cpu(),
        'betas': model.betas.detach().cpu()
    }, os.path.join(run_dir, 'final_model.pth'))
    
    # Save loss history
    with open(os.path.join(run_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)
    
    # Save run metadata
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_iterations': num_iterations,
        'best_loss': best_loss,
        'total_time': time.time() - start_time,
        'tshirt_mesh_path': tshirt_mesh_path
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    return model

if __name__ == "__main__":
    base_dir = 'D:/Code/differentiable_rendering'
    smpl_model_path = os.path.join(base_dir, 'smpl/models')
    tshirt_mesh_path = os.path.join(base_dir, "data/tshirt_scaled_decimate_1by128.obj")
    
    # Create run directory
    run_dir = create_run_directory(os.path.join(base_dir, 'results'))
    
    # First visualize
    visualizer = MeshComparisonVisualizer(smpl_model_path)
    initial_state_path = os.path.join(run_dir, 'visualizations', 'initial_state.png')
    visualizer.visualize_initial_state(tshirt_mesh_path, run_dir)
    
    # Then train
    model = train_pose_estimator(tshirt_mesh_path, run_dir, num_iterations=200)