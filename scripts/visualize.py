import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from smplx.body_models import SMPLLayer
from pytorch3d.transforms import axis_angle_to_matrix
import imageio
import trimesh

class SMPLVisualizer:
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
        R, T = look_at_view_transform(2.5, 0, 0)
        
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=60
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
            specular_color=((0.2, 0.2, 0.2),)
        )
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        
        return renderer
    
    def _create_mesh_with_textures(self, vertices, faces, color=None):
        """Create mesh with vertex colors"""
        if color is None:
            color = 0.7
        verts_rgb = torch.ones_like(vertices)[..., :3] * color
        textures = TexturesVertex(verts_features=verts_rgb)
        
        mesh = Meshes(
            verts=vertices,
            faces=faces,
            textures=textures
        )
        return mesh

    def _get_smpl_output(self, global_orient, body_pose, betas):
        """Get SMPL vertices given pose parameters"""
        global_orient_mat = axis_angle_to_matrix(global_orient)
        body_pose_reshaped = body_pose.reshape(-1, 3)
        body_pose_mats = axis_angle_to_matrix(body_pose_reshaped)
        body_pose_mats = body_pose_mats.reshape(1, 23, 3, 3)
        
        smpl_output = self.smpl(
            betas=betas,
            body_pose=body_pose_mats,
            global_orient=global_orient_mat
        )
        return smpl_output

    def create_transition_animation(self, pose_params_path, tshirt_mesh_path, 
                                 output_path='smpl_transition.gif', 
                                 num_frames=30, rotation_angle=30):
        """
        Create animation transitioning from initial T-pose to final optimized pose,
        showing both SMPL and t-shirt meshes
        """
        # Load final pose parameters
        final_params = torch.load(pose_params_path)
        final_global_orient = final_params['global_orient'].to(self.device)
        final_body_pose = final_params['body_pose'].to(self.device)
        final_betas = final_params['betas'].to(self.device)
        
        # Initial T-pose parameters
        initial_global_orient = torch.zeros_like(final_global_orient)
        initial_body_pose = torch.zeros_like(final_body_pose)
        initial_betas = final_betas.clone()
        
        # Save initial pose
        torch.save({
            'global_orient': initial_global_orient.cpu(),
            'body_pose': initial_body_pose.cpu(),
            'betas': initial_betas.cpu()
        }, 'D:/Code/differentiable_rendering/results/initial_pose.pth')
        
        # Load t-shirt mesh
        tshirt_mesh = trimesh.load(tshirt_mesh_path)
        tshirt_verts = torch.tensor(tshirt_mesh.vertices, dtype=torch.float32, device=self.device)
        tshirt_faces = torch.tensor(tshirt_mesh.faces, dtype=torch.int64, device=self.device)
        
        frames = []
        faces = self.smpl.faces_tensor.to(self.device)[None]
        
        # Create frames for pose transition
        for frame in range(num_frames * 2):
            if frame < num_frames:
                alpha = frame / (num_frames - 1)
            else:
                alpha = 1.0
                
            current_global_orient = (1 - alpha) * initial_global_orient + alpha * final_global_orient
            current_body_pose = (1 - alpha) * initial_body_pose + alpha * final_body_pose
            
            # Get SMPL output for current pose
            smpl_output = self._get_smpl_output(
                current_global_orient,
                current_body_pose,
                final_betas
            )
            vertices = smpl_output.vertices
            
            # Create meshes with different colors
            smpl_mesh = self._create_mesh_with_textures(vertices, faces, color=0.7)  # Gray
            tshirt_mesh = self._create_mesh_with_textures(tshirt_verts[None], tshirt_faces[None], color=0.2)  # Dark gray
            
            # Rotate camera for each frame
            angle = (frame * rotation_angle) % 360
            R, T = look_at_view_transform(2.5, 0, [angle])
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=60)
            
            self.renderer.rasterizer.cameras = cameras
            self.renderer.shader.cameras = cameras
            
            # Render both meshes
            image_smpl = self.renderer(smpl_mesh)
            image_tshirt = self.renderer(tshirt_mesh)
            
            # Create side-by-side comparison
            combined_image = torch.cat([image_smpl, image_tshirt], dim=1)
            image_np = (combined_image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            frames.append(image_np)
        
        # Save as GIF
        imageio.mimsave(output_path, frames, duration=50)
        print(f"Transition animation saved to {output_path}")
        
        # Create side-by-side comparison image with both SMPL and t-shirt
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Initial SMPL pose
        smpl_output = self._get_smpl_output(initial_global_orient, initial_body_pose, final_betas)
        mesh = self._create_mesh_with_textures(smpl_output.vertices, faces, color=0.7)
        image = self.renderer(mesh)
        axes[0, 0].imshow(image[0, ..., :3].cpu().numpy())
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Initial SMPL T-pose')
        
        # Initial t-shirt
        tshirt_mesh = self._create_mesh_with_textures(tshirt_verts[None], tshirt_faces[None], color=0.2)
        image = self.renderer(tshirt_mesh)
        axes[0, 1].imshow(image[0, ..., :3].cpu().numpy())
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Initial T-shirt Mesh')
        
        # Final SMPL pose
        smpl_output = self._get_smpl_output(final_global_orient, final_body_pose, final_betas)
        mesh = self._create_mesh_with_textures(smpl_output.vertices, faces, color=0.7)
        image = self.renderer(mesh)
        axes[1, 0].imshow(image[0, ..., :3].cpu().numpy())
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Final Optimized SMPL Pose')
        
        # Final t-shirt (same as initial, for reference)
        axes[1, 1].imshow(image[0, ..., :3].cpu().numpy())
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Target T-shirt Mesh')
        
        plt.tight_layout()
        plt.savefig('D:/Code/differentiable_rendering/results/pose_tshirt_comparison.png')
        plt.close()
        print("Side-by-side comparison saved to pose_tshirt_comparison.png")

# Example usage
if __name__ == "__main__":
    run_idx = 'c'
    smpl_model_path = 'D:/Code/differentiable_rendering/smpl/models'
    pose_params_path = f'D:/Code/differentiable_rendering/results/{run_idx}/final_model.pth'
    tshirt_mesh_path = "D:/Code/differentiable_rendering/data/tshirt.obj"
    
    visualizer = SMPLVisualizer(smpl_model_path)
    visualizer.create_transition_animation(
        pose_params_path,
        tshirt_mesh_path,
        output_path=f'D:/Code/differentiable_rendering/results/{run_idx}/smpl_tshirt_transition.gif',
        num_frames=30,
        rotation_angle=15
    )