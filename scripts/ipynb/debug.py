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
from pytorch3d.transforms import axis_angle_to_matrix
from smplx.body_models import SMPLLayer
import trimesh

class DualMeshVisualizer:
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
            # Enable depth testing to handle overlapping meshes
            z_clip_value=100.0,
            perspective_correct=True,
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
    
    def load_meshes(self, mesh1_path, mesh2_path=None):
        """Load two meshes and prepare for visualization"""
        # Load first mesh (e.g., T-shirt)
        mesh1 = trimesh.load(mesh1_path)
        verts1 = torch.tensor(mesh1.vertices, dtype=torch.float32, device=self.device)
        faces1 = torch.tensor(mesh1.faces, dtype=torch.int64, device=self.device)
        
        # Create blue color for first mesh
        verts_rgb1 = torch.ones_like(verts1)[..., :3] * torch.tensor([0.2, 0.5, 0.8]).to(self.device)
        
        meshes = []
        meshes.append(Meshes(
            verts=[verts1],
            faces=[faces1],
            textures=TexturesVertex(verts_features=verts_rgb1[None])
        ))
        
        if mesh2_path:
            # Load second mesh (e.g., SMPL)
            mesh2 = trimesh.load(mesh2_path)
            verts2 = torch.tensor(mesh2.vertices, dtype=torch.float32, device=self.device)
            faces2 = torch.tensor(mesh2.faces, dtype=torch.int64, device=self.device)
            
            # Create red color for second mesh
            verts_rgb2 = torch.ones_like(verts2)[..., :3] * torch.tensor([0.8, 0.2, 0.2]).to(self.device)
            
            meshes.append(Meshes(
                verts=[verts2],
                faces=[faces2],
                textures=TexturesVertex(verts_features=verts_rgb2[None])
            ))
        
        return meshes
    
    def render_meshes(self, meshes, dist=2.7, elev=0, azim=0):
        """Render multiple meshes from a specific viewpoint"""
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        
        self.renderer.rasterizer.cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=60
        )
        
        # Render each mesh
        images = []
        for mesh in meshes:
            image = self.renderer(mesh)
            images.append(image[0, ..., :3].cpu().numpy())
        
        # Combine images using alpha compositing
        combined_image = np.zeros_like(images[0])
        for image in images:
            mask = image.sum(axis=-1) > 0
            combined_image[mask] = image[mask]
        
        return combined_image
    
    def visualize_meshes(self, mesh1_path, mesh2_path=None, output_path=None):
        """Create multi-view visualization of two meshes"""
        meshes = self.load_meshes(mesh1_path, mesh2_path)
        
        # Create multi-view visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        view_angles = [
            (0, 0),    # Front view
            (0, 90),   # Side view
            (90, 0),   # Top view
            (45, 45),  # 3/4 view 1
            (45, 135), # 3/4 view 2
            (30, 0),   # Slight top view
        ]
        titles = ['Front View', 'Side View', 'Top View', '3/4 View 1', '3/4 View 2', 'Top Angled']
        
        for idx, ((elev, azim), title) in enumerate(zip(view_angles, titles)):
            row, col = idx // 3, idx % 3
            image = self.render_meshes(meshes, dist=2.7, elev=elev, azim=azim)
            axes[row, col].imshow(image)
            axes[row, col].axis('off')
            axes[row, col].set_title(title)
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Saved visualization to {output_path}")
        plt.show()
        plt.close()
        
        # Print mesh statistics
        for i, mesh in enumerate(meshes):
            print(f"\nMesh {i+1} Statistics:")
            verts = mesh.verts_packed()
            print(f"Y-axis (height): {verts[:, 1].min():.3f} to {verts[:, 1].max():.3f}")
            print(f"X-axis (width): {verts[:, 0].min():.3f} to {verts[:, 0].max():.3f}")
            print(f"Z-axis (depth): {verts[:, 2].min():.3f} to {verts[:, 2].max():.3f}")

def main():
    smpl_model_path = 'D:/Code/differentiable_rendering/smpl/models'
    mesh1_path = "D:/Code/differentiable_rendering/data/tshirt.obj"
    mesh2_path = "D:/Code/differentiable_rendering/data/smpl_default.obj"  # If you have SMPL mesh
    output_path = "D:/Code/differentiable_rendering/results/mesh_comparison.png"
    
    visualizer = DualMeshVisualizer(smpl_model_path)
    visualizer.visualize_meshes(mesh1_path, mesh2_path, output_path)

if __name__ == "__main__":
    main()


# import torch
# import numpy as np
# from smplx.body_models import SMPLLayer
# import trimesh

# def save_smpl_mesh(smpl_model_path, output_path, device="cuda"):
#     """Save default SMPL mesh as OBJ file"""
#     # Initialize SMPL model
#     smpl = SMPLLayer(
#         model_path=smpl_model_path,
#         gender='male',
#     ).to(device)
    
#     # Get SMPL output with default parameters
#     smpl_output = smpl()
    
#     # Get vertices and faces
#     vertices = smpl_output.vertices[0].detach().cpu().numpy()
#     faces = smpl.faces_tensor.cpu().numpy()
    
#     # Create trimesh object
#     mesh = trimesh.Trimesh(
#         vertices=vertices,
#         faces=faces,
#         process=False  # Disable processing to keep original vertices/faces
#     )
    
#     # Save as OBJ
#     mesh.export(output_path)
#     print(f"Saved SMPL mesh to {output_path}")
    
#     # Print mesh statistics
#     print("\nMesh Statistics:")
#     print(f"Number of vertices: {len(vertices)}")
#     print(f"Number of faces: {len(faces)}")
#     print(f"Y-axis (height) range: {vertices[:, 1].min():.3f} to {vertices[:, 1].max():.3f}")
#     print(f"X-axis (width) range: {vertices[:, 0].min():.3f} to {vertices[:, 0].max():.3f}")
#     print(f"Z-axis (depth) range: {vertices[:, 2].min():.3f} to {vertices[:, 2].max():.3f}")

# if __name__ == "__main__":
#     smpl_model_path = 'D:/Code/differentiable_rendering/smpl/models'
#     output_path = "D:/Code/differentiable_rendering/data/smpl_default.obj"
    
#     save_smpl_mesh(smpl_model_path, output_path)