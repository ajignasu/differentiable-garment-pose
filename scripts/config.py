import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RendererConfig:
    image_size: int = 512
    fov: int = 60
    dist: float = 2.7
    views: List[Tuple[float, float]] = ((0, 0), (90, 0), (0, 90))  # (elevation, azimuth)
    blur_radius: float = 0.0
    faces_per_pixel: int = 1

@dataclass
class OptimizationConfig:
    num_iterations: int = 1000
    lr_global_orient: float = 0.001
    lr_body_pose: float = 0.001
    lr_betas: float = 0.0005
    surface_loss_weight: float = 0.5
    pose_prior_weight: float = 0.001
    chamfer_points: int = 5000
    checkpoint_freq: int = 50
    use_scheduler: bool = True
    scheduler_patience: int = 100
    scheduler_factor: float = 0.5

@dataclass
class SMPLConfig:
    model_path: str = "D:/Code/differentiable_rendering/smpl/models"
    gender: str = "male"
    use_pose_blendshape: bool = True
    use_rotation_matrices: bool = True

@dataclass
class Config:
    # Paths
    base_dir: str = "D:/Code/differentiable_rendering"
    data_dir: str = "data"
    results_dir: str = "results"
    tshirt_mesh_name: str = "tshirt_scaled_decimate_1by128.obj"
    
    # Configurations
    renderer: RendererConfig = RendererConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    smpl: SMPLConfig = SMPLConfig()
    
    # Device
    device: str = "cuda"
    
    @property
    def tshirt_mesh_path(self) -> str:
        return os.path.join(self.base_dir, self.data_dir, self.tshirt_mesh_name)
    
    @property
    def smpl_model_path(self) -> str:
        return os.path.join(self.base_dir, "smpl/models")
