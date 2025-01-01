import os
from datetime import datetime


def create_run_directory(base_dir):
    """Create a new run directory with timestamp"""
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find the next run index
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if existing_runs:
        run_indices = [int(d.split('_')[1]) for d in existing_runs]
        next_idx = max(run_indices) + 1
    else:
        next_idx = 0
        
    # Create directory name
    run_dir = f"run_{next_idx:03d}_{timestamp}"
    full_path = os.path.join(base_dir, run_dir)
    
    # Create directories
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(os.path.join(full_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(full_path, 'visualizations'), exist_ok=True)
    
    print(f"Created run directory: {full_path}")
    return full_path