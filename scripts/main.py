import os
import optuna
from dataclasses import dataclass
from typing import Dict, Any
import json
from datetime import datetime
from hyperopt import run_hyperparameter_tuning
from models import TShirtPoseEstimator

@dataclass
class Config:
    base_dir: str = "D:/Code/differentiable_rendering"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune'],
                      help='Run mode: normal training or hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=50, 
                      help='Number of trials for hyperparameter tuning')
    parser.add_argument('--study_name', type=str, default=None,
                      help='Name for the optimization study')
    args = parser.parse_args()

    # Load base configuration
    config = Config()

    if args.mode == 'tune':
        print("Starting hyperparameter tuning...")
        study = run_hyperparameter_tuning(
            base_config=config,
            n_trials=args.n_trials,
            study_name=args.study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("\nBest hyperparameters found:")
        print(f"Best loss: {study.best_value:.4f}")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
    else:
        print("Running normal training with default parameters...")
        
        model = TShirtPoseEstimator.train_pose_estimator(config.tshirt_mesh_path, ...)

if __name__ == "__main__":
    main()