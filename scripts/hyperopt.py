import optuna
import json
from datetime import datetime
import os
import torch
from typing import Dict, Any

def run_hyperparameter_tuning(base_config: Config, n_trials: int = 50):
    """
    Run hyperparameter optimization study
    
    Args:
        base_config: Base configuration to modify during trials
        n_trials: Number of different parameter combinations to try
    """
    def objective(trial: optuna.Trial) -> float:
        """Objective function for each trial"""
        # Copy base config
        config = base_config
        
        # Suggest values for hyperparameters
        trial_params = {
            'lr_global_orient': trial.suggest_float('lr_global_orient', 1e-5, 1e-2, log=True),
            'lr_body_pose': trial.suggest_float('lr_body_pose', 1e-5, 1e-2, log=True),
            'lr_betas': trial.suggest_float('lr_betas', 1e-6, 1e-3, log=True),
            'surface_loss_weight': trial.suggest_float('surface_loss_weight', 0.1, 1.0),
            'pose_prior_weight': trial.suggest_float('pose_prior_weight', 1e-4, 1e-2, log=True),
            'num_iterations': trial.suggest_int('num_iterations', 100, 500)
        }
        
        # Create directory for this trial
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_dir = os.path.join(config.base_dir, config.results_dir, 
                               f"trial_{trial.number:03d}_{timestamp}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Save trial parameters
        with open(os.path.join(trial_dir, 'trial_params.json'), 'w') as f:
            json.dump(trial_params, f, indent=2)
        
        try:
            # Initialize model and optimizer with trial parameters
            model = TShirtPoseEstimator(device=config.device)
            
            optimizer = torch.optim.Adam([
                {'params': model.global_orient, 'lr': trial_params['lr_global_orient']},
                {'params': model.body_pose, 'lr': trial_params['lr_body_pose']},
                {'params': model.betas, 'lr': trial_params['lr_betas']}
            ])
            
            # Load target mesh
            target_mesh = model.load_tshirt_mesh(config.tshirt_mesh_path).to(config.device)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            for iteration in range(trial_params['num_iterations']):
                optimizer.zero_grad()
                loss, metrics = model(target_mesh)
                
                # Compute weighted loss
                weighted_loss = (metrics['silhouette_loss'] + 
                               trial_params['surface_loss_weight'] * metrics['surface_loss'] +
                               trial_params['pose_prior_weight'] * metrics['pose_prior_loss'])
                
                weighted_loss.backward()
                optimizer.step()
                
                # Early stopping logic
                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Trial {trial.number} stopped early at iteration {iteration}")
                    break
                
                # Report intermediate values
                if iteration % 10 == 0:
                    trial.report(weighted_loss, iteration)
                    
                    # Handle pruning (stop unpromising trials)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return best_loss.item()
        
        except (RuntimeError, optuna.TrialPruned) as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()
    
    # Create study
    study = optuna.create_study(
        study_name="smpl_pose_optimization",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_config.base_dir, base_config.results_dir, 
                              f"hyperopt_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best parameters
    best_params = {
        'parameters': study.best_params,
        'value': study.best_value,
        'trial': study.best_trial.number
    }
    with open(os.path.join(results_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save all trials history
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                'number': trial.number,
                'params': trial.params,
                'value': trial.value
            })
    
    with open(os.path.join(results_dir, 'all_trials.json'), 'w') as f:
        json.dump(trials_data, f, indent=2)
    
    # Print summary
    print("\nHyperparameter Optimization Summary:")
    print(f"Number of completed trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best loss: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(results_dir, 'optimization_history.html'))
        
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(results_dir, 'parallel_coordinate.html'))
    except Exception as e:
        print(f"Could not generate plots: {str(e)}")
    
    return study

# Usage example:
if __name__ == "__main__":
    base_config = Config()  # Your base configuration
    
    # Run hyperparameter tuning
    study = run_hyperparameter_tuning(
        base_config=base_config,
        n_trials=50  # Number of different parameter combinations to try
    )