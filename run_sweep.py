import wandb
import yaml
# Note: We don't need to import train_model directly anymore for the sweep controller,
# as the agent will execute the pytorch_revenue_model.py script which calls train_model_lightning().
from wandb_model.pytorch_revenue_model import DEFAULT_CONFIG

# 1. Define the sweep configuration
SWEEP_CONFIG = {
    'program': 'wandb_model/pytorch_revenue_model.py',
    'method': 'bayes',
    'metric': {
        'name': 'val_rmse', # Monitor val_rmse from LightningModule
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'max_epochs': { # Changed from 'epochs'
            'value': DEFAULT_CONFIG['max_epochs'] 
        },
        'hidden_layer_sizes': {
            'values': [[64, 32], [128, 64], [256, 128, 64], [128, 128], [512, 256, 128]]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'optimizer_name': {
            'values': ['Adam', 'AdamW', 'SGD']
        },
        'activation_function': {
            'values': ['ReLU', 'LeakyReLU', 'Tanh']
        },
        'dataset_path': {
            'value': DEFAULT_CONFIG['dataset_path']
        },
        'target_column': {
            'value': DEFAULT_CONFIG['target_column']
        },
        'num_workers': {
            'values': [0, 2, 4] # Example values for num_workers
        },
        'early_stopping_patience': {
            'values': [5, 10, 15] # Example values
        },
        'lr_scheduler_patience': {
            'values': [3, 5, 7] # Example values
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5, # Min epochs before stopping for hyperband
        's': 2
    }
}

if __name__ == '__main__':
    project_name = "whalehunter" 
    print(f"Initializing W&B sweep for project: {project_name} with PyTorch Lightning setup...")
    print("Sweep configuration:")
    print(yaml.dump(SWEEP_CONFIG, sort_keys=False))
    
    try:
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=project_name)
        print(f"Sweep initialized. Sweep ID: {sweep_id}")
        print(f"Run the following command to start an agent (or multiple agents):")
        print(f"wandb agent {wandb.apis.PublicApi().default_entity}/{project_name}/{sweep_id}")
        print("\nTo start training, run the printed 'wandb agent' command in your terminal.")
    except Exception as e:
        print(f"An error occurred during W&B sweep initialization: {e}")
        print("Please ensure you are logged into W&B and the project/entity names are correct.") 