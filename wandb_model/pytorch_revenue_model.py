import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb
import os
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Lock all random seeds for reproducibility
RANDOM_SEED = 42

def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True) # PyTorch Lightning seed utility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seeds locked to {seed} for reproducibility using seed_everything.")

set_random_seeds()

# Configuration (will be overridden by wandb.config during sweep)
DEFAULT_CONFIG = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 50, # Changed from 'epochs' to 'max_epochs' for Trainer
    "hidden_layer_sizes": [128, 64],
    "dropout_rate": 0.2,
    "optimizer_name": "Adam",
    "activation_function": "ReLU",
    "dataset_path": "data/scaled_revenue_30d_balanced.parquet", # Updated default dataset
    "target_column": "user_revenue_usd_30d",
    "num_workers": 2, # For DataLoader
    "early_stopping_patience": 10,
    "lr_scheduler_patience": 5
}

class RevenueDataset(Dataset):
    """PyTorch Dataset for revenue prediction."""
    def __init__(self, features, labels):
        # Ensure features are purely numeric (np.float32) before converting to tensor
        try:
            feature_values = features.values.astype(np.float32)
        except Exception as e:
            print("Error converting features to np.float32. Dumping problematic columns:")
            for col in features.columns:
                if features[col].dtype == 'object':
                    print(f"Column '{col}' has dtype object.")
            raise e
        self.features = torch.tensor(feature_values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_activation_function(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == "Sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class MLP(nn.Module):
    """Multi-Layer Perceptron for regression."""
    def __init__(self, input_size, hidden_layer_sizes, output_size=1, dropout_rate=0.2, activation_name="ReLU"):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(get_activation_function(activation_name))
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class RevenueDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, target_column: str, batch_size: int, num_workers: int, random_seed: int = RANDOM_SEED):
        super().__init__()
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.input_size = None

    def prepare_data(self):
        # download, tokenize, etc.
        # This is called only on 1 GPU in distributed training
        pass

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        # This is called on every GPU in DDP
        print(f"Loading data from {self.dataset_path} for stage: {stage}")
        df = pd.read_parquet(self.dataset_path)
        print(f"Loaded {len(df)} rows.")

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found. Available: {df.columns.tolist()}")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Ensure all feature columns are numeric
        original_X_cols = X.columns.tolist()
        X_numeric_cols = []
        for col in original_X_cols:
            try:
                # Attempt to convert to numeric, coercing errors to NaN
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].isnull().any():
                    # Fill NaNs with a placeholder, e.g., median or 0. 
                    # Using 0 for simplicity here, but median/mean might be better.
                    print(f"Warning: Column '{col}' contained NaNs after numeric conversion. Filling with 0.")
                    X[col] = X[col].fillna(0) 
                X_numeric_cols.append(col)
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to numeric: {e}. It will be dropped.")

        X = X[X_numeric_cols]
        if X.empty:
            raise ValueError("No numeric feature columns remaining after attempting conversion. Please check your dataset.")
        
        print(f"Numeric feature columns being used: {X.columns.tolist()}")
        self.input_size = X.shape[1]

        # Split: 70% train, 15% validation, 15% test
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=self.random_seed
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=(0.15/0.85), random_state=self.random_seed # 0.15 of original is (0.15/0.85) of X_train_val
        )
        
        print(f"Train size: {len(self.X_train)}, Val size: {len(self.X_val)}, Test size: {len(self.X_test)}")

    def train_dataloader(self):
        train_dataset = RevenueDataset(self.X_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        val_dataset = RevenueDataset(self.X_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        test_dataset = RevenueDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=self.num_workers > 0)

class LitRevenueModel(pl.LightningModule):
    def __init__(self, input_size: int, hidden_layer_sizes: list, dropout_rate: float, 
                 activation_function: str, optimizer_name: str, learning_rate: float, 
                 lr_scheduler_patience: int, **kwargs): # Allow kwargs for other hparams
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args to self.hparams
        
        self.model = MLP(
            input_size=self.hparams.input_size, 
            hidden_layer_sizes=self.hparams.hidden_layer_sizes,
            dropout_rate=self.hparams.dropout_rate,
            activation_name=self.hparams.activation_function
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, step_name):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        rmse = torch.sqrt(loss)
        
        self.log(f'{step_name}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{step_name}_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        elif self.hparams.optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', 
            patience=self.hparams.lr_scheduler_patience, 
            factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Metric to monitor for scheduler
                "interval": "epoch",
                "frequency": 1,
            },
        }

def train_model_lightning():
    # wandb.init() is called by WandbLogger
    # Use DEFAULT_CONFIG as fallback if not running in a sweep,
    # but wandb.config will override these when an agent runs this.
    
    # For sweeps, wandb agent passes config as command line args,
    # which pytorch_lightning.Trainer can parse, or we can merge with defaults.
    # However, wandb.init() is usually called first by the agent before this script starts.
    # So, we can rely on wandb.config being populated.
    
    # If wandb.run is None, it means we are not inside a wandb.agent run.
    # For direct script execution (testing), we might initialize wandb here for logging.
    if wandb.run is None:
        print("Not in a W&B sweep. Initializing W&B run with default config for direct testing.")
        wandb.init(project="whalehunter-direct-test", config=DEFAULT_CONFIG, job_type="train_direct")
    
    config = wandb.config # This should be populated by the W&B agent

    print(f"Starting train_model_lightning with effective W&B config: {dict(config)}")

    data_module = RevenueDataModule(
        dataset_path=config.dataset_path,
        target_column=config.target_column,
        batch_size=config.batch_size,
        num_workers=getattr(config, 'num_workers', DEFAULT_CONFIG['num_workers']) # Get from config or default
    )
    data_module.setup() # Call setup to determine input_size

    model = LitRevenueModel(
        input_size=data_module.input_size,
        hidden_layer_sizes=config.hidden_layer_sizes,
        dropout_rate=config.dropout_rate,
        activation_function=config.activation_function,
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
        lr_scheduler_patience=getattr(config, 'lr_scheduler_patience', DEFAULT_CONFIG['lr_scheduler_patience'])
    )

    wandb_logger = WandbLogger(project=wandb.run.project, name=wandb.run.name, log_model="all") # Use current run's project/name
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_rmse',
        dirpath=os.path.join(wandb.run.dir, 'checkpoints'), # Save checkpoints in W&B run directory
        filename='best-model-{epoch:02d}-{val_rmse:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_rmse',
        patience=getattr(config, 'early_stopping_patience', DEFAULT_CONFIG['early_stopping_patience']), # Get from config or default
        verbose=True, # Lightning's EarlyStopping verbose is fine
        mode='min'
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=config.max_epochs,
        accelerator="auto", # Automatically detects GPU/CPU/TPU
        devices="auto",
        deterministic=True # For reproducibility, might impact performance slightly
    )

    print("Starting trainer.fit()...")
    trainer.fit(model, datamodule=data_module)
    
    print("Starting trainer.test()...")
    # trainer.test(model, datamodule=data_module, ckpt_path='best') # Loads the best checkpoint automatically
    # Or load manually if needed and then test
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        trainer.test(model=model, datamodule=data_module, ckpt_path=best_model_path)
    else:
        print("No best model checkpoint found. Testing with last model state.")
        trainer.test(model=model, datamodule=data_module)
        
    wandb.finish() # Ensure W&B run is finished properly, especially if not using agent's auto-finish

if __name__ == "__main__":
    # This script is intended to be called by the wandb agent,
    # which sets up wandb.init() and wandb.config.
    # The agent passes hyperparameters as command-line arguments.
    train_model_lightning()

# if __name__ == "__main__":
#     # This part is primarily for testing the script directly.
#     # For sweeps, wandb.agent will call train_model().
    
#     # Example of a direct run (not part of a sweep)
#     # Ensure you have a WANDB_API_KEY environment variable set or run `wandb login`
    
#     # For a direct test run, you might want to wrap train_model()
#     # or ensure DEFAULT_CONFIG is used when not in a sweep context.
#     # train_model() # Calling this directly will start a wandb run with default_config
    
#     print("pytorch_revenue_model.py executed. To run a sweep, use a separate sweep controller script.") 