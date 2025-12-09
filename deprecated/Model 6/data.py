import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Tuple


class SyntheticLogReturnsDataset(Dataset):
    """
    Synthetic log returns dataset with flexible lookback and target windows.
    
    Supports both reconstruction (target = context period) and prediction (target = future period).
    """
    
    def __init__(
        self, 
        n_stock: int,
        n_windows: int,
        lookback_window: int,
        target_window: int,
        distribution: str = 'normal',
        overlapping_windows: bool = False,
        predict_future: bool = False,
        seed: int = 42
    ):
        """
        Args:
            n_stock: Number of stocks to generate
            n_windows: Number of windows to create per stock
            lookback_window: Size of historical context window (for LSTM input)
            target_window: Size of target window (for output)
            distribution: 'normal' or 't' (Student's t-distribution)
            overlapping_windows: If False, windows don't overlap. If True, they do.
            predict_future: If False (reconstruction), target overlaps with context.
                            If True (prediction), target comes after context.
            seed: Random seed for reproducibility
        """
        self.n_stock = n_stock
        self.n_windows = n_windows
        self.lookback_window = lookback_window
        self.target_window = target_window
        self.distribution = distribution
        self.overlapping_windows = overlapping_windows
        self.predict_future = predict_future
        self.seed = seed
        
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic returns data with market factor structure."""
        torch.manual_seed(self.seed)
        
        # Calculate total time points needed
        if self.overlapping_windows:
            context_points = self.lookback_window + (self.n_windows - 1)
        else:
            context_points = self.n_windows * self.lookback_window
        
        # Add target window points
        if self.predict_future:
            if self.overlapping_windows:
                target_points = self.target_window
            else:
                target_points = self.n_windows * self.target_window
        else:
            target_points = 0
        
        t_total = context_points + target_points
        
        # Generate market returns (SPY 20-year statistics: mean=0.0003, std=0.0122)
        self.r_market = self._sample_distribution(t_total, mean=0.0003, std=0.0122)
        
        # Generate stock-specific parameters
        self.alphas = torch.randn(self.n_stock) * 0.005
        self.betas = torch.randn(self.n_stock) * 0.3 + 1.0
        self.idio_vols = torch.rand(self.n_stock) * 0.015 + 0.005
        
        # Generate stock returns: r_i,t = alpha_i + beta_i * r_market,t + epsilon_i,t
        systematic = self.alphas.unsqueeze(1) + self.betas.unsqueeze(1) * self.r_market.unsqueeze(0)
        idiosyncratic = self._sample_distribution((self.n_stock, t_total)) * self.idio_vols.unsqueeze(1)
        self.returns = systematic + idiosyncratic  # Shape: (n_stock, t_total)
    
    def _sample_distribution(self, shape, mean=0.0, std=1.0):
        """Sample from specified distribution."""
        if self.distribution == 'normal':
            return torch.randn(shape) * std + mean
        elif self.distribution == 't':
            df = 5.0
            dist = torch.distributions.studentT.StudentT(df=df)
            samples = dist.sample(shape if isinstance(shape, tuple) else (shape,))
            # Adjust for Student's t variance: Var(T_df) = df/(df-2) for df > 2
            scale_factor = (df / (df - 2)) ** 0.5
            return samples * (std / scale_factor) + mean
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def __len__(self):
        """Total number of samples: n_stock * n_windows."""
        return self.n_stock * self.n_windows
    
    def __getitem__(self, idx):
        """
        Get a single stock-window sample.
        
        Returns dict with keys:
            - 'stock_returns_context': Historical stock returns
            - 'r_market_context': Historical market returns for LSTM input
            - 'r_market_target': Market returns for output calculation
            - 'target': Target stock returns to predict/reconstruct
        """
        # Decode flat index into stock and window indices
        stock_idx = idx % self.n_stock
        window_idx = idx // self.n_stock
        
        # Calculate time range for lookback (context) window
        if self.overlapping_windows:
            context_start = window_idx
            context_end = window_idx + self.lookback_window
        else:
            context_start = window_idx * self.lookback_window
            context_end = (window_idx + 1) * self.lookback_window
        
        # Calculate time range for target window
        if self.predict_future:
            # Target comes after context
            target_start = context_end
            target_end = target_start + self.target_window
        else:
            # Reconstruction: target is same as context
            target_start = context_start
            target_end = context_end
        
        # Extract data
        stock_returns_context = self.returns[stock_idx, context_start:context_end]
        r_market_context = self.r_market[context_start:context_end]
        r_market_target = self.r_market[target_start:target_end]
        target = self.returns[stock_idx, target_start:target_end]
        
        return {
            'stock_returns_context': stock_returns_context,
            'r_market_context': r_market_context,
            'r_market_target': r_market_target,
            'target': target
        }


def generate_datasets(config):
    seed_start = config['SEED']
    seed_end   = seed_start + config['N_DATASET_TRAIN'] + config['N_DATASET_VAL'] + config['N_DATASET_TEST']

    datasets = [
        SyntheticLogReturnsDataset(
            n_stock             = config['N_STOCK'],
            n_windows           = config['N_WINDOWS'],
            lookback_window     = config['LOOKBACK_WINDOW'],
            target_window       = config['TARGET_WINDOW'],
            distribution        = config['DISTRIBUTION'],
            predict_future      = config['TASK'] == 'prediction',
            overlapping_windows = config['OVERLAPPING_WINDOWS'],
            seed                = seed
        ) for seed in range(seed_start, seed_end)
    ]

    i = config['N_DATASET_TRAIN']
    j = i + config['N_DATASET_VAL']
    k = j + config['N_DATASET_TEST']
    
    train_datasets = datasets[0:i]
    val_datasets   = datasets[i:j]
    test_datasets  = datasets[j:k]

    return train_datasets, val_datasets, test_datasets