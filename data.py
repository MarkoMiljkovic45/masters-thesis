import torch
from torch.utils.data import Dataset

class SyntheticLogReturnsDataset(Dataset):
    def __init__(self, n_stock, t, window, distribution='normal', 
                 overlapping_windows=False, task='reconstruction', seed=42):
        """
        Synthetic log returns dataset for financial modeling.
        
        Args:
            n_stock: Number of stocks
            t: Number of time windows
            window: Window size (lookback period)
            distribution: 'normal' or 't' (Student's t)
            overlapping_windows: If False, windows don't overlap. If True, they do.
            task: 'reconstruction' or 'prediction'
            seed: Random seed for reproducibility
        """
        self.n_stock = n_stock
        self.t = t
        self.window = window
        self.distribution = distribution
        self.overlapping_windows = overlapping_windows
        self.task = task
        self.seed = seed
        
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic returns data with market factor structure."""
        torch.manual_seed(self.seed)
        
        # Calculate total time points needed
        if self.overlapping_windows:
            t_total = self.t + self.window - 1
        else:
            t_total = self.t * self.window
        
        # Add extra point for prediction task
        if self.task == 'prediction':
            t_total += 1
        
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
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def __len__(self):
        """Total number of samples: n_stock * t windows."""
        return self.n_stock * self.t
    
    def __getitem__(self, idx):
        """
        Get a single stock-window sample.
        
        Returns dict with keys: 'stock_returns', 'r_market', 'target'
        For prediction task, also includes 'r_market_next'
        """
        # Decode flat index into stock and window indices
        stock_idx = idx % self.n_stock
        window_idx = idx // self.n_stock
        
        # Calculate time range for this window
        if self.overlapping_windows:
            start_t = window_idx
            end_t = window_idx + self.window
        else:
            start_t = window_idx * self.window
            end_t = (window_idx + 1) * self.window
        
        # Extract stock returns and market returns for window
        stock_returns = self.returns[stock_idx, start_t:end_t]
        r_market = self.r_market[start_t:end_t]
        
        if self.task == 'reconstruction':
            return {
                'stock_returns': stock_returns,
                'r_market': r_market,
                'target': stock_returns  # Reconstruct the input
            }
        else:  # prediction
            next_return = self.returns[stock_idx, end_t]
            r_market_next = self.r_market[end_t]
            
            return {
                'stock_returns': stock_returns,
                'r_market': r_market,
                'r_market_next': r_market_next,
                'target': next_return
            }

def generate_datasets(config):
    seed_start = config['SEED']
    seed_end   = seed_start + config['N_DATASET_TRAIN']
    
    train_datasets = [
        SyntheticLogReturnsDataset(
            n_stock        = config['N_STOCK'],
            t              = config['T'],
            window         = config['WINDOW'],
            distribution   = config['DISTRIBUTION'],
            task           = config['TASK'],
            overlapping_windows=False,
            seed           = seed
        ) for seed in range(seed_start, seed_end)
    ]
    
    seed_start = seed_end
    seed_end   = seed_start + config['N_DATASET_VAL']
    
    val_datasets = [
        SyntheticLogReturnsDataset(
            n_stock        = config['N_STOCK'],
            t              = config['T'],
            window         = config['WINDOW'],
            distribution   = config['DISTRIBUTION'],
            task           = config['TASK'],
            overlapping_windows=False,
            seed           = seed
        ) for seed in range(seed_start, seed_end)
    ]
    
    seed_start = seed_end
    seed_end   = seed_start + config['N_DATASET_TEST']
    
    test_datasets = [
        SyntheticLogReturnsDataset(
            n_stock        = config['N_STOCK'],
            t              = config['T'],
            window         = config['WINDOW'],
            distribution   = config['DISTRIBUTION'],
            task           = config['TASK'],
            overlapping_windows=False,
            seed           = seed
        ) for seed in range(seed_start, seed_end)
    ]

    return train_datasets, val_datasets, test_datasets