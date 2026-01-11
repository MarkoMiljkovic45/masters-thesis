import torch

from dataclasses import dataclass, asdict
from collections.abc import Callable
from torch.utils.data import Dataset

type Datasets = tuple[list[Dataset], list[Dataset], list[Dataset]]

@dataclass(kw_only=True)
class DistParams:
    mean: float = 0
    std : float = 1

def sample_normal(shape, params: DistParams) -> torch.Tensor:
    return torch.randn(shape) * params.std + params.mean

@dataclass(kw_only=True)
class DatasetConfig:
    n_stock        : int
    n_windows      : int
    lookback_window: int
    target_window  : int
    sample_dist    : Callable = sample_normal
    prediction_task: bool = False
    seed           : int = None

@dataclass(kw_only=True)
class DatasetsConfig(DatasetConfig):
    n_dataset_train: int
    n_dataset_val  : int
    n_dataset_test : int

class SLRDatasetBase(Dataset):
    """
    Synthetic log returns dataset class.
    """
    
    def __init__(self, dataset_cfg: DatasetConfig):
        for key, value in asdict(dataset_cfg).items():
            setattr(self, key, value)

        if not self.prediction_task and self.target_window > self.lookback_window:
            raise ValueError('target window must be <= lookback window for reconstruciton task!')

        self._generate_data()

    def __len__(self):
        return self.n_windows

    def _generate_data(self):
        """Generate synthetic returns data with market factor structure."""

        if self.seed:
            torch.manual_seed(self.seed)
        
        context_points = self.n_windows * self.lookback_window
        
        if self.prediction_task:
            target_points = self.n_windows * self.target_window
            t_total = context_points + target_points
        else:
            t_total = context_points
        
        # Generate market returns (SPY 20-year statistics:   mean=0.0003, std=0.0122)
        self.r_market = self.sample_dist(t_total, DistParams(mean=0.0003, std=0.0122))
        
        # Generate stock-specific parameters
        self.alphas    = torch.rand(self.n_stock) * 0.004 - 0.002   # alpha [-0.002, 0.002]
        self.betas     = torch.rand(self.n_stock) * 3.4 - 1.7       # beta  [  -1.7, 1.7  ]
        self.idio_vols = torch.rand(self.n_stock) * 0.01 + 0.005    # idio  [ 0.005, 0.015 ]
        
        # Generate stock returns: r_i,t = alpha_i + beta_i * r_market,t + epsilon_i,t
        systematic    = self.alphas.unsqueeze(1) + self.betas.unsqueeze(1) * self.r_market.unsqueeze(0)
        idiosyncratic = self.sample_dist((self.n_stock, t_total), DistParams()) * self.idio_vols.unsqueeze(1)
        self.returns  = systematic + idiosyncratic  # Shape: (n_stock, t_total)

    def __getitem__(self, idx):
        """
        Get a single window sample.
        
        Returns dict with keys:
            - 'r_context'       : Historical stock returns
            - 'r_target'        : Target stock returns to predict/reconstruct
            - 'r_market_context': Historical market returns for LSTM input
            - 'r_market_target' : Market returns for output calculation
        """
        if idx >= len(self):
            raise IndexError(f'Index {idx} is out of bounds for dataset of length={len(self)}')

        if self.prediction_task:
            window_step = self.lookback_window + self.target_window
        else:
            window_step = self.lookback_window
        
        context_start = idx * window_step
        context_end   = context_start + self.lookback_window
        
        # Calculate time range for target window
        if self.prediction_task:
            target_start = context_end
            target_end   = target_start + self.target_window
        else:
            target_start = context_end - self.target_window
            target_end   = context_end
        
        # Extract data
        r_context = self.returns[:, context_start:context_end]
        r_target  = self.returns[:, target_start:target_end]
        
        r_market_context = self.r_market[context_start:context_end]
        r_market_target  = self.r_market[target_start:target_end]
        
        return {
            'r_context': r_context,
            'r_target': r_target,
            'r_market_context': r_market_context,
            'r_market_target': r_market_target
        }

    @staticmethod
    def generate_datasets(datasets_cfg: DatasetsConfig, DatasetClass: Dataset) -> Datasets:
        """
        Generate train, val, test datasets
        """
        n_datasets = datasets_cfg.n_dataset_train + datasets_cfg.n_dataset_val + datasets_cfg.n_dataset_test

        datasets = [DatasetClass(datasets_cfg) for k in range(n_datasets)]
    
        i = datasets_cfg.n_dataset_train
        j = i + datasets_cfg.n_dataset_val
        k = j + datasets_cfg.n_dataset_test
        
        train_datasets = datasets[0:i]
        val_datasets   = datasets[i:j]
        test_datasets  = datasets[j:k]
    
        return train_datasets, val_datasets, test_datasets

class SLRDataset(SLRDatasetBase):
    """
    Synthetic log returns dataset.
    """
    
    def __init__(self, dataset_cfg: DatasetConfig):
        super().__init__(dataset_cfg)

    @staticmethod
    def generate_datasets(datasets_cfg: DatasetsConfig) -> Datasets:
        return SLRDatasetBase.generate_datasets(datasets_cfg, SLRDataset)
        

def add_intercept(X: torch.Tensor) -> torch.Tensor:
    X_intercept = torch.ones(len(X), 2)
    X_intercept[:, 1] = X
    return X_intercept

class SLRDatasetOLS(SLRDatasetBase):
    """
    Synthetic log returns dataset with additional OLS betas for each sample.
    """
    
    def __init__(self, dataset_cfg: DatasetConfig):
        super().__init__(dataset_cfg)
        self.factor_var = torch.ones(self.n_windows) * torch.nan
        self.inv_psi = torch.ones(self.n_windows, self.n_stock, self.n_stock) * torch.nan

    def __getitem__(self, idx):
        """
        Get a single window sample.
        
        Returns dict with keys:
            - 'r_context'       : Historical stock returns
            - 'r_target'        : Target stock returns to predict/reconstruct
            - 'r_market_context': Historical market returns for LSTM input
            - 'r_market_target' : Market returns for output calculation
            - 'factor_var'      : Market factor variance
            - 'inv_psi'         : Inverse of residual covariance matrix
        """
        item = super().__getitem__(idx)
        
        if torch.isnan(self.factor_var[idx]):
            X = item['r_market_context']
            Y = item['r_context'].T
            
            X = add_intercept(X)
            
            # OLS
            OLS = torch.linalg.pinv(X.T @ X) @ (X.T @ Y)
            OLS_beta = OLS[1, :].unsqueeze(1)

            r_cov = item['r_context'].cov()
            f_var = item['r_market_context'].var()
    
            psi = torch.diag(r_cov - OLS_beta @ OLS_beta.T * f_var)
            inv_psi = torch.diag(1 / psi)

            self.factor_var[idx] = f_var
            self.inv_psi[idx]    = inv_psi

        item['factor_var'] = self.factor_var[idx]
        item['inv_psi']    = self.inv_psi[idx]
        return item

    @staticmethod
    def generate_datasets(datasets_cfg: DatasetsConfig) -> Datasets:
        return SLRDatasetBase.generate_datasets(datasets_cfg, SLRDatasetOLS)


"""""""""""""""""""""
WORK IN PROGRESS
"""""""""""""""""""""
def sample_t(shape, params: DistParams) -> torch.Tensor:

    # !!! WORK IN PROGRESS, needs DistParams implementation !!! #
    
    dist = torch.distributions.studentT.StudentT(df=params.df)
    samples = dist.sample(shape if isinstance(shape, tuple) else (shape,))
    
    # Adjust for Student's t variance: Var(T_df) = df/(df-2) for df > 2
    scale_factor = (params.df / (params.df - 2)) ** 0.5
    return samples * (std / scale_factor) + mean