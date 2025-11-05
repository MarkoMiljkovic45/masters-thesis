import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset

class LogReturnsDataset(Dataset):
    """
    Dataset representing log returns
    TODO
    """
    
    def __init__(self, data_file: str):
        """
        TODO
        """
        start_date = '1994-01-01'
        end_date = '2014-06-01'
        
        cols, parse_dates = ['permno', 'ret', 'date'], ['date']
        dtypes = {
            'permno': np.uint32,
            'ret': np.float32
        }
        
        raw_data = pd.read_csv(data_file, usecols=cols, dtype=dtypes, parse_dates=parse_dates)
        raw_data = raw_data.pivot(index='date', columns='permno', values='ret')
        raw_data = raw_data[start_date:end_date].dropna(axis='columns')
        EWM = raw_data.sum(axis=1) / len(raw_data)
        
        r = np.array(np.log(raw_data + 1))
        ewm = np.array(np.log(1 + EWM))
        
        self.r = torch.tensor(r, dtype=torch.float32)
        self.ewm = torch.tensor(ewm, dtype=torch.float32)

    def __len__(self):
        len(self.r)

    def __getitem__(self, idx):
        return self.r[idx]

    def get_synthetic_data(self, r_market, n_stock: int = 10):
        """
        Generate simple synthetic log returns based on single factor model (OLS) and market log returns
        
        Args:
            r_market - Required : Log market returns for the period (Array of length t)
            n_stock  - Optional : Number of stocks to generate
        
        Returns:
            s_r     - Synthetic returns
            s_alpha - Synthetic alpha
            s_beta  - Synthetic beta
            s_res   - Synthetic residuals
        """
        r   = self.r.numpy()
        ewm = self.ewm.numpy()

        # Fit
        ols = [stats.linregress(ewm, r[:, j]) for j in range(len(r.T))]
        alpha = np.array([ols_i.intercept for ols_i in ols])
        beta = np.array([ols_i.slope for ols_i in ols])
    
        residuals = r - (np.reshape(ewm, (len(ewm), 1)) @ np.reshape(beta, (1, len(beta))) + alpha)
    
        mu_res = 0                     # Residual mean is 0, so we only need residual variance
        sigma_res = np.std(residuals)  # Residual variance is specific for a stock so we will generate it using the normal distribution
        
        mu_sigma_res = np.mean(sigma_res)
        sigma_sigma_res = np.std(sigma_res)
        
        mu_a, sigma_a = np.mean(alpha), np.std(alpha)
        mu_b, sigma_b = np.mean(beta),  np.std(beta)

        t = len(r_market)

        # Generate
        s_alpha = np.random.normal(mu_a, sigma_a, n_stock)
        s_beta = np.random.normal(mu_b, sigma_b, n_stock)
        
        s_sigma_res = np.abs(np.random.normal(mu_sigma_res, sigma_sigma_res, n_stock)) # TODO Maybe use gamma distribution here
        s_res = np.array([np.random.normal(0, s_sigma_res_i, t) for s_sigma_res_i in s_sigma_res]).T
    
        # TODO Market lenght could be probelm, generate market data
        s_r = s_alpha + np.reshape(r_market, (len(r_market), 1)) @ np.reshape(s_beta, (1, len(s_beta))) + s_res
    
        return torch.tensor(s_r, dtype=torch.float32), s_alpha, s_beta, s_res


class SyntheticLogReturnsDataset(Dataset):
    """Generates synthetic log returns using a factor model with market exposure and idiosyncratic risk."""
    
    def __init__(self, n_stocks, t, window=1, seed=None):
        """
        Args:
            n_stocks: Number of stocks
            t: Number of time steps
            window: Size of historical window (1 = no windowing)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        self.n_stocks = n_stocks
        self.t = t
        self.window = window
        
        # Generate market returns (broader volatility)
        self.r_market = torch.randn(t + window - 1) * 0.015
        
        # Generate stock characteristics
        # Alpha: excess returns, typically small (can be positive or negative)
        self.alpha = torch.randn(n_stocks) * 0.002
        
        # Beta: market sensitivity, typically around 1.0 with some variation
        self.beta = torch.randn(n_stocks) * 0.3 + 1.0
        
        # Idiosyncratic volatility: varies by stock (between 0.5% and 2%)
        idio_vol = torch.rand(n_stocks) * 0.015 + 0.005
        
        # Generate returns: r = alpha + beta * r_market + idiosyncratic noise
        # Broadcasting: (t, 1) for market, (1, n_stocks) for alpha/beta
        systematic = self.alpha.unsqueeze(0) + self.beta.unsqueeze(0) * self.r_market.unsqueeze(1)
        
        # Add stock-specific noise with varying volatility
        self.idio_noise = torch.randn(t + window - 1, n_stocks) * idio_vol.unsqueeze(0)

        self.returns = systematic + self.idio_noise

        # Pre-compute windowed data for efficient slicing
        if self.window > 1:
            self.windowed_returns = torch.stack([
                self.returns[i:i + self.window].T 
                for i in range(self.t)
            ])
        
    def __len__(self):
        return self.t
    
    def __getitem__(self, idx):
        if self.window == 1:
            # Shape: (n_stocks,)
            return self.returns[idx]
        else:
            # Shape: (n_stocks, window)
            return self.windowed_returns[idx]