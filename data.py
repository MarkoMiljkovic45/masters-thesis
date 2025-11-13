import torch
from torch.utils.data import Dataset

class SyntheticLogReturnsDataset(Dataset):
    def __init__(self, n_stock=None, t=None, window=None, distribution='normal', include_market=False, seed=42):
        self.n_stock = n_stock
        self.t = t
        self.window = window
        self.distribution = distribution
        self.include_market = include_market
        self.seed = seed
        
        if n_stock is not None and t is not None and window is not None:
            self._generate_data()
    
    def _generate_data(self):
        torch.manual_seed(self.seed)
        
        t_total = self.t + self.window - 1
        # SPY 20-year mean = 0.0003
        # SPY 20-year std  = 0.0122
        self.r_market = self._generate_returns(t_total, self.distribution, mean=0.0003, std=0.0122)
        self.alphas = torch.randn(self.n_stock) * 0.005
        self.betas = torch.randn(self.n_stock) * 0.3 + 1.0
        self.idio_vols = torch.rand(self.n_stock) * 0.015 + 0.005
        
        r_systematic = self.alphas.unsqueeze(1) + self.betas.unsqueeze(1) * self.r_market.unsqueeze(0)
        idio_noise = self._generate_returns((self.n_stock, t_total), self.distribution)
        idio_noise = idio_noise * self.idio_vols.unsqueeze(1)
        self.returns = r_systematic + idio_noise
        
        self._precompute_windows()
    
    def _generate_returns(self, shape, distribution, mean=0.0, std=1.0):
        if distribution == 'normal':
            return torch.randn(shape) * std + mean
        elif distribution == 't':
            dist = torch.distributions.studentT.StudentT(df=5.0)
            samples = dist.sample(shape if isinstance(shape, tuple) else (shape,))
            return samples * std / (5/3)**0.5 + mean
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def _precompute_windows(self):
        windows = []
        for i in range(self.t):
            stock_windows = []
            for stock_idx in range(self.n_stock):
                stock_window = self.returns[stock_idx, i:i+self.window]
                stock_windows.append(stock_window)
            
            stock_data = torch.cat(stock_windows)
            
            if self.include_market:
                market_window = self.r_market[i:i+self.window]
                window_data = torch.cat([stock_data, market_window])
            else:
                window_data = stock_data
            
            windows.append(window_data)
        
        self.windowed_data = torch.stack(windows)
    
    def __len__(self):
        return self.t
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            X = self.windowed_data[list(indices)]
        else:
            X = self.windowed_data[idx]

        start = window - 1
        end   = -window
        step  = window
        
        y = X[start:end:step]
        
        r_market = X[-1]

        return X, y, r_market
    
    def partition_stocks(self, n_partitions):
        """Partition dataset into subsets by stock while sharing the same market returns."""
        stocks_per_partition = self.n_stock // n_partitions
        partitions = []
        
        for i in range(n_partitions):
            start_stock = i * stocks_per_partition
            end_stock = start_stock + stocks_per_partition if i < n_partitions - 1 else self.n_stock
            
            partition = SyntheticLogReturnsDataset()
            partition.n_stock = end_stock - start_stock
            partition.t = self.t
            partition.window = self.window
            partition.distribution = self.distribution
            partition.include_market = self.include_market
            partition.seed = self.seed
            
            partition.r_market = self.r_market
            partition.alphas = self.alphas[start_stock:end_stock]
            partition.betas = self.betas[start_stock:end_stock]
            partition.idio_vols = self.idio_vols[start_stock:end_stock]
            partition.returns = self.returns[start_stock:end_stock]
            
            partition._precompute_windows()
            partitions.append(partition)
        
        return partitions