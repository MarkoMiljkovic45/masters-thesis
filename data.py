import torch
import lightning as L

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


class SyntheticLogReturnsDGP():
    """
    Synthetic log returns data generating process (DGP).

    Creates dataset with synthetic daily log returns for n_stocks that
    depend on the market returns and adds idiosyncratic noise (Single factor model)

    Each sample returns tensors: context and target:

        Context: shape=(n_stock, lookback_window, features), features by index:
            0 - r_stock
            1 - r_market
            2 - interaction r_stock * r_market
    
        Target: shape=(n_stock, target_window, features), features by index:
            0 - r_stock
            1 - r_market
            2 - ground truth stock alpha
            3 - ground truth stock beta
    """
    
    def __init__(self,
        n_windows      : int = 100,
        n_stock        : int = 6,
        lookback_window: int = 60,
        target_window  : int = 20,
        prediction_task: bool = True
    ):
        self.n_windows       = n_windows
        self.n_stock         = n_stock
        self.lookback_window = lookback_window
        self.target_window   = target_window
        self.prediction_task = prediction_task

        if not self.prediction_task and self.target_window > self.lookback_window:
            raise ValueError('target window must be <= lookback window for reconstruciton task')

    def _generate_data(self):
        if self.prediction_task:
            t_total = self.n_windows * (self.lookback_window + self.target_window)
        else:
            t_total = self.n_windows * self.lookback_window
        
        # Generate market returns (SPY 20-year daily statistics: mean=0.0003, std=0.0122)
        r_market = torch.randn(1, t_total) * 0.0122 + 0.0003
        
        # Generate stock-specific parameters
        alphas    = torch.rand(self.n_stock, 1) * 0.004 - 0.002   # alpha [-0.002, 0.002]
        betas     = torch.rand(self.n_stock, 1) * 3.4 - 1.7       # beta  [  -1.7, 1.7  ]
        idio_vols = torch.rand(self.n_stock, 1) * 0.01 + 0.005    # idio  [ 0.005, 0.015 ]
        
        # Generate stock returns: r_i,t = alpha_i + beta_i * r_market,t + epsilon_i,t
        r_systematic    = alphas + betas * r_market
        r_idiosyncratic = torch.randn(self.n_stock, t_total) * idio_vols
        
        r_stocks  = r_systematic + r_idiosyncratic

        return r_stocks, r_market, alphas, betas

    def _transform_data(self, data):
        r_stocks, r_market, alphas, betas = data
        
        # Broadcast, add features and stack
        broadcast = torch.broadcast_tensors(
            r_stocks,
            r_market,
            r_stocks * r_market, # Interaction
            alphas,
            betas
        )
        
        stack = torch.stack(broadcast, dim=-1)
        
        # Lookback-target window split
        lt_split = stack.split([self.lookback_window, self.target_window] * self.n_windows, dim=1)
        context = torch.stack(lt_split[::2])
        target  = torch.stack(lt_split[1::2])
        
        # Feature selection
        # 0 - r_stock
        # 1 - r_market
        # 2 - interaction r_stock * r_market
        # 3 - alpha
        # 4 - beta
        context = context[:, :, :, [0, 1, 2]]
        target  = target[:, :, :, [0, 1, 3, 4]]

        return context, target

    def generate_dataset(self, n_dgp: int = 1):
        datasets = []
        for k in range(n_dgp):
            data        = self._generate_data()
            transformed = self._transform_data(data)
            datasets.append(TensorDataset(*transformed))

        return ConcatDataset(datasets)


class SyntheticLogReturnsDGP_OLS(SyntheticLogReturnsDGP):
    """
    Synthetic log returns data generating process (DGP) with additional OLS features.

    Creates dataset with synthetic daily log returns for n_stocks that
    depend on the market returns and adds idiosyncratic noise (Single factor model)

    Each sample returns four tensors: context, target, inv_psi and f_var:

        Context: shape=(n_stock, lookback_window, features), features by index:
            0 - r_stock
            1 - r_market
            2 - interaction r_stock * r_market
    
        Target : shape=(n_stock, target_window, features), features by index:
            0 - r_stock
            1 - r_market
            2 - ground truth stock alpha
            3 - ground truth stock beta
    
        Inv_psi: shape=(n_stock) Inverse of residual covariance in flattened form

        F_var  : shape=() Factor variance
    """

    def __init__(self,
        n_windows      : int = 100,
        n_stock        : int = 6,
        lookback_window: int = 60,
        target_window  : int = 20,
        prediction_task: bool = True
    ):
        super().__init__(n_windows, n_stock, lookback_window, target_window, prediction_task)

    def _transform_data(self, data):
        context, target = super()._transform_data(data)
        
        r_stocks = context[:, :, :, 0]
        r_market = context[:, 0, :, 1]

        # Use OLS to calculate residual covariance matrix inverse
        X = r_market
        y = r_stocks
        
        # Add intercept
        intercept = torch.ones_like(X)
        X = torch.stack([intercept, X], dim=-1)

        # OLS formula
        OLS = torch.matmul(
            torch.linalg.pinv(torch.matmul(X.mT, X)),
            torch.matmul(X.mT, y.mT)
        )
        
        OLS_alphas = OLS[:, 0, :].unsqueeze(-1)
        OLS_betas  = OLS[:, 1, :].unsqueeze(-1)
        
        r_recon   = OLS_alphas + torch.matmul(OLS_betas, r_market.unsqueeze(-1).mT)
        residuals = r_stocks - r_recon

        f_var   = r_market.var(dim=-1)
        psi     = residuals.var(dim=-1)
        inv_psi = 1 / psi
        
        return context, target, inv_psi, f_var


class SLRDataModule(L.LightningDataModule):
    """
    Data module for handling SyntheticLogReturnsDGP processes
    """
    
    def __init__(self,
        dgp        : SyntheticLogReturnsDGP,
        n_dgp_train: int = 70,
        n_dgp_val  : int = 20,
        n_dgp_test : int = 10,
        batch_size : int = 1,
        seed       : int | None = None
    ):
        super().__init__()

        if seed:
            torch.manual_seed(self.seed)

        self.dgp = dgp
        self.save_hyperparameters(ignore=["dgp"])
        
    def prepare_data(self):
        # No preparation necessary
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.dgp.generate_dataset(n_dgp=self.hparams.n_dgp_train)
            self.val_dataset   = self.dgp.generate_dataset(n_dgp=self.hparams.n_dgp_val)

        if stage == "test":
            self.test_dataset = self.dgp.generate_dataset(n_dgp=self.hparams.n_dgp_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=2, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=1, shuffle=False)