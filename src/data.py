import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import lightning as L
import torch
from torch import Tensor
from torch.distributions import Normal, StudentT
from torch.utils.data import DataLoader, TensorDataset, Subset

from .common import lookback_target_split, add_quadratic_features, ols_features


class SyntheticLogReturns:
    """
    Synthetic log returns data generating process (DGP).

    Creates dataset with synthetic daily log returns for n_stocks that
    depend on the market returns and adds idiosyncratic noise (Single factor model)

    The log returns are expressed in percentages, so they are already multiplied
    with 100 (e.g. 0.45 means 0.45%)

    The generator returns four tensors:
        r_stock : shape=(n_stock, n_samples)
        r_market: shape=(n_samples)
        alphas  : shape=(n_stock)
        betas   : shape=(n_stock)
    """

    # Distribution parameters estimated from the 25_Portfolios dataset
    # NO OUTLIERS
    mkt_params  : dict = {'loc': 0.0678, 'scale': 0.5099, 'df': 5}  # Student T
    idio_params : dict = {'loc': 0.0000, 'scale': 0.3140, 'df': 5}  # Student T
    alpha_params: dict = {'loc': 0.0098, 'scale': 0.1271}           # Normal
    beta_params : dict = {'loc': 0.9444, 'scale': 0.3521}           # Normal

    """
    WITH OUTLIERS
    mkt_params  : dict = {'loc': 0.0538, 'scale': 0.6616, 'df': 5}  # Student T
    idio_params : dict = {'loc': 0.0000, 'scale': 0.3539, 'df': 5}  # Student T
    alpha_params: dict = {'loc': 0.0056, 'scale': 0.1501}           # Normal
    beta_params : dict = {'loc': 1.0046, 'scale': 0.3785}           # Normal
    """

    @staticmethod
    def generate(n_stocks: int, n_samples: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r_market = StudentT(**SyntheticLogReturns.mkt_params).sample((n_samples,))
        r_idiosyncratic = StudentT(**SyntheticLogReturns.idio_params).sample((n_stocks, n_samples))
        alphas = Normal(**SyntheticLogReturns.alpha_params).sample((n_stocks,))
        betas = Normal(**SyntheticLogReturns.beta_params).sample((n_stocks,))

        r_systematic = alphas.unsqueeze(1) + betas.unsqueeze(1) * r_market.unsqueeze(0)
        r_stocks     = r_systematic + r_idiosyncratic

        return r_stocks, r_market, alphas, betas


class FamaFrench25Portfolios:
    """
    Class used for loading 25 Portfolios Formed on Size and Book-to-Market [Daily] dataset
    provided by Eugene F. Fama and Kenneth R. French
    """

    n_samples = 26129

    skip_old_data = 3125

    ff3_filename = 'F-F_Research_Data_Factors_daily.csv'
    ff3_skip = 4
    ff3_cols = ['DATE', 'Mkt-RF', 'SMB', 'HML', 'RF']

    p25_filename = '25_Portfolios_5x5_Daily.csv'
    p25_skip = 18
    p25_cols = ['DATE', 'SMALL LoBM', 'ME1 BM2', 'ME1 BM3', 'ME1 BM4', 'SMALL HiBM',
                         'ME2 BM1', 'ME2 BM2', 'ME2 BM3', 'ME2 BM4', 'ME2 BM5',
                         'ME3 BM1', 'ME3 BM2', 'ME3 BM3', 'ME3 BM4', 'ME3 BM5',
                         'ME4 BM1', 'ME4 BM2', 'ME4 BM3', 'ME4 BM4', 'ME4 BM5',
                         'BIG LoBM', 'ME5 BM2', 'ME5 BM3', 'ME5 BM4', 'BIG HiBM']

    @staticmethod
    def load(data_dir: Path) -> tuple[Tensor, Tensor]:
        ff3_usecols = ['DATE', 'Mkt-RF', 'RF']
        ff3_types = defaultdict(lambda: np.float32, DATE=np.int32)

        ff3_df = pd.read_csv(data_dir / FamaFrench25Portfolios.ff3_filename,
            header=0,
            index_col=0,
            names=FamaFrench25Portfolios.ff3_cols,
            usecols=ff3_usecols,
            dtype=ff3_types,
            skiprows=FamaFrench25Portfolios.ff3_skip + FamaFrench25Portfolios.skip_old_data,
            nrows=FamaFrench25Portfolios.n_samples - FamaFrench25Portfolios.skip_old_data
        )

        p25_types = defaultdict(lambda: np.float32, DATE=np.int32)
        p25_df = pd.read_csv(data_dir / FamaFrench25Portfolios.p25_filename,
            header=0,
            index_col=0,
            names=FamaFrench25Portfolios.p25_cols,
            dtype=p25_types,
            skiprows=FamaFrench25Portfolios.p25_skip + FamaFrench25Portfolios.skip_old_data,
            nrows=FamaFrench25Portfolios.n_samples - FamaFrench25Portfolios.skip_old_data
        )

        # Convert arithmetic returns expressed in percents (%) to tensors
        MKT = torch.tensor(ff3_df['Mkt-RF'].values)
        RF = torch.tensor(ff3_df['RF'].values)
        P25 = torch.tensor(p25_df.values).T - RF

        # Remove rows with missing data
        mask = (P25 == -99.99).any(dim=0) | (P25 == -999).any(dim=0)
        P25 = P25[:, ~mask]
        MKT = MKT[~mask]

        # Conver to log returns expressed in percents (%)
        mkt = 100 * (torch.log(MKT + 100) - torch.log(torch.tensor(100)))
        p25 = 100 * (torch.log(P25 + 100) - torch.log(torch.tensor(100)))

        return p25, mkt


def _append_feature(y, feature):
    return torch.cat(
        [y, feature.view(-1, y.shape[1], 1, 1).expand(y.shape[0], -1, y.shape[2], 1)],
        dim=-1
    )


class FinancialLstmDataModule(L.LightningDataModule):
    """
    Data module for handling market data:
        Stock returns  (r_stocks)
        Market returns (r_market)
        Alphas         (alphas)
        Betas          (betas)
    """

    def __init__(self,
                 data_dir: Path,
                 lookback_window: int = 60,
                 target_window: int = 20,
                 stride: int = 80,
                 prediction_task: bool = True,
                 interaction_only: bool = True,
                 batch_size: int = 1
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters(ignore=['data_dir'])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if not prediction_task and target_window > lookback_window:
            raise ValueError('target window must be <= lookback window for reconstruction task')

    def _load_if_exists(self, filename):
        path = self.data_dir / filename
        return torch.load(path) if path.exists() else None

    def _compute_hparams_hash(self):
        hparams_dict = {
            'lookback_window': self.hparams.lookback_window,
            'target_window': self.hparams.target_window,
            'stride': self.hparams.stride,
            'prediction_task': self.hparams.prediction_task,
            'interaction_only': self.hparams.interaction_only,
        }
        hparams_str = json.dumps(hparams_dict, sort_keys=True)
        return hashlib.sha256(hparams_str.encode()).hexdigest()

    def prepare_data(self):
        hparams_hash = self._compute_hparams_hash()
        datasets_dir = self.data_dir / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)
        hash_file = datasets_dir / 'hparams_hash.txt'
        dataset_file = datasets_dir / 'dataset.pt'

        if hash_file.exists() and dataset_file.exists():
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()

            if stored_hash == hparams_hash:
                print("Dataset parameters unchanged, skipping data preparation")
                return

        r_stocks = torch.load(self.data_dir / 'stocks.pt')
        r_market = torch.load(self.data_dir / 'market.pt')
        alphas = self._load_if_exists('alphas.pt')
        betas = self._load_if_exists('betas.pt')

        X, y = lookback_target_split(
            r_stocks,
            r_market,
            lookback_window=self.hparams.lookback_window,
            target_window=self.hparams.target_window,
            stride=self.hparams.stride,
            prediction=self.hparams.prediction_task
        )

        X = add_quadratic_features(X, interaction_only=self.hparams.interaction_only)
        target_alphas, target_betas, target_factor, target_inv_psi = ols_features(y)

        if alphas is None or betas is None:
            alphas = target_alphas
            betas = target_betas

        y = _append_feature(y, alphas)
        y = _append_feature(y, betas)

        dataset = TensorDataset(X, y, target_factor, target_inv_psi)
        torch.save(dataset, dataset_file)
        with open(hash_file, 'w') as f:
            f.write(hparams_hash)

    def setup(self, stage: str):
        dataset_path = self.data_dir / 'datasets' / 'dataset.pt'
        dataset = torch.load(dataset_path, weights_only=False)

        n = len(dataset)
        train_end = int(0.7 * n)
        val_end = int(0.9 * n)

        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(dataset, range(0, train_end))
            self.val_dataset = Subset(dataset, range(train_end, val_end))

        if stage == 'test' or stage is None:
            self.test_dataset = Subset(dataset, range(val_end, n))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=2, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0, shuffle=False)

    def teardown(self, stage: str):
        if stage == 'cleanup':
            datasets_dir = self.data_dir / 'datasets'
            Path.unlink(datasets_dir / 'dataset.pt', missing_ok=True)
            datasets_dir.rmdir()
