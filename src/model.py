import math

import lightning as L
import torch
from torch import Tensor
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError

from .common import inverse_returns_covariance, ols
from .plots import scatter_plot, estimation_plots, estimation_scatter, hist_plot


class MultivariateGaussianNLLLoss(Metric):
    """
    Custom Multivariate Gaussian Negative Log Likelihood Loss as a TorchMetric.
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    log2pi = math.log(2 * math.pi)

    def __init__(self):
        super().__init__()

        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mean: torch.Tensor, inv_cov: torch.Tensor, target: torch.Tensor):
        """
        Compute the negative log likelihood.

        Args:
            mean   : Predicted mean (K, 1)
            inv_cov: Inverse covariance matrix (K, K)
            target : Target samples (K, n)
        """
        loss = MultivariateGaussianNLLLoss.multivariate_gaussian_nll_loss(mean, inv_cov, target)

        self.loss_sum += loss
        self.total += 1

    def compute(self):
        return self.loss_sum / self.total

    @staticmethod
    def multivariate_gaussian_nll_loss(mean: torch.Tensor, inv_cov: torch.Tensor, target: torch.Tensor):
        """
                Compute the negative log likelihood.

                Args:
                    mean   : Predicted mean (K, 1)
                    inv_cov: Inverse covariance matrix (K, K)
                    target : Target samples (K, n)
                """
        K, n = target.shape

        mean_diff = target - mean  # (K, n)

        quadratic = torch.trace(
            torch.matmul(
                torch.matmul(mean_diff.T, inv_cov),
                mean_diff
            )
        )

        log_det = torch.logdet(inv_cov)
        loss = n * (K * MultivariateGaussianNLLLoss.log2pi - log_det) + quadratic
        loss = loss * 0.5

        return loss


class FinancialLstm(L.LightningModule):
    """
    Lightning module base for training FinancialLSTM
    """

    def __init__(
            self,
            input_size: int = 3,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            lr_patience: int = 4
    ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.alpha_head = torch.nn.Linear(hidden_size, 1)
        self.beta_head = torch.nn.Linear(hidden_size, 1)

        self.test_mae = MeanAbsoluteError()
        self.test_nll = MultivariateGaussianNLLLoss()
        self.test_results = None

    def forward(self, lstm_input: Tensor) -> tuple[Tensor, Tensor]:
        lstm_out, _ = self.lstm(lstm_input)
        final_hidden = lstm_out[:, -1, :]

        alpha = self.alpha_head(final_hidden)
        beta = self.beta_head(final_hidden)
        return alpha, beta

    def on_test_start(self):
        self.test_results = {
            'test_loss': {
                'mae': None,
                'nll': None,
            }
        }

    def test_step(self, batch, batch_idx):
        context = batch[0].flatten(0, 1)  # (n_stocks, lookback_window, features)
        target = batch[1].flatten(0, 1)  # (n_stocks, target_window, features)
        factor = batch[2].flatten(0, 1)  # (2)
        inv_psi = batch[3].flatten(0, 1)  # (n_stocks,)

        r_target = target[:, :, 0]  # (n_stocks, target_window)
        r_market_target = target[:, :, 1]  # (n_stocks, target_window)
        inv_psi = torch.diag(inv_psi)  # (n_stocks, n_stocks)
        f_mean = factor[0]
        f_var = factor[1]

        # MSE
        alpha_model, beta_model = self(context)                         # 2x(n_stocks, 1)
        r_pred_model = alpha_model + beta_model * r_market_target

        # NLL
        r_mean = alpha_model + beta_model * f_mean
        r_inv_cov = inverse_returns_covariance(beta_model, inv_psi, f_var)

        self.test_mae.update(r_pred_model, r_target)
        self.test_nll.update(r_mean, r_inv_cov, r_target)

    def on_test_end(self):
        test_mae_loss = self.test_mae.compute()
        test_nll_loss = self.test_nll.compute()

        self.test_results['test_loss']['mae'] = test_mae_loss.cpu().item()
        self.test_results['test_loss']['nll'] = test_nll_loss.cpu().item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.hparams.lr_patience
        )

        frequency = getattr(self.trainer, 'check_val_every_n_epoch', 1) if hasattr(self, 'trainer') else 1

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/total/val",
                "interval": "epoch",
                "frequency": frequency,
            }
        }


class FinancialLstmMse(FinancialLstm):

    def __init__(
            self,
            input_size: int = 3,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            lr_patience: int = 4
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, learning_rate, weight_decay, lr_patience)
        self.save_hyperparameters()
        self.train_metric = MeanSquaredError()
        self.val_metric = MeanSquaredError()

    def _shared_step(self, batch, batch_idx):
        context = batch[0].flatten(0, 1)  # (n_stocks, lookback_window, features)
        target = batch[1].flatten(0, 1)  # (n_stocks, target_window, features)

        r_target = target[:, :, 0]  # (n_stocks, target_window)
        r_market_target = target[:, :, 1]  # (n_stocks, target_window)

        alpha, beta = self(context)
        r_pred = alpha + beta * r_market_target

        return r_pred, r_target

    def training_step(self, batch, batch_idx):
        r_pred, r_target = self._shared_step(batch, batch_idx)
        loss = self.train_metric(r_pred, r_target)
        self.log('loss/mse/train', self.train_metric, on_step=False, on_epoch=True)
        self.log('loss/total/train', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        r_pred, r_target = self._shared_step(batch, batch_idx)
        loss = self.val_metric(r_pred, r_target)
        self.log('loss/mse/val', self.val_metric, on_step=False, on_epoch=True)
        self.log('loss/total/val', loss, on_step=False, on_epoch=True)


class FinancialLstmNll(FinancialLstm):

    def __init__(
            self,
            input_size: int = 3,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            lr_patience: int = 4
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, learning_rate, weight_decay, lr_patience)
        self.save_hyperparameters()
        self.train_metric = MultivariateGaussianNLLLoss()
        self.val_metric = MultivariateGaussianNLLLoss()

    def _shared_step(self, batch, batch_idx):
        context = batch[0].flatten(0, 1)  # (n_stocks, lookback_window, features)
        target = batch[1].flatten(0, 1)  # (n_stocks, target_window, features)
        factor = batch[2].flatten(0, 1)  # (2)
        inv_psi = batch[3].flatten(0, 1)  # (n_stocks,)

        r_target = target[:, :, 0]  # (n_stocks, target_window)
        inv_psi = torch.diag(inv_psi)  # (n_stocks, n_stocks)
        f_mean = factor[0]
        f_var = factor[1]

        alpha, beta = self(context)
        r_mean = alpha + beta * f_mean
        r_inv_cov = inverse_returns_covariance(beta, inv_psi, f_var)

        return r_mean, r_inv_cov, r_target

    def training_step(self, batch, batch_idx):
        r_mean, r_inv_cov, r_target = self._shared_step(batch, batch_idx)
        loss = self.train_metric(r_mean, r_inv_cov, r_target)
        self.log('loss/nll/train', self.train_metric, on_step=False, on_epoch=True)
        self.log('loss/total/train', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        r_mean, r_inv_cov, r_target = self._shared_step(batch, batch_idx)
        loss = self.val_metric(r_mean, r_inv_cov, r_target)
        self.log('loss/nll/val', self.val_metric, on_step=False, on_epoch=True)
        self.log('loss/total/val', loss, on_step=False, on_epoch=True)


class FinancialLstmCombined(FinancialLstm):

    def __init__(
            self,
            input_size: int = 3,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            lr_patience: int = 4,
            mse_weight: float = 1e2,
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, learning_rate, weight_decay, lr_patience)
        self.save_hyperparameters()

        self.train_mse = MeanSquaredError()
        self.train_nll = MultivariateGaussianNLLLoss()

        self.val_mse = MeanSquaredError()
        self.val_nll = MultivariateGaussianNLLLoss()

    def _shared_step(self, batch, batch_idx):
        context = batch[0].flatten(0, 1)  # (n_stocks, lookback_window, features)
        target = batch[1].flatten(0, 1)  # (n_stocks, target_window, features)
        factor = batch[2].flatten(0, 1)  # (2)
        inv_psi = batch[3].flatten(0, 1)  # (n_stocks,)

        r_target = target[:, :, 0]  # (n_stocks, target_window)
        r_market_target = target[:, :, 1]  # (n_stocks, target_window)
        inv_psi = torch.diag(inv_psi)  # (n_stocks, n_stocks)
        f_mean = factor[0]
        f_var = factor[1]

        # MSE
        alpha, beta = self(context)  # 2x(n_stocks, 1)
        r_pred = alpha + beta * r_market_target

        # NLL
        r_mean = alpha + beta * f_mean
        r_inv_cov = inverse_returns_covariance(beta, inv_psi, f_var)

        return r_pred, r_mean, r_inv_cov, r_target

    def training_step(self, batch, batch_idx):
        r_pred, r_mean, r_inv_cov, r_target = self._shared_step(batch, batch_idx)

        train_mse_loss = self.train_mse(r_pred, r_target)
        train_nll_loss = self.train_nll(r_mean, r_inv_cov, r_target)

        self.log('loss/mse/train', self.train_mse, on_step=False, on_epoch=True)
        self.log('loss/nll/train', self.train_nll, on_step=False, on_epoch=True)

        loss = train_nll_loss + self.hparams.mse_weight * train_mse_loss
        self.log('loss/total/train', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        r_pred, r_mean, r_inv_cov, r_target = self._shared_step(batch, batch_idx)

        val_mse_loss = self.val_mse(r_pred, r_target)
        val_nll_loss = self.val_nll(r_mean, r_inv_cov, r_target)

        self.log('loss/mse/val', self.val_mse, on_step=False, on_epoch=True)
        self.log('loss/nll/val', self.val_nll, on_step=False, on_epoch=True)

        loss = val_nll_loss + self.hparams.mse_weight * val_mse_loss
        self.log('loss/total/val', loss, on_step=False, on_epoch=True)