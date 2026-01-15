import torch
import lightning as L

from torchmetrics import Metric, MeanSquaredError, MetricCollection


class MultivariateGaussianNLLLoss(Metric):
    """
    Custom Multivariate Gaussian Negative Log Likelihood Loss as a TorchMetric.
    This allows it to be used both as a loss function and a tracked metric.
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, n: int, K: int):
        """
            n: Number of samples (stocks)
            K: Number of variables (dimensions)
        """
        super().__init__()
        
        # Register as buffers for GPU compatibility
        self.register_buffer('n', torch.tensor(n))
        self.register_buffer('K', torch.tensor(K))
    
        # Calculate constant term: n*K*0.5*log(2*pi)
        pi = torch.acos(torch.tensor(0)) * 2
        constant = self.n * self.K * 0.5 * torch.log(2 * pi)
        self.register_buffer('C', constant)
        
        # State for metric tracking
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mean: torch.Tensor, inv_cov: torch.Tensor, target: torch.Tensor):
        """Update state with batch of predictions and targets."""
        loss = self._compute_loss(mean, inv_cov, target)
        assert not torch.isnan(loss)
        
        self.sum_loss += loss
        self.total += 1

    def compute(self):
        return self.sum_loss / self.total

    def _compute_loss(self, mean: torch.Tensor, inv_cov: torch.Tensor, target: torch.Tensor):
        """
        Compute the negative log likelihood.
        
        Args:
            mean   : Predicted mean (K, 1)
            inv_cov: Inverse covariance matrix (K, K)
            target : Target samples (K, n)
        """
        mean_diff = target - mean # (K, n)
        
        # Quadratic form: 0.5 * tr((x-μ)^T Σ^-1 (x-μ))
        quadratic = 0.5 * torch.trace(
            torch.matmul(
                torch.matmul(mean_diff.T, inv_cov),
                mean_diff
            )
        )
        
        # Log determinant term
        log_det = torch.logdet(inv_cov)
        
        # Full NLL: C - 0.5*n*log|Σ^-1| + 0.5*tr((x-μ)^T Σ^-1 (x-μ))
        loss = self.C - self.n * 0.5 * log_det + quadratic
        
        return loss
        

###########################################################################################################


class FinancialLSTM_Model(torch.nn.Module):
    def __init__(self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size  = 3,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True
        )
        
        self.alpha_head = torch.nn.Linear(hidden_size, 1)
        self.beta_head  = torch.nn.Linear(hidden_size, 1)

    def forward(self, lstm_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _  = self.lstm(lstm_input)
        final_hidden = lstm_out[:, -1, :]
        
        alpha = self.alpha_head(final_hidden)
        beta  = self.beta_head(final_hidden)
        
        return alpha, beta


class FinancialLSTM_MSE(L.LightningModule):
    """
    Lightning System for training FinancialLSTM with MSE
    """

    def __init__(
        self,
        n_stocks       : int = 6,
        lookback_window: int = 60,
        target_window  : int = 20,
        input_size     : int = 3,
        hidden_size    : int = 64,
        num_layers     : int = 2,
        dropout        : float = 0.2,
        learning_rate  : float = 1e-4,
        lr_patience    : int = 4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = FinancialLSTM_Model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Initialize metrics
        self.train_metric = MeanSquaredError()
        self.val_metric   = MeanSquaredError()
        
        # For test: MSE on reconstructed returns plus alpha/beta metrics
        self.test_metrics = MetricCollection({
            'mse_returns': MeanSquaredError(),
            'mse_alpha'  : MeanSquaredError(),
            'mse_beta'   : MeanSquaredError()
        })

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(context)

    def training_step(self, batch, batch_idx):
        context  = batch[0].squeeze()  # (n_stocks, lookback_window, features)
        target   = batch[1].squeeze()  # (n_stocks, target_window, features)
        
        # Extract components
        r_target        = target[:, :, 0]   # (n_stocks, target_window)
        r_market_target = target[:, :, 1]   # (n_stocks, target_window)
        
        # Forward pass
        alpha_pred, beta_pred = self(context)  # (n_stocks, 1)
        r_pred = alpha_pred + beta_pred * r_market_target
    
        loss = self.train_metric(r_pred, r_target)
        self.log('train_loss', self.train_metric, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        context  = batch[0].squeeze()  # (n_stocks, lookback_window, features)
        target   = batch[1].squeeze()  # (n_stocks, target_window, features)
        
        # Extract components
        r_target        = target[:, :, 0]   # (n_stocks, target_window)
        r_market_target = target[:, :, 1]   # (n_stocks, target_window)
        
        # Forward pass
        alpha_pred, beta_pred = self(context)  # (n_stocks, 1)
        r_pred = alpha_pred + beta_pred * r_market_target
    
        loss = self.val_metric(r_pred, r_target)
        self.log('val_loss', self.val_metric, on_step=True, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        """
        Test step: evaluate reconstruction MSE and alpha/beta accuracy.
        """
        # Unpack batch
        context = batch[0].squeeze(0)   # (n_stocks, lookback_window, features)
        target  = batch[1].squeeze(0)   # (n_stocks, target_window, features)
        
        # Extract components
        r_target        = target[:, :, 0]   # (n_stocks, target_window)
        r_market_target = target[:, :, 1]   # (n_stocks, target_window)
        alpha_gt        = target[:, 0, 2]   # (n_stocks)
        beta_gt         = target[:, 0, 3]   # (n_stocks)
        
        # Forward pass
        alpha_pred, beta_pred = self(context)  # (n_stocks, 1)
        r_pred = alpha_pred + beta_pred * r_market_target
        
        # Update metrics
        self.test_metrics['mse_returns'].update(r_pred, r_target)
        self.test_metrics['mse_alpha'].update(alpha_pred.squeeze(), alpha_gt)
        self.test_metrics['mse_beta'].update(beta_pred.squeeze(), beta_gt)
        

    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        # Compute and log all test metrics
        metrics = self.test_metrics.compute()
        
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.hparams.lr_patience
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch",
                "interval": "epoch",
                "frequency": 1,
            }
        }
    


class FinancialLSTM_System(L.LightningModule):
    """
    Lightning System for training FinancialLSTM with Multivariate Gaussian NLL.
    
    This system handles:
    - Training with custom MGNLL loss
    - Validation with MGNLL metric
    - Testing with MSE on reconstructed returns plus alpha/beta analysis
    """

    def __init__(
        self,
        n_stocks       : int = 6,
        lookback_window: int = 60,
        target_window  : int = 20,
        input_size     : int = 3,
        hidden_size    : int = 64,
        num_layers     : int = 2,
        dropout        : float = 0.2,
        learning_rate  : float = 1e-4,
        lr_patience    : int = 4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = FinancialLSTM_Model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Initialize metrics
        self.train_metric = MultivariateGaussianNLLLoss(n=target_window, K=n_stocks)
        self.val_metric   = MultivariateGaussianNLLLoss(n=target_window, K=n_stocks)
        
        # For test: MSE on reconstructed returns plus alpha/beta metrics
        self.test_metrics = MetricCollection({
            'mse_returns': MeanSquaredError(),
            'mse_alpha'  : MeanSquaredError(),
            'mse_beta'   : MeanSquaredError()
        })

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arg:
            context: (n_stocks, lookback_window, features)
        
        Returns:
            alpha, beta: Each (n_stocks, 1)
        """
        return self.model(context)

    def _compute_inverse_covariance(
        self,
        beta: torch.Tensor,
        inv_psi: torch.Tensor,
        f_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute inverse covariance using Woodbury matrix identity.
        
       Args:
            beta: (n_stocks, 1)
            inv_psi: (n_stocks, n_stocks) diagonal matrix
            f_var: scalar factor variance
        
        Returns:
            r_inv_cov: (n_stocks, n_stocks)
        """
        inv_psi_beta   = torch.matmul(inv_psi, beta)   # (n_stocks, 1)
        beta_T_inv_psi = torch.matmul(beta.T, inv_psi) # (1, n_stocks)
        
        beta_T_inv_psi_beta = torch.matmul(            # (1, 1)
            beta_T_inv_psi, 
            beta
        )
        
        denominator = (1.0 / f_var) + beta_T_inv_psi_beta                       # (1, 1)
        correction  = torch.matmul(inv_psi_beta, beta_T_inv_psi) / denominator  # (n_stock, n_stock)
        
        # Apply Woodbury identity
        r_inv_cov = inv_psi - correction
        
        return r_inv_cov

    def training_step(self, batch, batch_idx):
        context = batch[0].squeeze()  # (n_stocks, lookback_window, features)
        target  = batch[1].squeeze()  # (n_stocks, target_window, features)
        inv_psi = batch[2].squeeze()  # (n_stocks,)
        f_var   = batch[3].squeeze()  # scalar
        
        # Un-flatten psi
        inv_psi = torch.diag(inv_psi)  # (n_stocks, n_stocks)
        
        # Forward pass
        alpha, beta = self(context)  # (n_stocks, 1), (n_stocks, 1)
        
        r_inv_cov = self._compute_inverse_covariance(beta, inv_psi, f_var)
        r_target = target[:, :, 0]
        
        loss = self.train_metric(alpha, r_inv_cov, r_target)
        self.log('train_mgnll', self.train_metric, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        context = batch[0].squeeze()  # (n_stocks, lookback_window, features)
        target  = batch[1].squeeze()  # (n_stocks, target_window, features)
        inv_psi = batch[2].squeeze()  # (n_stocks,)
        f_var   = batch[3].squeeze()  # scalar

        # Un-flatten psi
        inv_psi = torch.diag(inv_psi)
        
        # Forward pass
        alpha, beta = self(context)  # (n_stocks, 1), (n_stocks, 1)
        
        r_inv_cov = self._compute_inverse_covariance(beta, inv_psi, f_var)
        r_target = target[:, :, 0]
        
        loss = self.val_metric(alpha, r_inv_cov, r_target)
        self.log('val_mgnll', self.val_metric, on_step=True, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        """
        Test step: evaluate reconstruction MSE and alpha/beta accuracy.
        """
        # Unpack batch
        context = batch[0].squeeze(0)   # (n_stocks, lookback_window, features)
        target  = batch[1].squeeze(0)   # (n_stocks, target_window, features)
        
        # Extract components
        r_target        = target[:, :, 0]   # (n_stocks, target_window)
        r_market_target = target[:, :, 1]   # (n_stocks, target_window)
        alpha_gt        = target[:, 0, 2]   # (n_stocks)
        beta_gt         = target[:, 0, 3]   # (n_stocks)
        
        # Forward pass
        alpha_pred, beta_pred = self(context)  # (n_stocks, 1)
        r_pred = alpha_pred + beta_pred * r_market_target
        
        # Update metrics
        self.test_metrics['mse_returns'].update(r_pred, r_target)
        self.test_metrics['mse_alpha'].update(alpha_pred.squeeze(), alpha_gt)
        self.test_metrics['mse_beta'].update(beta_pred.squeeze(), beta_gt)
        

    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        # Compute and log all test metrics
        metrics = self.test_metrics.compute()
        
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.hparams.lr_patience
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mgnll",
                "interval": "epoch",
                "frequency": 1,
            }
        }


class FinancialLSTM_MinVar(L.LightningModule):
    """
    Lightning System for training FinancialLSTM for MinVar optimization
    """

    def __init__(
        self,
        input_size     : int = 3,
        hidden_size    : int = 64,
        num_layers     : int = 2,
        dropout        : float = 0.2,
        learning_rate  : float = 1e-4,
        lr_patience    : int = 4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = FinancialLSTM_Model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # For test: MSE on reconstructed returns plus alpha/beta metrics
        self.test_metrics = MetricCollection({
            'mse_returns': MeanSquaredError(),
            'mse_alpha'  : MeanSquaredError(),
            'mse_beta'   : MeanSquaredError()
        })

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(context)

    def _compute_inverse_covariance(
        self,
        beta: torch.Tensor,
        inv_psi: torch.Tensor,
        f_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute inverse covariance using Woodbury matrix identity.
        
       Args:
            beta: (n_stocks, 1)
            inv_psi: (n_stocks, n_stocks) diagonal matrix
            f_var: scalar factor variance
        
        Returns:
            r_inv_cov: (n_stocks, n_stocks)
        """
        inv_psi_beta   = torch.matmul(inv_psi, beta)   # (n_stocks, 1)
        beta_T_inv_psi = torch.matmul(beta.T, inv_psi) # (1, n_stocks)
        
        beta_T_inv_psi_beta = torch.matmul(            # (1, 1)
            beta_T_inv_psi, 
            beta
        )
        
        denominator = (1.0 / f_var) + beta_T_inv_psi_beta                       # (1, 1)
        correction  = torch.matmul(inv_psi_beta, beta_T_inv_psi) / denominator  # (n_stock, n_stock)
        
        # Apply Woodbury identity
        r_inv_cov = inv_psi - correction
        
        return r_inv_cov

    def _shared_step(self, batch):
        context      = batch[0].squeeze()  # (n_stocks, lookback_window, features)
        target       = batch[1].squeeze()  # (n_stocks, target_window, features)
        inv_psi      = batch[2].squeeze()  # (n_stocks,)
        f_var        = batch[3].squeeze()  # scalar
        r_target_cov = batch[4].squeeze()  # (n_stock, n_stock)
        sigma_target = batch[5].squeeze()  # scalar
        
        # Un-flatten psi
        inv_psi = torch.diag(inv_psi)  # (n_stocks, n_stocks)
        
        # Forward pass
        alpha, beta = self(context)  # (n_stocks, 1), (n_stocks, 1)
        
        r_inv_cov = self._compute_inverse_covariance(beta, inv_psi, f_var)
        w_pred = r_inv_cov.sum(dim=1, keepdim=True) / r_inv_cov.sum()
        
        sigma_pred = torch.matmul(
            torch.matmul(w_pred.T, r_target_cov),
            w_pred
        ).squeeze()
        
        loss = sigma_pred - sigma_target
        return loss
        

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step: evaluate reconstruction MSE and alpha/beta accuracy.
        """
        # Unpack batch
        context = batch[0].squeeze(0)   # (n_stocks, lookback_window, features)
        target  = batch[1].squeeze(0)   # (n_stocks, target_window, features)
        
        # Extract components
        r_target        = target[:, :, 0]   # (n_stocks, target_window)
        r_market_target = target[:, :, 1]   # (n_stocks, target_window)
        alpha_gt        = target[:, 0, 2]   # (n_stocks)
        beta_gt         = target[:, 0, 3]   # (n_stocks)
        
        # Forward pass
        alpha_pred, beta_pred = self(context)  # (n_stocks, 1)
        r_pred = alpha_pred + beta_pred * r_market_target
        
        # Update metrics
        self.test_metrics['mse_returns'].update(r_pred, r_target)
        self.test_metrics['mse_alpha'].update(alpha_pred.squeeze(), alpha_gt)
        self.test_metrics['mse_beta'].update(beta_pred.squeeze(), beta_gt)
        

    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        # Compute and log all test metrics
        metrics = self.test_metrics.compute()
        
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.hparams.lr_patience
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }