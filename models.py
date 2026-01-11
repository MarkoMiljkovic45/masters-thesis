import torch
import lightning.pytorch as pl

from dataclasses import dataclass, asdict

@dataclass(kw_only=True)
class ModelConfig:
    hidden_size    : int = 64
    num_layers     : int = 2
    dropout        : float = 0.2
    learning_rate  : float = 1e-4
    num_epochs     : int = 16
    batch_size     : int = 1
    shuffle_batches: bool = True
    lr_patience    : int = 4
    loss_fn        : object = torch.nn.MSELoss()

    def dict(self):
        return asdict(self)

class MultivariateGaussianNLLLoss:
    """
    Custom Multivariate Gaussian Negative Log Likelihood Loss
    """

    def __init__(self, n: int, K: int):
        self.n = n
        self.K = K

        pi = torch.acos(torch.zeros(1)) * 2
        self.C = n * K * 0.5 * torch.log(2 * pi)
        self.ONES = torch.ones(n, 1)

    def __call__(self, mean, inv_cov, target):
        mean_diff = target - mean
        quadratic = 0.5 * (mean_diff.T @ inv_cov @ mean_diff @ self.ONES)
        log_det = torch.logdet(inv_cov)
        
        loss = self.C - self.n * 0.5 * log_det + quadratic
        return loss.mean()

class FinancialLSTM_MGNLL(pl.LightningModule):
    """
    FinancialLSTM with Multivariate Gaussian Negative Log likelihood loss

    Task-agnostic LSTM that estimates alpha and beta from historical data,
    then applies them to any target market return sequence
    """
    
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.save_hyperparameters(asdict(model_cfg))

        # LSTM processes historical data (stock returns, market returns, interaction)
        self.lstm = torch.nn.LSTM(
            input_size  = 3,
            hidden_size = self.hparams.hidden_size,
            num_layers  = self.hparams.num_layers,
            dropout     = self.hparams.dropout if self.hparams.num_layers > 1 else 0,
            batch_first = True
        )
        
        # Heads to estimate alpha and beta
        self.alpha_head = torch.nn.Linear(self.hparams.hidden_size, 1)
        self.beta_head  = torch.nn.Linear(self.hparams.hidden_size, 1)
        
        self.loss_fn = model_cfg.loss_fn
        self.test_loss_fn = torch.nn.MSELoss()
        
    def forward(self, r_context: torch.Tensor, r_market_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        interaction = r_context * r_market_context
        lstm_input  = torch.stack([r_context, r_market_context, interaction], dim=-1)
        
        lstm_out, _  = self.lstm(lstm_input)
        final_hidden = lstm_out[:, -1, :]
        
        alpha = self.alpha_head(final_hidden)
        beta  = self.beta_head(final_hidden)
        
        return alpha, beta
    
    def _shared_step(self, batch, batch_idx):
        r_context        = batch['r_context'].squeeze()
        r_market_context = batch['r_market_context'].squeeze()
        r_market_target  = batch['r_market_target'].squeeze()
        r_target         = batch['r_target'].squeeze()

        r_market_expanded = r_market_context.repeat(r_context.shape[0], 1)
        
        alpha, beta = self(r_context, r_market_expanded) # (n_stock, 1), (n_stock, 1)

        # Woodbury matrix identity
        f_var   = batch['factor_var'].squeeze()
        inv_psi = batch['inv_psi'].squeeze()
        
        r_inv_cov = inv_psi - (inv_psi @ beta @ beta.T @ inv_psi) / (1/f_var + beta.T @ inv_psi @ beta)
        
        loss = self.loss_fn(alpha, r_inv_cov, r_target)
        
        return loss, alpha, beta
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        r_context = batch['r_context']
        r_market_context = batch['r_market_context']
        r_market_target = batch['r_market_target']
        r_target = batch['r_target']
        
        alpha, beta = self(r_context, r_market_context)
        predictions = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * r_market_target

        loss = self.test_loss_fn(predictions, r_target)

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        
        # Log additional metrics for analysis
        return {
            'test_loss': loss,
            'predictions': predictions,
            'alpha': alpha,
            'beta': beta
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode     = 'min', 
            factor   = 0.5, 
            patience = self.hparams.lr_patience
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

def min_var_loss(w, r_target):
    one = torch.ones(r_target.shape[0], 1)
    
    cov_oos = r_target.cov()
    inv_cov_oos = torch.linalg.pinv(cov_oos)
    w_target = (inv_cov_oos @ one) / (one.T @ inv_cov_oos @ one)

    sigma_oos    = w.T @ cov_oos @ w
    sigma_target = w_target.T @ cov_oos @ w_target

    loss = sigma_oos - sigma_target
    return loss

class FinancialLSTM_MinVar(FinancialLSTM_MGNLL):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__(model_cfg)

    def _shared_step(self, batch, batch_idx):
        r_context        = batch['r_context'].squeeze()
        r_market_context = batch['r_market_context'].squeeze()
        r_market_target  = batch['r_market_target'].squeeze()
        r_target         = batch['r_target'].squeeze()

        r_market_expanded = r_market_context.repeat(r_context.shape[0], 1)
        
        alpha, beta = self(r_context, r_market_expanded) # (n_stock, 1), (n_stock, 1)

        # Woodbury matrix identity
        f_var   = batch['factor_var'].squeeze()
        inv_psi = batch['inv_psi'].squeeze()
        r_inv_cov = inv_psi - (inv_psi @ beta @ beta.T @ inv_psi) / (1/f_var + beta.T @ inv_psi @ beta)

        one = torch.ones(r_context.shape[0], 1)
        w = (r_inv_cov @ one) / (one.T @ r_inv_cov @ one)
        
        loss = self.loss_fn(w, r_target)
        
        return loss, (alpha, beta), w