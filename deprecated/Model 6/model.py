import torch
import torch.nn as nn
from typing import Tuple
import lightning.pytorch as pl


class FinancialLSTM(nn.Module):
    """
    Task-agnostic LSTM that estimates alpha and beta from historical data,
    then applies them to any target market return sequence.
    """
    
    def __init__(
        self, 
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(FinancialLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM processes historical data (stock returns, market returns, interaction)
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Heads to estimate alpha and beta
        self.alpha_head = nn.Linear(hidden_size, 1)
        self.beta_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        stock_returns_context: torch.Tensor,    # (batch, lookback_window)
        r_market_context: torch.Tensor,         # (batch, lookback_window)
        r_market_target: torch.Tensor           # (batch, target_window)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        interaction = stock_returns_context * r_market_context
        lstm_input = torch.stack([stock_returns_context, r_market_context, interaction], dim=-1)
        

        lstm_out, _ = self.lstm(lstm_input)
        final_hidden = lstm_out[:, -1, :]
        
        alpha = self.alpha_head(final_hidden).squeeze(-1)  # (batch,)
        beta = self.beta_head(final_hidden).squeeze(-1)    # (batch,)
        
        predictions = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * r_market_target
        
        return predictions, alpha, beta


class FinancialLSTMModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for FinancialLSTM.
    Task-agnostic: handles both reconstruction and prediction based on data provided.
    """
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.model = FinancialLSTM(
            self.hparams.HIDDEN_SIZE,
            self.hparams.NUM_LAYERS,
            self.hparams.DROPOUT
        )
        
    def forward(self, stock_returns_context, r_market_context, r_market_target):
        return self.model(stock_returns_context, r_market_context, r_market_target)
    
    def _shared_step(self, batch, batch_idx):
        stock_returns_context = batch['stock_returns_context']
        r_market_context = batch['r_market_context']
        r_market_target = batch['r_market_target']
        target = batch['target']
        
        predictions, alpha, beta = self(stock_returns_context, r_market_context, r_market_target)
        
        loss = nn.functional.mse_loss(predictions, target)
        
        return loss, predictions, alpha, beta
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, predictions, alpha, beta = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        
        # Log additional metrics for analysis
        return {
            'test_loss': loss,
            'predictions': predictions,
            'alpha': alpha,
            'beta': beta
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.LEARNING_RATE)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode     = 'min', 
            factor   = 0.5, 
            patience = self.hparams.NUM_EPOCHS // 2
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