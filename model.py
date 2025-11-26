import torch
import torch.nn as nn
from typing import Tuple, Optional

class FinancialLSTM(nn.Module):
    
    def __init__(
        self, 
        window: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = 'reconstruction'
    ):
        super(FinancialLSTM, self).__init__()
        
        self.window = window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task = task
        
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.alpha_head = nn.Linear(hidden_size, 1)
        self.beta_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        stock_returns: torch.Tensor,
        r_market: torch.Tensor,
        r_market_next: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        interaction = stock_returns * r_market
        lstm_input = torch.stack([stock_returns, r_market, interaction], dim=-1)
        
        lstm_out, _ = self.lstm(lstm_input)
        final_hidden = lstm_out[:, -1, :]
        
        alpha = self.alpha_head(final_hidden).squeeze(-1)
        beta = self.beta_head(final_hidden).squeeze(-1)
        
        if self.task == 'reconstruction':
            reconstruction = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * r_market
        else:
            if r_market_next is None:
                raise ValueError("r_market_next required for prediction task")
            reconstruction = alpha + beta * r_market_next.squeeze(-1)
        
        return reconstruction, alpha, beta


import lightning.pytorch as pl

class FinancialLSTMModule(pl.LightningModule):
    
    def __init__(
        self,
        window: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = 'reconstruction',
        lr: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = FinancialLSTM(window, hidden_size, num_layers, dropout, task)
        self.task = task
        
    def forward(self, stock_returns, r_market, r_market_next=None):
        return self.model(stock_returns, r_market, r_market_next)
    
    def _shared_step(self, batch, batch_idx):
        stock_returns = batch['stock_returns']
        r_market = batch['r_market']
        target = batch['target']
        
        if self.task == 'prediction':
            r_market_next = batch['r_market_next']
            reconstruction, alpha, beta = self(stock_returns, r_market, r_market_next)
        else:
            reconstruction, alpha, beta = self(stock_returns, r_market)
        
        loss = nn.functional.mse_loss(reconstruction, target)
        return loss, reconstruction, alpha, beta
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)