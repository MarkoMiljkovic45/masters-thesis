import torch
import torch.nn as nn
from typing import Tuple, List

class FinancialVAE(nn.Module):
    
    def __init__(
        self, 
        n_stock: int,
        window: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.25,
        activation: str = 'elu'
    ):
        super(FinancialVAE, self).__init__()
        
        self.input_dim = n_stock * window + window
        self.window = window

        self.activation = self._get_activation(activation)
        self.hidden = self._build_hidden(hidden_dims, dropout)
        
        self.alpha_embedder = nn.Linear(hidden_dims[-1], n_stock)
        self.beta_embedder  = nn.Linear(hidden_dims[-1], n_stock)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }
        return activations.get(activation.lower(), nn.ELU())
    
    def _build_hidden(self, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers = []
        prev_dim = self.input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor, r_market: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.hidden(X)
        
        alpha = self.alpha_embed(h)
        beta  = self.beta_embed(h)

        r_market = r_market.unsqueeze(1)
        reconstruction = alpha + beta * r_market
        
        return reconstruction, alpha, beta