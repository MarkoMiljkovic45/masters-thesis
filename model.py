import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class FinancialVAE(nn.Module):
    """
    Variational Autoencoder optimized for financial time series and feature data.
    
    This implementation addresses common challenges in financial modeling:
    - Fat-tailed distributions and outliers
    - Multiple market regimes
    - High-dimensional feature spaces
    - Numerical stability in latent space
    
    Use cases:
    - Dimensionality reduction for portfolio optimization
    - Anomaly detection in trading patterns
    - Synthetic scenario generation for risk analysis
    - Feature extraction from market data
    
    Args:
        input_dim     : Number of input features
        hidden_dims   : List of hidden layer dimensions (e.g., [128, 64, 32])
        latent_dim    : Dimension of latent space (recommend 2-16 for financial data)
        dropout       : Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization (recommended for larger datasets)
        activation    : Activation function ('elu', 'relu', 'selu')
    """
    
    def __init__(
        self, 
        n_stock: int,
        window: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 1,
        dropout: float = 0.25,
        activation: str = 'elu'
    ):
        super(FinancialVAE, self).__init__()
        
        self.input_dim = n_stock * window
        self.latent_dim = latent_dim
        self.window = window

        # Flatten input
        self.flatten = nn.Flatten()
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build encoder
        self.encoder = self._build_encoder(hidden_dims, dropout)
        
        # Latent space parameters (use log variance for numerical stability)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        self.decoder = nn.Linear(latent_dim, n_stock)
        self.decoder.bias.data.zero_()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Select activation function suitable for financial data."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }
        return activations.get(activation.lower(), nn.ELU())
    
    def _build_encoder(self, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        """Build encoder network with flexible architecture."""
        layers = []
        prev_dim = self.input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        return nn.Sequential(*layers)
    
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Encode
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        z = mu + eps * sigma, where sigma = exp(0.5 * logvar)
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output space.
        
        Args:
            z: Latent tensor [batch_size, latent_dim]
        
        Returns:
            reconstruction: Reconstructed output [batch_size, n_stock]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            reconstruction: Reconstructed output [batch_size, n_stock]
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        x = self.flatten(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        beta: float = 1.0,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss function = Reconstruction Loss + Beta * KL Divergence
        
        Args:
            recon_x: Reconstructed output
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence (use <1 for beta-VAE, anneal during training)
            reduction: 'mean' or 'sum' for loss reduction
        
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            kld_loss: KL divergence component
        """
        # Reconstruction loss (MSE is standard for continuous financial data)
        # Can also use MAE for more robustness to outliers

        if self.window > 1:
            x = x[:, :, -1]
        
        recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        if reduction == 'mean':
            kld = kld / x.size(0)
        
        # Total loss with beta weighting
        total_loss = recon_loss + beta * kld
        
        return total_loss, recon_loss, kld
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples from the learned latent distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            samples: Generated samples [num_samples, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Decode
            samples = self.decode(z)
        
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input (encode then decode).
        
        Args:
            x: Input tensor
        
        Returns:
            reconstruction: Reconstructed output
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = mu  # Use mean during reconstruction (no sampling)
            reconstruction = self.decode(z)
        
        return reconstruction
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (useful for dimensionality reduction).
        
        Args:
            x: Input tensor
        
        Returns:
            z: Latent representation (mean of distribution)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        
        return mu