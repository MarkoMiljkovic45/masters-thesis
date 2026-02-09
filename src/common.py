import torch
from torch import Tensor


def ols(X: torch.Tensor, y: torch.Tensor):
    """
    Calculate OLS regression coefficients (intercept and slope).

    Args:
        X: Independent variable(s)
           - Unbatched: shape (n_samples)
           - Batched  : shape (batch_size, n_samples)
        y: Dependent variable
           - Unbatched: shape (n_stock, n_samples)
           - Batched  : shape (batch_size, n_stock, n_samples)

    Returns:
        alphas: Intercept coefficients, shape (n_stock) or (batch_size, n_stock)
        betas : Slope     coefficients, shape (n_stock) or (batch_size, n_stock)
    """
    if X.dim() <= 2 and y.dim() <= 2:
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)

        b_alphas, b_betas = _batched_ols(X, y)
        alphas = b_alphas.squeeze()
        betas = b_betas.squeeze()
    else:
        alphas, betas = _batched_ols(X, y)

    return alphas, betas

def _batched_ols(X: torch.Tensor, y: torch.Tensor):
    # Add intercept column
    intercept = torch.ones_like(X)
    X = torch.stack([intercept, X], dim=-1)

    # Compute OLS: (X^T X)^{-1} X^T y
    OLS = torch.matmul(
        torch.linalg.pinv(torch.matmul(X.mT, X)),
        torch.matmul(X.mT, y.mT)
    )

    alphas = OLS[:, 0, :]
    betas  = OLS[:, 1, :]

    return alphas, betas


def inverse_returns_covariance(
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
    inv_psi_beta   = torch.matmul(inv_psi, beta)    # (n_stocks, 1)
    beta_T_inv_psi = torch.matmul(beta.T, inv_psi)  # (1, n_stocks)

    beta_T_inv_psi_beta = torch.matmul(  # (1, 1)
        beta_T_inv_psi,
        beta
    )

    denominator = (1.0 / f_var) + beta_T_inv_psi_beta  # (1, 1)
    correction = torch.matmul(inv_psi_beta, beta_T_inv_psi) / denominator  # (n_stock, n_stock)

    r_inv_cov = inv_psi - correction
    return r_inv_cov


def lookback_target_split(
        r_stocks : Tensor,
        r_market : Tensor,
        lookback_window: int,
        target_window  : int,
        stride         : int | None = None,
        prediction     : bool = True) -> tuple[Tensor, Tensor]:

    if stride is None:
        stride = lookback_window + target_window

    if prediction:
        total_window = lookback_window + target_window
    else:
        total_window = lookback_window

    stacked = torch.stack(
        torch.broadcast_tensors(r_stocks, r_market),
        dim=-1
    )

    windowed = stacked.unfold(dimension=1, size=total_window, step=stride).permute(1, 0, 3, 2)

    if prediction:
        X = windowed[:, :, :lookback_window, :]
        y = windowed[:, :, lookback_window:, :]
    else:
        target_start = lookback_window - target_window
        X = windowed[:, :, :, :]
        y = windowed[:, :, target_start:, :]

    return X, y


def add_quadratic_features(X: Tensor, interaction_only: bool = False, include_bias: bool = False) -> Tensor:
    r_stock  = X[:, :, :, 0]
    r_market = X[:, :, :, 1]

    features = [r_stock, r_market, r_stock * r_market]

    if not interaction_only:
        features.extend([
            r_stock * r_stock,
            r_market * r_market
        ])

    if include_bias:
        features.append(torch.ones_like(r_stock))

    return torch.stack(features, dim=-1)

def ols_features(target: Tensor) -> tuple[Tensor, ...]:
    r_stocks = target[:, :, :, 0]   # (n_samples, n_stocks, target_window)
    r_market = target[:, 0, :, 1]   # (n_samples, target_window)

    alphas, betas = ols(r_market, r_stocks) # 2x(n_samples, n_stocks)

    r_pred = alphas.unsqueeze(-1) + betas.unsqueeze(-1) * r_market.unsqueeze(1)
    residuals = r_stocks - r_pred

    factor = torch.stack([
        r_market.mean(dim=-1),
        r_market.var(dim=-1),
    ], dim=-1)                  # (n_samples, 2)

    psi = residuals.var(dim=-1)
    inv_psi = 1 / psi           # (n_samples, n_stock)

    return alphas, betas, factor, inv_psi