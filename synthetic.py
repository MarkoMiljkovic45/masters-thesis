import numpy as np
import pandas as pd
from scipy import stats

def generate_data(r, r_market, shape=None):
    """
    Generate simple synthetic log returns based on single factor model (OLS) and market log returns
    @params:
        r        - Required  : Sample log returns (Array of shape mxn)
        r_market - Required  : Log market returns for the period (Array of shape mx1)
        shape    - Optional  : Shape of synthetic data (Tuple)
    @return:
        s_r     - Synthetic returns
        s_alpha - Synthetic alpha
        s_beta  - Synthetic beta
        s_res   - Synthetic residuals
    """
    if not shape:
        shape = np.shape(r)

    ols = [stats.linregress(r_market, r[i]) for i in r]
    alpha = np.array([ols_i.intercept for ols_i in ols])
    beta = np.array([ols_i.slope for ols_i in ols])

    residuals = r - (np.reshape(r_market, (len(r_market), 1)) @ np.reshape(beta, (1, len(beta))) + alpha)

    mu_res = 0                     # Residual mean is 0, so we only need residual variance
    sigma_res = np.std(residuals)  # Residual variance is specific for a stock so we will generate it using the normal distribution
    
    mu_sigma_res = np.mean(sigma_res)
    sigma_sigma_res = np.std(sigma_res)
    
    mu_a, sigma_a = np.mean(alpha), np.std(alpha)
    mu_b, sigma_b = np.mean(beta),  np.std(beta)

    m, n = shape

    s_alpha = np.random.normal(mu_a, sigma_a, n)
    s_beta = np.random.normal(mu_b, sigma_b, n)
    
    s_sigma_res = np.abs(np.random.normal(mu_sigma_res, sigma_sigma_res, n)) # TODO Maybe use gamma distribution here
    s_res = np.array([np.random.normal(0, s_sigma_res_i, m) for s_sigma_res_i in s_sigma_res]).T

    # TODO Market lenght could be probelm, generate market data
    s_r = s_alpha + np.reshape(r_market, (len(r_market), 1)) @ np.reshape(s_beta, (1, len(s_beta))) + s_res

    return s_r, s_alpha, s_beta, s_res