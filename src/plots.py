import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

plt.style.use('default')
figsize = (16, 9)


def scatter_plot(model: Tensor, ols: Tensor, title: str):
    fig = plt.figure(figsize=figsize)

    plt.scatter(model.flatten(), ols.flatten(), marker='.')
    plt.xlabel('Model')
    plt.ylabel('OLS')

    identity = (model.min(), model.max())
    plt.plot(identity, identity, 'r--')

    corr = torch.corrcoef(torch.stack([
        model.flatten(),
        ols.flatten()
    ]))[0, 1]

    plt.title(title + f', corr={corr:.4f}')
    plt.grid(alpha=0.5)

    return fig

def hist_plot(model: Tensor, ols: Tensor, title: str):
    fig = plt.figure(figsize=figsize)

    model = model.flatten()
    ols = ols.flatten()

    bins = int(len(model) * 0.01) + 1

    plt.hist(model, bins=bins, density=True, alpha=0.6, label='Model', color='blue')
    plt.hist(ols, bins=bins, density=True, alpha=0.6, label='OLS', color='orange')

    avg_model = model.mean().item()
    avg_ols = ols.mean().item()

    std_model = model.std().item()
    std_ols = ols.std().item()

    plt.axvline(avg_model, color='blue', linestyle='--', label=f'Model Residual Avg: {avg_model:.4f} (std={std_model:.4f})')
    plt.axvline(avg_ols, color='orange', linestyle='--', label=f'OLS   Residual Avg: {avg_ols:.4f} (std={std_ols:.4f})')

    plt.title(title)
    plt.grid(alpha=0.5)
    plt.legend()

    return fig

def estimation_plots(tb, model_ests, ols_ests, trues, est_kind: str = 'alpha'):
    n_examples = min(model_ests.shape[1], 9)

    for stock_idx in range(n_examples):
        fig = plt.figure(figsize=(16, 9))
        plt.title(f'Model vs OLS {est_kind} estimation (Stock {stock_idx})')

        est_gt = trues[:, stock_idx]
        model_est = model_ests[:, stock_idx]
        ols_est = ols_ests[:, stock_idx]

        sample = np.arange(len(model_est))

        plt.plot(sample, est_gt, color='magenta', linestyle='--', label=f'True {est_kind}', alpha=0.5)
        plt.scatter(sample, model_est, label='Model', marker='.', color='blue')
        plt.scatter(sample, ols_est, label='OLS', marker='.', color='orange')

        plt.legend()
        plt.grid(alpha=0.5)

        tb.add_figure(f'estimation/examples_{est_kind}', fig, global_step=stock_idx)

def estimation_scatter(model_ests, ols_ests, trues, est_kind: str = 'alpha'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(f'Ground Truth {est_kind} vs Estimated {est_kind}')

    model_ests = model_ests.flatten()
    ols_ests = ols_ests.flatten()
    trues = trues.flatten()
    identity = (trues.min(), trues.max())

    # Top subplot - Model
    model_corr = torch.corrcoef(torch.stack([
        model_ests, trues
    ]))[0, 1]

    ax1.set_title(f'Model corr={model_corr:.4f}')
    ax1.set_ylabel(f'Model {est_kind}')
    ax1.plot(identity, identity, color='magenta', linestyle='--')
    ax1.grid()

    # Bottom subplot - OLS
    ols_corr = torch.corrcoef(torch.stack([
        ols_ests, trues
    ]))[0, 1]

    ax2.set_title(f'OLS corr={ols_corr:.4f}')
    ax2.set_xlabel(f'Ground Truth {est_kind}')
    ax2.set_ylabel(f'OLS {est_kind}')
    ax2.plot(identity, identity, color='magenta', linestyle='--')
    ax2.grid()

    ax1.scatter(trues, model_ests, marker='.', alpha=0.15, color='blue')
    ax2.scatter(trues, ols_ests, marker='.', alpha=0.15, color='orange')

    return fig