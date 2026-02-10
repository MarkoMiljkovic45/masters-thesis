from pathlib import Path

import hydra
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from src import FinancialLstmDataModule
from src.common import ols
from src.plots import scatter_plot, hist_plot, estimation_plots, estimation_scatter
from train import get_model_class


def init_test_results():
    return {
        'recon_residuals': {
            'model': [],
            'ols': []
        },
        'alpha_residuals': {
            'model': [],
            'ols': []
        },
        'beta_residuals': {
            'model': [],
            'ols': []
        },
        'alpha': {
            'model': [],
            'ols': [],
            'true': []
        },
        'beta': {
            'model': [],
            'ols': [],
            'true': []
        }
    }


def test_step(model, test_results, batch):
    context = batch[0].flatten(0, 1)  # (n_stocks, lookback_window, features)
    target = batch[1].flatten(0, 1)  # (n_stocks, target_window, features)

    r_target = target[:, :, 0]  # (n_stocks, target_window)
    r_market_target = target[:, :, 1]  # (n_stocks, target_window)
    alpha_target = target[:, 0, 2]  # (n_stocks)
    beta_target = target[:, 0, 3]  # (n_stocks)

    # MSE
    alpha_model, beta_model = model(context)  # 2x(n_stocks, 1)
    alpha_ols, beta_ols = ols(context[0, :, 1], context[:, :, 0])  # 2x(n_stocks)

    r_pred_model = alpha_model + beta_model * r_market_target
    r_pred_ols = alpha_ols.unsqueeze(-1) + beta_ols.unsqueeze(-1) * r_market_target

    # Fix shape
    alpha_model = alpha_model.squeeze()
    beta_model = beta_model.squeeze()

    test_results['recon_residuals']['model'].append((r_target - r_pred_model).detach())
    test_results['alpha_residuals']['model'].append((alpha_target - alpha_model).detach())
    test_results['beta_residuals']['model'].append((beta_target - beta_model).detach())
    test_results['recon_residuals']['ols'].append((r_target - r_pred_ols).detach())
    test_results['alpha_residuals']['ols'].append((alpha_target - alpha_ols).detach())
    test_results['beta_residuals']['ols'].append((beta_target - beta_ols).detach())
    test_results['alpha']['model'].append(alpha_model.detach())
    test_results['beta']['model'].append(beta_model.detach())
    test_results['alpha']['ols'].append(alpha_ols.detach())
    test_results['beta']['ols'].append(beta_ols.detach())
    test_results['alpha']['true'].append(alpha_target.detach())
    test_results['beta']['true'].append(beta_target.detach())


def transform_test_results(test_results):
    test_results['recon_residuals']['model'] = torch.stack(test_results['recon_residuals']['model']).mean(dim=-1).cpu()
    test_results['recon_residuals']['ols'] = torch.stack(test_results['recon_residuals']['ols']).mean(dim=-1).cpu()
    test_results['alpha_residuals']['model'] = torch.stack(test_results['alpha_residuals']['model']).cpu()
    test_results['alpha_residuals']['ols'] = torch.stack(test_results['alpha_residuals']['ols']).cpu()
    test_results['beta_residuals']['model'] = torch.stack(test_results['beta_residuals']['model']).cpu()
    test_results['beta_residuals']['ols'] = torch.stack(test_results['beta_residuals']['ols']).cpu()

    test_results['alpha']['model'] = torch.stack(test_results['alpha']['model']).cpu()
    test_results['beta']['model'] = torch.stack(test_results['beta']['model']).cpu()
    test_results['alpha']['ols'] = torch.stack(test_results['alpha']['ols']).cpu()
    test_results['beta']['ols'] = torch.stack(test_results['beta']['ols']).cpu()
    test_results['alpha']['true'] = torch.stack(test_results['alpha']['true']).cpu()
    test_results['beta']['true'] = torch.stack(test_results['beta']['true']).cpu()


def plot(tb_logger, test_results):
    tb = tb_logger.experiment

    tb.add_figure('scatter/recon_residuals',
                  scatter_plot(test_results['recon_residuals']['model'],
                               test_results['recon_residuals']['ols'],
                               title='Model vs OLS Reconstruction Residuals'))

    tb.add_figure('scatter/alphas',
                  scatter_plot(test_results['alpha']['model'], test_results['alpha']['ols'],
                               title='Model vs OLS Alphas'))

    tb.add_figure('scatter/betas',
                  scatter_plot(test_results['beta']['model'], test_results['beta']['ols'],
                               title='Model vs OLS Betas'))

    tb.add_figure('hist/recon_residuals',
                  hist_plot(test_results['recon_residuals']['model'], test_results['recon_residuals']['ols'],
                            title='Model vs OLS Reconstruction Residuals'))

    tb.add_figure('hist/alphas',
                  hist_plot(test_results['alpha_residuals']['model'], test_results['alpha_residuals']['ols'],
                            title='Model vs OLS Alpha Residuals'))

    tb.add_figure('hist/betas',
                  hist_plot(test_results['beta_residuals']['model'], test_results['beta_residuals']['ols'],
                            title='Model vs OLS Beta Residuals'))

    estimation_plots(tb,
                     test_results['alpha']['model'],
                     test_results['alpha']['ols'],
                     test_results['alpha']['true'],
                     est_kind='alpha'
                     )

    estimation_plots(tb,
                     test_results['beta']['model'],
                     test_results['beta']['ols'],
                     test_results['beta']['true'],
                     est_kind='beta'
                     )

    tb.add_figure('estimation/alpha', estimation_scatter(
        test_results['alpha']['model'],
        test_results['alpha']['ols'],
        test_results['alpha']['true'],
        est_kind='alpha'
    ))

    tb.add_figure('estimation/beta', estimation_scatter(
        test_results['beta']['model'],
        test_results['beta']['ols'],
        test_results['beta']['true'],
        est_kind='beta'
    ))

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Used to test and generate plots for a model
    """

    if not cfg.checkpoint:
        print("No checkpoint provided")
        return

    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Datamodule: {cfg.datamodule.name}")
    print(f"Model: {cfg.model.name}")
    print(f"Checkpoint: {cfg.checkpoint}")
    print("=" * 70 + "\n")

    # ==================== Initialize DataModule ====================
    dm = FinancialLstmDataModule(
        data_dir=Path(cfg.datamodule.data_dir),
        lookback_window=cfg.datamodule.lookback_window,
        target_window=cfg.datamodule.target_window,
        stride=cfg.datamodule.stride,
        prediction_task=cfg.datamodule.prediction_task,
        interaction_only=cfg.datamodule.interaction_only,
        batch_size=cfg.datamodule.batch_size
    )

    # ==================== Initialize Model ====================
    ModelClass = get_model_class(cfg.loss.module_class)
    model = ModelClass.load_from_checkpoint(cfg.checkpoint, map_location='cpu')
    print(model)

    # ==================== Initialize Logger ====================
    checkpoint = Path(cfg.checkpoint)
    save_dir = checkpoint.parts[0]
    name = '/'.join(checkpoint.parts[1:3])
    version = checkpoint.parts[3]

    tb_logger = TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        default_hp_metric=False
    )

    # ==================== Test Model ====================
    print("\n" + "=" * 50)
    print("Starting Testing...")
    print("=" * 50 + "\n")

    dm.prepare_data()
    dm.setup('test')
    teat_loader = dm.test_dataloader()
    test_results = init_test_results()

    model.eval()
    with torch.no_grad():
        for batch in teat_loader:
            test_step(model, test_results, batch)

    transform_test_results(test_results)
    plot(tb_logger, test_results)

    # ==================== Print Results ====================
    print("\n" + "=" * 50)
    print("Testing and plotting Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()