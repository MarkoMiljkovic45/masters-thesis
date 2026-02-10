from pathlib import Path

import hydra
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src import SyntheticLogReturns, FamaFrench25Portfolios, FinancialLstmDataModule
from src.model import FinancialLstmMse, FinancialLstmNll, FinancialLstmCombined

torch.set_float32_matmul_precision('medium')

raw_real_data_dir  = Path.cwd() / 'data' / 'real_raw'
real_data_dir      = Path.cwd() / 'data' / 'real'
synthetic_data_dir = Path.cwd() / 'data' / 'synthetic'

if not raw_real_data_dir.exists():
    # TODO Download data from Fama French website
    # https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    pass

if not real_data_dir.exists():
    real_data_dir.mkdir(parents=True, exist_ok=True)
    p25, mkt = FamaFrench25Portfolios.load(raw_real_data_dir)
    torch.save(p25, real_data_dir / 'stocks.pt')
    torch.save(mkt, real_data_dir / 'market.pt')

if not synthetic_data_dir.exists():
    synthetic_data_dir.mkdir(parents=True, exist_ok=True)
    r_stocks, r_market, alphas, betas = SyntheticLogReturns.generate(100, 1_000_000)
    torch.save(r_stocks, synthetic_data_dir / 'stocks.pt')
    torch.save(r_market, synthetic_data_dir / 'market.pt')
    torch.save(alphas, synthetic_data_dir / 'alphas.pt')
    torch.save(betas, synthetic_data_dir / 'betas.pt')


OmegaConf.register_new_resolver(
    "input_size_from_interaction",
    lambda interaction_only: 3 if interaction_only else 5
)


def get_model_class(module_class_name: str):
    """
    Map config string to actual model class.

    Args:
        module_class_name: String name of the Lightning module class

    Returns:
        The corresponding Lightning module class
    """
    model_classes = {
        'FinancialLstmMse': FinancialLstmMse,
        'FinancialLstmNll': FinancialLstmNll,
        'FinancialLstmCombined': FinancialLstmCombined,
    }

    if module_class_name not in model_classes:
        raise ValueError(
            f"Unknown module class: {module_class_name}. "
            f"Available: {list(model_classes.keys())}"
        )

    return model_classes[module_class_name]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function with Hydra configuration management.

    Example runs:
        # Use defaults (MSE loss)
        python train.py

        # Use different loss functions
        python train.py loss=nll
        python train.py loss=combined

        # Override combined loss weight
        python train.py loss=combined loss.mse_weight=0.7

        # Full configuration
        python train.py datamodule=real model=large loss=combined trainer=slower

        # Sweep across loss functions
        python train.py -m loss=mse,nll,combined model=small,medium,large

        # Sweep combined loss weights
        python train.py -m loss=combined loss.mse_weight=0.3,0.5,0.7
    """

    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Datamodule: {cfg.datamodule.name}")
    print(f"Model: {cfg.model.name} (hidden={cfg.model.hidden_size}, layers={cfg.model.num_layers})")
    print(f"Loss: {cfg.loss.name} (class={cfg.loss.module_class})")
    if cfg.loss.name == 'combined':
        print(f"  └─ MSE Weight: {cfg.loss.mse_weight}")
    print(f"Trainer: {cfg.trainer.name} (epochs={cfg.trainer.max_epochs})")
    print(f"Learning Rate: {cfg.model.learning_rate}")
    print(f"Logger: {cfg.logger.name}/{cfg.logger.version}")
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

    # Base model parameters (common to all loss types)
    model_params = {
        'input_size': cfg.model.input_size,
        'hidden_size': cfg.model.hidden_size,
        'num_layers': cfg.model.num_layers,
        'dropout': cfg.model.dropout,
        'learning_rate': cfg.model.learning_rate
    }

    # Add mse_weight parameter only for FinancialLstmCombined
    if cfg.loss.name == 'combined':
        model_params['mse_weight'] = cfg.loss.mse_weight

    model = ModelClass(**model_params)
    model.compile()

    print(model)
    print(f"\nUsing Lightning Module: {ModelClass.__name__}\n")

    # ==================== Initialize Logger ====================
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logger.save_dir,
        name=cfg.logger.name,
        version=cfg.logger.version,
        default_hp_metric=cfg.logger.default_hp_metric
    )

    # ==================== Initialize Callbacks ====================
    checkpoint_dir = Path(cfg.logger.save_dir) / cfg.logger.name / cfg.logger.version / "checkpoints"

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best_epoch={epoch:02d}_val={loss/total/val:.4f}",
            monitor="loss/total/val",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(
            logging_interval="epoch",
            log_momentum=False
        )
    ]

    # ==================== Initialize Trainer ====================
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        logger=tb_logger,
        callbacks=callbacks,
        # Add check_val_every_n_epoch if it exists in config
        **({"check_val_every_n_epoch": cfg.trainer.check_val_every_n_epoch}
           if "check_val_every_n_epoch" in cfg.trainer else {})
    )

    # ==================== Train the Model ====================
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50 + "\n")

    trainer.fit(model, datamodule=dm)

    # ==================== Print Results ====================
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    print("=" * 50 + "\n")

    # ==================== Test the Model ====================
    trainer.test(model, datamodule=dm)

    # ==================== Log Final Metrics ====================
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict.pop('logger', None)

    tb_logger.log_hyperparams(
        params=config_dict,
        metrics={
            'test/mae': model.test_results['test_loss']['mae'],
            'test/nll': model.test_results['test_loss']['nll'],
            'test/best_val_loss': trainer.checkpoint_callback.best_model_score.item(),
        }
    )

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Test MAE: {model.test_results['test_loss']['mae']:.6f}")
    print(f"Test NLL: {model.test_results['test_loss']['nll']:.6f}")
    print("=" * 50 + "\n")

    return trainer.checkpoint_callback.best_model_score.item()


if __name__ == '__main__':
    main()