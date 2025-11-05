import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import SyntheticLogReturnsDataset
from model import FinancialVAE

def config2str(config: dict):
    config_str = ""
    for key in config:
        config_str += f"{key:>20s}: {config[key]}\n"
    
    return config_str

def train_loop(dataloader, model, optimizer, save_freq=0.05):
    # KL annealing schedule
    #beta = min(1.0, epoch / warmup_epochs)
    beta = 0
    size = len(dataloader.dataset)
    freq = int(len(dataloader) * save_freq)
    train_loss = []

    model.train()
    for batch, X in enumerate(dataloader):
        # Compute reconstructiona and loss
        recon_X, mu, logvar = model(X)
        loss, recon_loss, kld = model.loss_function(recon_X, X, mu, logvar, beta=beta, reduction='mean')
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        train_loss.append(loss)
        
        if batch % freq == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss

def val_loop(dataloader, model):
    val_loss = []

    model.eval()
    with torch.no_grad():
        for X in dataloader:
            recon_X, mu, logvar = model(X)
            loss, recon_loss, kld = model.loss_function(recon_X, X, mu, logvar, beta=0, reduction='mean')
            
            val_loss.append(loss.item())

    avg_val_loss = np.mean(val_loss)
    print(f"Validation Error: \n Avg loss: {avg_val_loss:>8f} \n")

    return avg_val_loss

def run_model(config, log_dir="runs"):
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('hparams', config2str(config))
    
    layout = {
        "TrainValLoss": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]]
        },
    }
    
    writer.add_custom_scalars(layout)
    
    dataset = SyntheticLogReturnsDataset(config['N_STOCK'], config['T'], window=config['WINDOW'])
    dataset_loader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=config['SHUFFLE_BATCHES'])
    
    train_len = int(len(dataset) * config['DATASET_SPLIT_RATIO'])
    train_loader = DataLoader(dataset[:train_len], batch_size=config['BATCH_SIZE'], shuffle=config['SHUFFLE_BATCHES'])
    val_loader   = DataLoader(dataset[train_len:], batch_size=config['BATCH_SIZE'], shuffle=config['SHUFFLE_BATCHES'])

    # Initialize model
    model = FinancialVAE(
        n_stock=config['N_STOCK'],
        window=config['WINDOW'],
        hidden_dims=config['HIDDEN_DIMS'],
        latent_dim=config['LATENT_DIM'],
        dropout=config['DROPOUT'],
        activation=config['ACTIVATION']
    ).to(torch.device('cpu'))
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    # Train
    best_val_loss = float('inf')
    patience = config['PATIENCE']
    patience_counter = 0
    for epoch in range(1, config['NUM_EPOCHS'] + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        avg_train_loss = np.mean(train_loop(train_loader, model, optimizer))
        avg_val_loss = val_loop(val_loader, model)

        writer.add_scalar('loss/train', avg_train_loss, epoch)
        writer.add_scalar('loss/validation', avg_val_loss, epoch)

        model_alpha = model.decoder.bias.cpu().detach().numpy()
        model_beta = model.decoder.weight.cpu().detach().numpy()[:, 0]
        
        writer.add_histogram('Parameters/ModelAlpha', model_alpha, global_step=epoch)
        writer.add_histogram('Parameters/ModelBeta', model_beta, global_step=epoch)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
    print("Done!")

    # Evaluate latent factors
    X = dataset[:]
    
    if config['WINDOW'] > 1:
        X = torch.flatten(X, start_dim=1, end_dim=-1)
    
    mu = model.get_latent_representation(X)
    r_market = dataset.r_market.numpy()[config['WINDOW'] - 1:]
    
    fig = plt.figure(figsize=(14, 8))
    plt.plot(mu, label='$\\mu_z$')
    plt.plot(r_market, label='r_market')
    plt.title(f'corr($\\mu_z$, r_market): {np.corrcoef(r_market, mu.T)[0, 1]:.5f}')
    plt.legend()
    plt.grid()
    writer.add_figure('Plots/LatentFactor', fig)

    # Evaluate decoder params
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    #Get reael alpha and beta
    s_alpha = dataset.alpha.numpy()
    s_beta  = dataset.beta.numpy()
    
    # Get model alpha and beta
    model_alpha = model.decoder.bias.cpu().detach().numpy()
    model_beta = model.decoder.weight.cpu().detach().numpy()[:, 0]
    
    # Calculate correlations
    alpha_corr = np.corrcoef(s_alpha, model_alpha)[0, 1]
    beta_corr = np.corrcoef(s_beta, model_beta)[0, 1]
    
    # Top left: Alpha scatter plot
    axes[0, 0].scatter(s_alpha, model_alpha)
    axes[0, 0].set_xlabel('OLS Alpha')
    axes[0, 0].set_ylabel('Model Alpha')
    axes[0, 0].set_title(f'Alpha OLS vs Model scatter plot (corr = {alpha_corr:.3f})')
    axes[0, 0].grid()
    
    # Top right: Beta scatter plot
    axes[0, 1].scatter(s_beta, model_beta)
    axes[0, 1].set_xlabel('OLS Beta')
    axes[0, 1].set_ylabel('Model Beta')
    axes[0, 1].set_title(f'Beta OLS vs Model scatter plot (corr = {beta_corr:.3f})')
    axes[0, 1].grid()
    
    # Bottom left: Alpha histogram
    axes[1, 0].hist(s_alpha, bins=25, alpha=0.7, label='Synthetic')
    axes[1, 0].hist(model_alpha, bins=25, alpha=0.7, label='Model')
    axes[1, 0].set_title('Alpha histogram')
    axes[1, 0].grid(alpha=0.5)
    axes[1, 0].legend()
    
    # Bottom right: Beta histogram
    axes[1, 1].hist(s_beta, bins=25, alpha=0.7, label='OLS')
    axes[1, 1].hist(model_beta, bins=25, alpha=0.7, label='Model')
    axes[1, 1].set_title('Beta histogram')
    axes[1, 1].grid(alpha=0.5)
    axes[1, 1].legend()
    
    plt.tight_layout()
    writer.add_figure('Plots/ModelAlphaBeta', fig)
    
    writer.flush()
    writer.close()

    return model

def main():
    
    config = {
        'N_STOCK': 300,
        'T': 250,
        'WINDOW': 3,
        'HIDDEN_DIMS': [128, 64],
        'LATENT_DIM': 1,
        'BATCH_SIZE': 1,
        'NUM_EPOCHS': 30,
        'LEARNING_RATE': 1e-4,
        'SHUFFLE_BATCHES': True,
        'DATASET_SPLIT_RATIO': 0.8,
        'DROPOUT': 0.25,
        'PATIENCE': 5,
        'ACTIVATION': 'elu'
    }
    
    ts = [200, 500, 1000, 2000]
    windows = [1, 2 ,3, 5]
    
    for window in windows:
        for t in ts:
            config['T'] = t
            config['WINDOW'] = window

            print(f'Running model t={t}, window={window}')
            run_model(config, log_dir=f'runs/exp_t{t}_w{window}')

if __name__ == "__main__":
    main()