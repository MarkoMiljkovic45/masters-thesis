def on_test_start(self):
    self.test_results = {
        'test_loss': {
            'mae': None,
            'nll': None,
        },
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


def test_step(self, batch, batch_idx):
    context = batch[0].flatten(0, 1)  # (n_stocks, lookback_window, features)
    target = batch[1].flatten(0, 1)  # (n_stocks, target_window, features)
    factor = batch[2].flatten(0, 1)  # (2)
    inv_psi = batch[3].flatten(0, 1)  # (n_stocks,)

    r_target = target[:, :, 0]  # (n_stocks, target_window)
    r_market_target = target[:, :, 1]  # (n_stocks, target_window)
    alpha_target = target[:, 0, 2]  # (n_stocks)
    beta_target = target[:, 0, 3]  # (n_stocks)
    inv_psi = torch.diag(inv_psi)  # (n_stocks, n_stocks)
    f_mean = factor[0]
    f_var = factor[1]

    # MSE
    alpha_model, beta_model = self(context)  # 2x(n_stocks, 1)
    alpha_ols, beta_ols = ols(context[0, :, 1], context[:, :, 0])  # 2x(n_stocks)

    r_pred_model = alpha_model + beta_model * r_market_target
    r_pred_ols = alpha_ols.unsqueeze(-1) + beta_ols.unsqueeze(-1) * r_market_target

    # NLL
    r_mean = alpha_model + beta_model * f_mean
    r_inv_cov = inverse_returns_covariance(beta_model, inv_psi, f_var)

    # Fix shape
    alpha_model = alpha_model.squeeze()
    beta_model = beta_model.squeeze()

    test_mae_loss = self.test_mae(r_pred_model, r_target)
    test_nll_loss = self.test_nll(r_mean, r_inv_cov, r_target)

    self.log('test/step/mae', test_mae_loss)
    self.log('test/step/nll', test_nll_loss)

    # TODO Enable later
    """
    self.test_results['recon_residuals']['model'].append((r_target - r_pred_model).detach())
    self.test_results['alpha_residuals']['model'].append((alpha_target - alpha_model).detach())
    self.test_results['beta_residuals']['model'].append((beta_target - beta_model).detach())
    self.test_results['recon_residuals']['ols'].append((r_target - r_pred_ols).detach())
    self.test_results['alpha_residuals']['ols'].append((alpha_target - alpha_ols).detach())
    self.test_results['beta_residuals']['ols'].append((beta_target - beta_ols).detach())

    self.test_results['alpha']['model'].append(alpha_model.detach())
    self.test_results['beta']['model'].append(beta_model.detach())
    self.test_results['alpha']['ols'].append(alpha_ols.detach())
    self.test_results['beta']['ols'].append(beta_ols.detach())
    self.test_results['alpha']['true'].append(alpha_target.detach())
    self.test_results['beta']['true'].append(beta_target.detach())
    """


def _plot(self):
    tb = self.logger.experiment

    tb.add_figure('scatter/recon_residuals',
                  scatter_plot(self.test_results['recon_residuals']['model'],
                               self.test_results['recon_residuals']['ols'],
                               title='Model vs OLS Reconstruction Residuals'))

    tb.add_figure('scatter/alphas',
                  scatter_plot(self.test_results['alpha']['model'], self.test_results['alpha']['ols'],
                               title='Model vs OLS Alphas'))

    tb.add_figure('scatter/betas',
                  scatter_plot(self.test_results['beta']['model'], self.test_results['beta']['ols'],
                               title='Model vs OLS Betas'))

    tb.add_figure('hist/recon_residuals',
                  hist_plot(self.test_results['recon_residuals']['model'], self.test_results['recon_residuals']['ols'],
                            title='Model vs OLS Reconstruction Residuals'))

    tb.add_figure('hist/alphas',
                  hist_plot(self.test_results['alpha_residuals']['model'], self.test_results['alpha_residuals']['ols'],
                            title='Model vs OLS Alpha Residuals'))

    tb.add_figure('hist/betas',
                  hist_plot(self.test_results['beta_residuals']['model'], self.test_results['beta_residuals']['ols'],
                            title='Model vs OLS Beta Residuals'))

    estimation_plots(tb,
                     self.test_results['alpha']['model'],
                     self.test_results['alpha']['ols'],
                     self.test_results['alpha']['true'],
                     est_kind='alpha'
                     )

    estimation_plots(tb,
                     self.test_results['beta']['model'],
                     self.test_results['beta']['ols'],
                     self.test_results['beta']['true'],
                     est_kind='beta'
                     )

    tb.add_figure('estimation/alpha', estimation_scatter(
        self.test_results['alpha']['model'],
        self.test_results['alpha']['ols'],
        self.test_results['alpha']['true'],
        est_kind='alpha'
    ))

    tb.add_figure('estimation/beta', estimation_scatter(
        self.test_results['beta']['model'],
        self.test_results['beta']['ols'],
        self.test_results['beta']['true'],
        est_kind='beta'
    ))

    self.logger.save()


def on_test_end(self):
    test_mae_loss = self.test_mae.compute()
    test_nll_loss = self.test_nll.compute()

    self.log('test/total/mae', test_mae_loss)
    self.log('test/total/nll', test_nll_loss)

    self.test_results['test_loss']['mae'] = test_mae_loss.cpu().item()
    self.test_results['test_loss']['nll'] = test_nll_loss.cpu().item()

    # TODO Move plotting outside model
    """
    self.test_results['recon_residuals']['model'] = torch.stack(self.test_results['recon_residuals']['model']).mean(dim=-1).cpu()
    self.test_results['recon_residuals']['ols'] = torch.stack(self.test_results['recon_residuals']['ols']).mean(dim=-1).cpu()
    self.test_results['alpha_residuals']['model'] = torch.stack(self.test_results['alpha_residuals']['model']).cpu()
    self.test_results['alpha_residuals']['ols'] = torch.stack(self.test_results['alpha_residuals']['ols']).cpu()
    self.test_results['beta_residuals']['model'] = torch.stack(self.test_results['beta_residuals']['model']).cpu()
    self.test_results['beta_residuals']['ols'] = torch.stack(self.test_results['beta_residuals']['ols']).cpu()

    self.test_results['alpha']['model'] = torch.stack(self.test_results['alpha']['model']).cpu()
    self.test_results['beta']['model'] = torch.stack(self.test_results['beta']['model']).cpu()
    self.test_results['alpha']['ols'] = torch.stack(self.test_results['alpha']['ols']).cpu()
    self.test_results['beta']['ols'] = torch.stack(self.test_results['beta']['ols']).cpu()
    self.test_results['alpha']['true'] = torch.stack(self.test_results['alpha']['true']).cpu()
    self.test_results['beta']['true'] = torch.stack(self.test_results['beta']['true']).cpu()

    self._plot()
    """