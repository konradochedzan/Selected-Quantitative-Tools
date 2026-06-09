import datetime
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Type
import warnings
import random
from Base_models import FeedForwardPredictor, AutoEncoder, ElasticNetLoss, SharpeRatioLoss
from TM_models import TemporalConvNet, TemporalFusionTransformer, TemporalFusionTransformer2
from Simple_models import SimpleConvolutional, SimpleFeedForward, SimpleLSTM, SimpleTransformer
from saving import save_model, save_predictions
import itertools
warnings.filterwarnings('ignore')
import joblib

# Fixed parameters for architecture selection
PARAMS = {
    'window_strategy': 'expanding',
    'train_window_years': 3,
    'test_window_years': 1,
    'encoding_dim': 10,
    'seq_length': 24,
    'epochs': 120,
    'lr': 0.0001,
    'batch_size': 128,
    'device': 'cuda',
    'plot_results': True,
    'do_print': False,
    'use_autoencoder':True
}
PARAMS.setdefault('use_autoencoder', True)

# Different architectures for each model
ARCHITECTURE_GRID = {
    'TCN': {
        'class': TemporalConvNet,
        'grid': {
            'num_channels': [[32, 64, 32], [32, 48, 64, 48, 32], [48, 64, 48]],
            'kernel_size': [5],
            'dropout': [0.1],
            'pool': ['last']
        }
    },
    'LSTM': {
        'class': SimpleLSTM,
        'grid': {
            'hidden_dim': [64, 200, 256],
            'num_layers': [2, 3],
            'dropout': [0.1]
        }
    },
    'FeedForward': {
        'class': SimpleFeedForward,
        'grid': {
            'hidden_dim': [100,200,300],
            'dropout': [0.1]
        }
    },
    'TFT': {
        'class': TemporalFusionTransformer,
        'grid': {
            'hidden_dim': [64, 128],
            'num_heads': [4, 8],
            'num_layers': [2, 3],
            'dropout': [0.1]
        }
    },
    'Transformer': {
        'class': SimpleTransformer,
        'grid': {
            'model_dim': [128, 256],
            'nhead': [8],
            'num_layers': [2, 3],
            'dropout': [0.1],
            'max_seq_length': [500]
        }
    },
    'CNN': {
        'class': SimpleConvolutional,
        'grid': {
            'num_channels': [[32, 64, 32], [32, 64, 64, 32], [48,72,72,48]],
            'kernel_size': [5],
            'dropout': [0.1],
            'seq_length': [24]
        }
    }
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # might slow down but ensures reproducibility
    torch.backends.cudnn.benchmark = False
    
set_seed()

def calculate_sharpe_ratio(strategy_returns: np.ndarray, rf_rate: float) -> float:
    """Calculate annualized Sharpe ratio using strategy returns and actual risk-free rate"""
    if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
        return 0.0
    # rf_rate is already annual, convert to daily
    daily_rf = rf_rate / 252  # Assuming daily data
    excess_returns = strategy_returns - daily_rf
    return np.sqrt(252) * np.mean(excess_returns) / np.std(strategy_returns)

def calculate_sortino_ratio(strategy_returns: np.ndarray, rf_rate: float) -> float:
    """Calculate annualized Sortino ratio using strategy returns and actual risk-free rate"""
    if len(strategy_returns) == 0:
        return 0.0
    # rf_rate is already annual, convert to daily
    daily_rf = rf_rate / 252  # Assuming daily data
    excess_returns = strategy_returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

def hit_rate(pred: np.ndarray, targ: np.ndarray) -> float:
    """Percentage of months the sign is predicted correctly."""
    return np.mean(np.sign(pred) == np.sign(targ))

def train_autoencoder(X_train: np.ndarray, encoding_dim: int, epochs: int = 100, 
                     lr: float = 0.001, device: str = 'cpu') -> AutoEncoder:
    """Train autoencoder on training data only"""
    input_dim = X_train.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    autoencoder.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructed = autoencoder(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    autoencoder.eval()
    return autoencoder

def train_model_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                     optimizer: optim.Optimizer, scheduler=None, model_type: str = 'feedforward', device: str = 'cpu') -> float:
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)
        loss.backward()
        
        if model_type.lower() == 'transformer':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        if scheduler and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
#                   device: str) -> Tuple[float, np.ndarray, np.ndarray]:
#     """Evaluate model and return loss, predictions, and targets"""
#     model.eval()
#     total_loss = 0
#     all_predictions = []
#     all_targets = []
    
#     with torch.no_grad():
#         for X_batch, y_batch in dataloader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             predictions = model(X_batch).squeeze()
#             loss = criterion(predictions, y_batch)
#             total_loss += loss.item()
#             all_predictions.extend(predictions.cpu().numpy())
#             all_targets.extend(y_batch.cpu().numpy())
    
#     return total_loss / len(dataloader), np.array(all_predictions), np.array(all_targets)

def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: str) -> Tuple[float, np.ndarray, np.ndarray]:

    model.eval()
    total_loss = 0.0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # ── 1. keep the batch axis ───────────────────────────────
            predictions = model(X_batch).squeeze(-1)      # shape: (batch_size,)
            y_batch     = y_batch.squeeze(-1)             # same for targets

            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

            # ── 2. flatten before extending the lists ───────────────
            all_predictions.extend(predictions.cpu().numpy().ravel())
            all_targets.extend(y_batch.cpu().numpy().ravel())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_predictions), np.array(all_targets)

def get_optimizer_for_model(model: nn.Module, model_type: str, lr: float = 0.001):
    """Get appropriate optimizer based on model type."""
    if model_type.lower() in ['transformer', 'temporalfusiontransformer']:
        # AdamW often works better for transformers
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    else:
        # Standard Adam for other models
        return optim.Adam(model.parameters(), lr=lr)

def get_learning_rate_scheduler(optimizer, model_type: str, num_training_steps: int = None):
    """Get appropriate learning rate scheduler based on model type."""
    if model_type.lower() == 'transformer' and num_training_steps:
        # Warmup scheduler for transformers
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=optimizer.param_groups[0]['lr'] * 10,
            total_steps=num_training_steps,
            pct_start=0.1  # 10% warmup
        )
    else:
        # Simple step scheduler for other models
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

def prepare_data_for_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    seq_length: int = 12,
    ar_lags: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert raw arrays to tensors, respecting model expectations."""
    seq_models = ['lstm', 'transformer', 'tcn', 'temporalconvnet',
              'cnn', 'conv1d', 'simpleconvolutional','temporalfusiontransformer']
    if model_type.lower() in seq_models:
        X_seq, y_seq = [], []
        for i in range(max(seq_length, ar_lags), len(X)):
            feature_seq = X[i - seq_length:i]
            ar_returns  = y[i - ar_lags:i].reshape(-1, 1)

            ar_pad = np.zeros((seq_length, ar_lags))
            ar_pad[-1, :] = ar_returns.ravel()

            X_seq.append(np.concatenate([feature_seq, ar_pad], axis=1))
            y_seq.append(y[i])

        return torch.FloatTensor(np.asarray(X_seq)), torch.FloatTensor(np.asarray(y_seq))

    # ── feed‑forward path ───────────────────────────────────────────
    X_ff, y_ff = [], []
    for i in range(ar_lags, len(X)):
        current_features = X[i]
        ar_returns       = y[i - ar_lags:i]
        X_ff.append(np.concatenate([current_features, ar_returns]))
        y_ff.append(y[i])

    return torch.FloatTensor(np.asarray(X_ff)), torch.FloatTensor(np.asarray(y_ff))


def improved_position_sizing(predictions, volatility_scaling=True, max_position=1.0):
    if volatility_scaling:
        pred_vol = np.std(predictions[-20:]) if len(predictions) >= 20 else np.std(predictions)
        if pred_vol > 0:
            scaled_predictions = predictions / (2 * pred_vol)
        else:
            scaled_predictions = predictions
    else:
        scaled_predictions = predictions

    positions = np.tanh(scaled_predictions) * max_position
    return positions


def sp500_training_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    tbill3m: np.ndarray,
    model_class: type,
    model_kwargs: Dict[str, Any],
    model_type: str = 'feedforward',
    window_strategy: str = 'rolling',
    train_window_years: int = 3,
    test_window_years: int = 1,
    use_autoencoder: bool = True,
    encoding_dim: int = 10,
    walk_forward_cv: bool = False,
    cv_months: int = 2,
    seq_length: int = 12,
    ar_lags: int = 5,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    alpha: float = 0,
    l1_ratio: float = 0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    random_seed: int = 42,
    plot_results: bool = True,
    do_print = True,
    plot_dir: str = "results_plots",
    plt_show = False

) -> Dict[str, Any]:
    """Rolling / expanding‑window S&P‑500 forecast pipeline plus TCN support."""

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    # === master container =========================================
    results: Dict[str, Any] = {
        'fold_results': [], 'scalers': [], 'autoencoders': [], 'models': [],
        'all_train_predictions': [], 'all_train_targets': [],
        'all_test_predictions':  [], 'all_test_targets':  [],
        'all_train_dates': [],     'all_test_dates':     [],
        'metrics': {
            'train_mse': [], 'test_mse': [],
            'train_sharpe': [], 'test_sharpe': [],
            'train_mae': [], 'test_mae': [],
            'train_r2':  [], 'test_r2':  [],
            'train_sortino': [], 'test_sortino': [],
            'train_hit': [], 'test_hit': []
        }
    }

    # --- per‑fold arrays for optional plotting --------------------
    fold_predictions, fold_targets, fold_dates = [], [], []

    # === build fold timeline ======================================
    start_date, end_date = dates[0], dates[-1]
    fold_start_dates, cur = [], start_date + pd.DateOffset(years=train_window_years)
    while cur + pd.DateOffset(years=test_window_years) <= end_date:
        fold_start_dates.append(cur)
        cur += pd.DateOffset(years=test_window_years)
    if do_print:
        print(f"Starting {window_strategy} window training with {len(fold_start_dates)} folds")
    seq_models = ['lstm', 'transformer', 'tcn', 'temporalconvnet',
                'cnn', 'conv1d', 'simpleconvolutional','temporalfusiontransformer']

    # === walk through every fold ==================================
    for fold_idx, test_start_date in enumerate(fold_start_dates, 1):
        if do_print:
            print(f"\n— Fold {fold_idx}/{len(fold_start_dates)} —")
        test_end_date  = test_start_date + pd.DateOffset(years=test_window_years)
        train_start_date = (test_start_date - pd.DateOffset(years=train_window_years)
                            if window_strategy == 'rolling' else start_date)

        train_mask = (dates >= train_start_date) & (dates < test_start_date)
        test_mask  = (dates >= test_start_date) & (dates < test_end_date)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print("Skipping fold – insufficient data")
            continue

        # ----------- slice data ----------------------------------
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        train_dates, test_dates = dates[train_mask], dates[test_mask]

        # ----------- scaling -------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ----------- optional autoencoder ------------------------
        autoencoder = None
        if use_autoencoder:
            autoencoder = train_autoencoder(X_train_scaled, encoding_dim, epochs=50, lr=lr, device=device)
            with torch.no_grad():
                X_train_scaled = autoencoder.encode(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
                X_test_scaled  = autoencoder.encode(torch.FloatTensor(X_test_scaled) .to(device)).cpu().numpy()

        # ----------- AR‑lag augmentation -------------------------
        X_train_tensor, y_train_tensor = prepare_data_for_model(X_train_scaled, y_train,
                                                                model_type, seq_length, ar_lags)
        X_test_tensor,  y_test_tensor  = prepare_data_for_model(X_test_scaled,  y_test,
                                                                model_type, seq_length, ar_lags)
        offset = max(seq_length, ar_lags) if model_type.lower() in seq_models else ar_lags
        train_dates_adj, test_dates_adj = train_dates[offset:], test_dates[offset:]

        # ----------- model / optimiser ---------------------------
        input_dim = X_train_tensor.shape[-1]
        model     = model_class(input_dim=input_dim, **model_kwargs).to(device)
        criterion = ElasticNetLoss(model=model, alpha=alpha, l1_ratio=l1_ratio)
    #         criterion = lambda predictions, targets: (
    #     SharpeRatioLoss(risk_free_rate=np.mean(tbill3m[train_mask]))(predictions, targets)
    #     + 5*nn.MSELoss()(predictions, targets)
    # )
        
        optimizer = get_optimizer_for_model(model, model_type, lr)
        scheduler = get_learning_rate_scheduler(optimizer, model_type)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_test_tensor,  y_test_tensor),  batch_size=batch_size, shuffle=False)

        # ----------- training loop -------------------------------
        if do_print:
            print("Training final model…")
        final_epochs = epochs // 2 if walk_forward_cv else epochs
        for ep in range(final_epochs):
                loss = train_model_epoch(model, train_loader, criterion, optimizer, scheduler=scheduler,
                                        model_type=model_type, device=device)
                if (ep + 1) % 20 == 0:
                    print(f"  epoch {ep+1}/{final_epochs}  loss={loss:.6f}")

        # ----------- evaluation ----------------------------------
        _, train_predictions, train_targets = evaluate_model(model, train_loader, criterion, device)
        _, test_predictions,  test_targets  = evaluate_model(model, test_loader,  criterion, device)

        # ----------- metric calc ---------------------------------
        train_mse = mean_squared_error(train_targets, train_predictions)
        test_mse  = mean_squared_error(test_targets,  test_predictions)
        train_mae = mean_absolute_error(train_targets, train_predictions)
        test_mae  = mean_absolute_error(test_targets,  test_predictions)
        train_r2  = r2_score(train_targets, train_predictions)
        test_r2   = r2_score(test_targets,  test_predictions)

        train_positions = improved_position_sizing(train_predictions, volatility_scaling=True)
        test_positions = improved_position_sizing(test_predictions, volatility_scaling=True)

        train_strategy_returns = train_positions * train_targets
        test_strategy_returns  = test_positions  * test_targets

        train_rf = np.mean(tbill3m[train_mask])
        test_rf  = np.mean(tbill3m[test_mask])

        train_sharpe  = calculate_sharpe_ratio(train_strategy_returns, train_rf)
        test_sharpe   = calculate_sharpe_ratio(test_strategy_returns,  test_rf)
        train_sortino = calculate_sortino_ratio(train_strategy_returns, train_rf)
        test_sortino  = calculate_sortino_ratio(test_strategy_returns,  test_rf)

        train_hit = hit_rate(train_predictions, train_targets)
        test_hit  = hit_rate(test_predictions,  test_targets)
        if do_print:
            print(f"Train MSE {train_mse:.6f} | MAE {train_mae:.6f} | R² {train_r2:.4f} | Sharpe {train_sharpe:.3f} | Sortino {train_sortino:.3f} | Hit {train_hit:.2%}")
            print(f"Test  MSE {test_mse:.6f} | MAE {test_mae:.6f} | R² {test_r2:.4f} | Sharpe {test_sharpe:.3f} | Sortino {test_sortino:.3f} | Hit {test_hit:.2%}")

        # Store results - everything below remains the same
        fold_result = {
            'fold': fold_idx,
            'train_period': (train_dates[0], train_dates[-1]),
            'test_period': (test_dates[0], test_dates[-1]),
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_sortino': train_sortino,
            'test_sortino': test_sortino,
            'train_hit': train_hit,
            'test_hit': test_hit,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'train_targets': train_targets,
            'test_targets': test_targets,
            'train_dates': train_dates_adj,
            'test_dates': test_dates_adj
        }
        
        results['fold_results'].append(fold_result)
        results['scalers'].append(scaler)
        results['autoencoders'].append(autoencoder)
        results['models'].append(model.state_dict())
        
        # Accumulate for overall plotting
        results['all_train_predictions'].extend(train_predictions)
        results['all_train_targets'].extend(train_targets)
        results['all_test_predictions'].extend(test_predictions)
        results['all_test_targets'].extend(test_targets)
        results['all_train_dates'].extend(train_dates_adj)
        results['all_test_dates'].extend(test_dates_adj)
        
        results['metrics']['train_mse'].append(train_mse)
        results['metrics']['test_mse'].append(test_mse)
        results['metrics']['train_sharpe'].append(train_sharpe)
        results['metrics']['test_sharpe'].append(test_sharpe)
        results['metrics']['train_mae'].append(train_mae)
        results['metrics']['test_mae'].append(test_mae)
        results['metrics']['train_r2'].append(train_r2)
        results['metrics']['test_r2'].append(test_r2)
        results['metrics']['train_sortino'].append(train_sortino)
        results['metrics']['test_sortino'].append(test_sortino)
        results['metrics']['train_hit'].append(train_hit)
        results['metrics']['test_hit'].append(test_hit)
        
        # Store for individual fold plotting
        fold_predictions.append((train_predictions, test_predictions))
        fold_targets.append((train_targets, test_targets))
        fold_dates.append((train_dates_adj, test_dates_adj))
    
    # Calculate overall metrics
    if results['metrics']['train_mse']:
        results['overall_metrics'] = {
            'avg_train_mse': np.mean(results['metrics']['train_mse']),
            'avg_test_mse': np.mean(results['metrics']['test_mse']),
            'avg_train_sharpe': np.mean(results['metrics']['train_sharpe']),
            'avg_test_sharpe': np.mean(results['metrics']['test_sharpe']),
            'std_train_mse': np.std(results['metrics']['train_mse']),
            'std_test_mse': np.std(results['metrics']['test_mse']),
            'std_train_sharpe': np.std(results['metrics']['train_sharpe']),
            'std_test_sharpe': np.std(results['metrics']['test_sharpe']),
            'avg_train_mae': np.mean(results['metrics']['train_mae']),
            'avg_test_mae': np.mean(results['metrics']['test_mae']),
            'std_train_mae': np.std(results['metrics']['train_mae']),
            'std_test_mae': np.std(results['metrics']['test_mae']),
            'avg_train_r2': np.mean(results['metrics']['train_r2']),
            'avg_test_r2': np.mean(results['metrics']['test_r2']),
            'std_train_r2': np.std(results['metrics']['train_r2']),
            'std_test_r2': np.std(results['metrics']['test_r2']),
            'avg_train_sortino': np.mean(results['metrics']['train_sortino']),
            'avg_test_sortino': np.mean(results['metrics']['test_sortino']),
            'std_train_sortino': np.std(results['metrics']['train_sortino']),
            'std_test_sortino': np.std(results['metrics']['test_sortino']),
            'avg_train_hit': np.mean(results['metrics']['train_hit']),
            'avg_test_hit': np.mean(results['metrics']['test_hit']),
            'std_train_hit': np.std(results['metrics']['train_hit']),
            'std_test_hit': np.std(results['metrics']['test_hit'])
        }
        if do_print:
            print("\n=== OVERALL RESULTS ===")
            print(f"Average Train MSE: {results['overall_metrics']['avg_train_mse']:.6f} ± {results['overall_metrics']['std_train_mse']:.6f}")
            print(f"Average Test MSE: {results['overall_metrics']['avg_test_mse']:.6f} ± {results['overall_metrics']['std_test_mse']:.6f}")
            print(f"Average Train Sharpe: {results['overall_metrics']['avg_train_sharpe']:.4f} ± {results['overall_metrics']['std_train_sharpe']:.4f}")
            print(f"Average Test Sharpe: {results['overall_metrics']['avg_test_sharpe']:.4f} ± {results['overall_metrics']['std_test_sharpe']:.4f}")
            print(f"Average Train MAE: {results['overall_metrics']['avg_train_mae']:.6f} ± {results['overall_metrics']['std_train_mae']:.6f}")
            print(f"Average Test MAE: {results['overall_metrics']['avg_test_mae']:.6f} ± {results['overall_metrics']['std_test_mae']:.6f}")
            print(f"Average Train R²: {results['overall_metrics']['avg_train_r2']:.4f} ± {results['overall_metrics']['std_train_r2']:.4f}")
            print(f"Average Test R²: {results['overall_metrics']['avg_test_r2']:.4f} ± {results['overall_metrics']['std_test_r2']:.4f}")
            print(f"Average Train Sortino: {results['overall_metrics']['avg_train_sortino']:.4f} ± {results['overall_metrics']['std_train_sortino']:.4f}") 
            print(f"Average Test Sortino: {results['overall_metrics']['avg_test_sortino']:.4f} ± {results['overall_metrics']['std_test_sortino']:.4f}")
            print(f"Average Train Hit Rate: {results['overall_metrics']['avg_train_hit']:.2%} ± {results['overall_metrics']['std_train_hit']:.2%}")
            print(f"Average Test Hit Rate: {results['overall_metrics']['avg_test_hit']:.2%} ± {results['overall_metrics']['std_test_hit']:.2%}")


    # Plotting
    if plot_results and fold_predictions:
        plot_dir = Path("results_plots") 
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_suffix = (
            f"{model_type}_"
            f"{window_strategy}-{train_window_years}tr{test_window_years}te_"
            f"{'AE' if use_autoencoder else 'NoAE'}"
        )
        n_plots = len(fold_predictions)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        for i, (fold_pred, fold_targ, fold_date) in enumerate(zip(fold_predictions, fold_targets, fold_dates)):
            train_pred, test_pred = fold_pred
            train_targ, test_targ = fold_targ
            train_date, test_date = fold_date
            
            ax = axes[i]

            # Plot actual returns (always black)
            ax.plot(train_date, train_targ, 'k-', label='Actual Returns', linewidth=1)
            ax.plot(test_date, test_targ, 'k-', linewidth=1)
            
            # Plot predictions (blue for train, red for test)
            ax.plot(train_date, train_pred, 'b-', label='Train Predictions', alpha=0.7)
            ax.plot(test_date, test_pred, 'r-', label='Test Predictions', alpha=0.7)
            
            # Add vertical line to separate train/test
            if len(test_date) > 0:
                ax.axvline(x=test_date[0], color='gray', linestyle='--', alpha=0.5)
            
            ax.set_title(f'Fold {i+1}: Predictions vs Actual Returns')
            ax.set_ylabel('Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if use_autoencoder:
            plt.savefig(plot_dir / f"{plot_suffix}_folds_predictions_vs_actual_ae_on.png")
        else:
            plt.savefig(plot_dir / f"{plot_suffix}_folds_predictions_vs_actual_ae_off.png")
        if plt_show:
            plt.show()
        
        # Overall performance plot
        if len(results['all_train_dates']) > 0:
            plt.figure(figsize=(15, 6))
            
            # Combine and sort all data by date
            all_data = []
            for train_pred, train_targ, train_date in zip(
                [results['fold_results'][i]['train_predictions'] for i in range(len(results['fold_results']))],
                [results['fold_results'][i]['train_targets'] for i in range(len(results['fold_results']))],
                [results['fold_results'][i]['train_dates'] for i in range(len(results['fold_results']))]
            ):
                for pred, targ, date in zip(train_pred, train_targ, train_date):
                    all_data.append((date, targ, pred, 'train'))
            
            for test_pred, test_targ, test_date in zip(
                [results['fold_results'][i]['test_predictions'] for i in range(len(results['fold_results']))],
                [results['fold_results'][i]['test_targets'] for i in range(len(results['fold_results']))],
                [results['fold_results'][i]['test_dates'] for i in range(len(results['fold_results']))]
            ):
                for pred, targ, date in zip(test_pred, test_targ, test_date):
                    all_data.append((date, targ, pred, 'test'))
            
            # Sort by date
            all_data.sort(key=lambda x: x[0])
            
            # Separate data
            dates_all = [x[0] for x in all_data]
            targets_all = [x[1] for x in all_data]
            predictions_all = [x[2] for x in all_data]
            types_all = [x[3] for x in all_data]
            
            # Plot
            plt.plot(dates_all, targets_all, 'k-', label='Actual Returns', linewidth=1)
            
            # Plot predictions with different colors
            train_mask = np.array(types_all) == 'train'
            test_mask = np.array(types_all) == 'test'
            
            if np.any(train_mask):
                plt.plot(np.array(dates_all)[train_mask], np.array(predictions_all)[train_mask], 
                        'b.', label='Train Predictions', alpha=0.7, markersize=3)
            if np.any(test_mask):
                plt.plot(np.array(dates_all)[test_mask], np.array(predictions_all)[test_mask], 
                        'r.', label='Test Predictions', alpha=0.7, markersize=3)
            
            plt.title('Overall Model Performance: Predictions vs Actual Returns')
            plt.ylabel('Returns')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if use_autoencoder:
                plt.savefig(plot_dir / f"{plot_suffix}_overall_performance_ae_on.png")
            else:
                plt.savefig(plot_dir / f"{plot_suffix}_overall_performance_ae_off.png")
            if plt_show:
                plt.show()
    
    return results


def hyperparameter_tuning(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    tbill3m: np.ndarray,
    model_class: type,
    model_type: str = 'feedforward',
    window_strategy: str = 'rolling',
    train_window_years: int = 3,
    test_window_years: int = 1,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Hyperparameter tuning for S&P-500 forecasting models."""
    if model_type.lower() not in ['feedforward', 'lstm', 'transformer', 'simpleconvolutional',
                                  'temporalfusiontransformer', 'tempconvnet']:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'feedforward', 'lstm',"
                         f" 'transformer', 'simpleconvolutional'.")
    if model_type.lower() == 'feedforward':
        parameter_grid = {
            'use_autoencoder': [True],
            'encoding_dim': [10],
            'ar_lags': [1, 5, 10],
            'seq_length': [0],
            'epochs': [60, 80, 120],
            'lr': [0.0001, 0.001],
            'batch_size': [32, 64],
            'alpha': [0.0, 0.0001],
            'l1_ratio': [0.0, 0.5],
            'hidden_dim': [50, 100, 150], 
            'dropout':[0.1]
        }
    elif model_type.lower() == 'lstm':
        parameter_grid = {
            'use_autoencoder': [False],
            'encoding_dim': [1],
            'seq_length': [24],
            'ar_lags': [1],
            'epochs': [40, 75, 100],
            'lr': [0.0001, 0.001],
            'batch_size': [32, 64, 128],
            'alpha': [0.0, 0.0001, 0.001],
            'l1_ratio': [0.0, 0.5],
            'hidden_dim': [50, 100, 150], 
            'num_layers': [1, 2,4], 
            'dropout':[0.0, 0.1]
        }
    elif model_type.lower() == 'transformer':
        parameter_grid = {
            'use_autoencoder': [False],
            'encoding_dim': [1],
            'seq_length': [24],
            'ar_lags': [1],
            'epochs': [40, 75, 100],
            'lr': [0.0001, 0.001],
            'batch_size': [32, 64, 128],
            'alpha': [0.0, 0.0001],
            'l1_ratio': [0.0, 0.5],
            'model_dim': [64, 128], 
            'num_heads': [2,4], 
            'num_layers': [2,4], 
            'dropout':[0.0, 0.1],
        }
    elif model_type.lower() == 'simpleconvolutional':
        parameter_grid = {
            'use_autoencoder': [True, False],
            'encoding_dim': [10, 15],
            'seq_length': [24],
            'ar_lags': [1],
            'epochs': [40, 75, 100],
            'lr': [0.0001, 0.001],
            'batch_size': [32, 64, 128],
            'alpha': [0.0, 0.0001],
            'l1_ratio': [0.0, 0.5],
            'num_channels': [[32], [32,64,32], [16,32,64,32,16]],
            'kernel_size': [3,5],
            'dropout':[0.0, 0.1]
        }
    elif model_type.lower() == 'tempconvnet':
        parameter_grid = {
            'use_autoencoder': [True, False],
            'encoding_dim': [10, 15],
            'seq_length': [24],
            'ar_lags': [1],
            'epochs': [40, 75, 100],
            'lr': [0.0001, 0.001],
            'batch_size': [32, 64, 128],
            'alpha': [0.0, 0.0001],
            'l1_ratio': [0.0, 0.5],
            'num_channels': [[32], [32,64,32], [16,32,64,32,16]],
            'kernel_size': [3,5],
            'dropout':[0.0, 0.1]
        }
    elif model_type.lower() == 'temporalfusiontransformer':
        parameter_grid = {
            'use_autoencoder': [False],
            'encoding_dim': [1],
            'seq_length': [24],
            'ar_lags': [1],
            'epochs': [40, 75, 100],
            'lr': [0.0001, 0.001],
            'batch_size': [32, 64, 128],
            'alpha': [0.0, 0.0001],
            'l1_ratio': [0.0, 0.5],
            'hidden_dim': [64, 128], 
            'num_heads': [2,4], 
            'num_layers': [3,4], 
            'dropout':[0.0, 0.1]
        }

    all_results = []
    for param_combination in itertools.product(*parameter_grid.values()):
        params = dict(zip(parameter_grid.keys(), param_combination))
        use_autoencoder = params['use_autoencoder']
        encoding_dim = params['encoding_dim']

        seq_length = params['seq_length']
        ar_lags = params['ar_lags']
        epochs = params['epochs']
        lr = params['lr']
        batch_size = params['batch_size']
        alpha = params['alpha']
        l1_ratio = params['l1_ratio']
        if model_type.lower() == 'feedforward':
            model_kwargs = {
                'hidden_dim': params['hidden_dim'],
                'dropout': params['dropout']
            }
        elif model_type.lower() == 'lstm':  
            model_kwargs = {
                'hidden_dim': params['hidden_dim'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout']
            }
        elif model_type.lower() == 'transformer':
            model_kwargs = {
                'model_dim': params['model_dim'],
                'num_heads': params['num_heads'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout']
            }
        elif model_type.lower() == 'simpleconvolutional':
            model_kwargs = {
                'num_channels': params['num_channels'],
                'kernel_size': params['kernel_size'],
                'dropout': params['dropout']
            }
        elif model_type.lower() == 'tempconvnet':
            model_kwargs = {
                'num_channels': params['num_channels'],
                'kernel_size': params['kernel_size'],
                'dropout': params['dropout']
            }
        elif model_type.lower() == 'temporalfusiontransformer':
            model_kwargs = {
                'hidden_dim': params['hidden_dim'],
                'num_heads': params['num_heads'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout']
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'feedforward', 'lstm', 'transformer', 'simpleconvolutional', 'tempconvnet', 'temporalfusiontransformer'.")
        # Run the training pipeline with the current parameters
        results = sp500_training_pipeline(
            X=X,
            y=y,
            dates=dates,
            tbill3m=tbill3m,
            model_class=model_class,
            model_type=model_type,
            model_kwargs=model_kwargs,
            window_strategy=window_strategy,
            train_window_years=train_window_years,
            test_window_years=test_window_years,
            use_autoencoder=use_autoencoder,
            encoding_dim=encoding_dim,
            seq_length=seq_length,
            ar_lags=ar_lags,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            plot_results=False,  # No need to plot during tuning
            alpha=alpha,
            l1_ratio=l1_ratio,
            device=device,
            do_print=False,  # Suppress print statements during tuning
        )

        all_results.append(
    (params, results['overall_metrics']['avg_test_mse'])
)   

    # Find the best parameters based on MSE
    best_params, best_mse = min(all_results, key=lambda x: x[1])
    return {
        'best_params': best_params,
        'best_mse': best_mse,
        'all_results': all_results
    }

def select_best_architectures(
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        tbill3m: np.ndarray,
        file_path: str = 'best_architectures.pickle'
) -> pd.DataFrame:
    """Find best architecture for each model"""
    results = []
    if PARAMS['use_autoencoder']:
        out_dir_models = Path('model_autoencoder_on')
        print('Autoencoder turned on')
    else:
        out_dir_models = Path('model_autoencoder_off')
        print('Autoencoder turned off')
    out_dir_models.mkdir(parents=True, exist_ok=True)

    for model_type, cfg in ARCHITECTURE_GRID.items():
        print(f'Tuning {model_type} architecture...')
        model_class = cfg['class']
        grid = cfg['grid']
        print(f'Running architecture selection for model {model_type}')

        best_mse = float('inf')
        best_params = None
        best_metrics = None
        best_models = None
        best_dates = None
        best_predictions = None
        errors = []

        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), combo))
            print(f'Testing params: {params}')
            try:
                res = sp500_training_pipeline(
                    X=X,
                    y=y,
                    dates=dates,
                    model_class=model_class,
                    model_kwargs=params,
                    model_type = model_type.lower(),
                    tbill3m = tbill3m,
                    **PARAMS
                )

                mse = res['overall_metrics']['avg_test_mse']
                print(f'  MSE: {mse:.6f}')
                if mse < best_mse:
                    best_mse = mse
                    best_params = params
                    best_metrics = res['overall_metrics']
                    best_models = res['models']
                    best_dates = res['all_test_dates']
                    best_predictions = res['all_test_predictions']


            except Exception as e:
                error = str(e)
                print(f'Error with params {params}: {error}')
                errors.append({'params': params, 'error': error})



        row = {
            'model_type': model_type,
            'best_params': best_params,
            'best_mse': None if best_params is None else best_mse,
            'errors': errors
        }
        if best_metrics is not None and best_params is not None:
            filtered = {
                k: v
                for k, v in best_metrics.items()
                if k.startswith('avg_test_') or k.startswith('std_test_')
            }
            row.update(filtered)

        results.append(row)

        if best_params is not None:
            for k, fold_model in enumerate(best_models, 1):
                fname = out_dir_models / f'{model_type}_best_fold_{k:02d}.joblib'
                save_model(fold_model, fname)
            save_predictions(
                dates=best_dates,
                predictions=best_predictions,
                csv_path=out_dir_models / f'{model_type}_best_predictions.csv'
            )

    results_df = pd.DataFrame(results)
    with open(file_path, 'wb') as f:
        pickle.dump(results_df, f)

    return results_df

def prepare_for_tuning(df_best_arch: pd.DataFrame) -> List[Tuple[Type, str, Dict[str, Any]]]:
    """Convert best architectures dataframe into a list of tuples"""
    out = []
    for _, row in df_best_arch.iterrows():
        model_type = row['model_type']
        model_class = ARCHITECTURE_GRID[model_type]['class']
        params = row['best_params']
        out.append((model_class, model_type.lower(), params))
    return out

def train_final_models(
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        tbill3m: np.ndarray,
        to_tune: List[Tuple[Type, str, Dict[str, Any]]],
        tuning_results: List[Dict[str, Any]],
        final_pickle_path: str,
        models_dir: str
) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Train each model on the full dataset using its best architecture and hyperparameters"""
    trained_models = []
    for (model_class, model_type, arch_params), tune_res in zip(to_tune, tuning_results):
        print(f'Training final model {model_type}')
        best_hyperparams = tune_res['best_params']
        model_kwargs = {**arch_params, **{k: v for k, v in best_hyperparams.items() if k in arch_params}}
        result = sp500_training_pipeline(
            X=X,
            y=y,
            dates=dates,
            tbill3m=tbill3m,
            model_class=model_class,
            model_type=model_type,
            model_kwargs=model_kwargs,
            window_strategy='expanding',
            train_window_years=3,
            test_window_years=0,
            device='cpu',
            plot_results=True,
            do_print=True
            **best_hyperparams
        )
        model = result['model']
        metrics = result['overall_metrics']
        model_path = os.path.join(models_dir, f'{model_type}.pt')
        torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, model_path)
        print(f'Saved final model {model_type} to {model_path}')
        trained_models.append((model_type, model, metrics))

    with open(final_pickle_path, 'wb') as f:
        pickle.dump(trained_models, f)
    print(f'Saved all final models and metrics list to {final_pickle_path}')
    return trained_models

def compare_models(csv_files: Dict[str, str],
                   results_dir: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dfs = []
    for label, path in csv_files.items():
        df = pd.read_csv(path)
        df = df.rename(columns={
            'model_type': 'model',
            'avg_test_sharpe': 'sharpe',
            'avg_test_mse':    'mse',
            'avg_test_sortino':'sortino',
            'avg_test_hit':    'hit_rate'
        })[['model','sharpe','mse','sortino','hit_rate']]
        df['scenario'] = label
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    summary_df = combined.set_index(['scenario', 'model'])

    comparisons = {}
    rankings = {}
    pct_improve = {}
    stats_res = {}

    for scenario, grp in summary_df.groupby(level=0):
        df_sc = grp.droplevel(0)
        rankings[scenario] = {
            metric: df_sc[metric]
            .sort_values(ascending=(metric != 'sharpe'))
            .index
            .tolist()
            for metric in df_sc.columns
        }
        # Percentage improvement best vs second best
        imps = {}
        for metric in df_sc.columns:
            vals = df_sc[metric]
            if metric == 'sharpe':
                best, second = vals.nlargest(2).values
                imps[metric] = (second - best) / abs(second) * 100
            else:
                best, second = vals.nsmallest(2).values
                imps[metric] = (second - best) / abs(second) * 100
        pct_improve[scenario] = imps

        res = {}
        # Paired t-test on Sharpe between 2 best models
        if len(df_sc) >= 2:
            top2 = np.argsort(-df_sc['sharpe'].values)[:2]
            t, p = stats.ttest_rel(
                df_sc['sharpe'].values[top2[0]],
                df_sc['sharpe'].values[top2[1]]
            )
            res['paired_t_sharpe'] = p
        # Anova on mse
        if len(df_sc) >= 3:
            f, p = stats.f_oneway(*[[v] for v in df_sc['mse'].values])
            res['anova_mse'] = p
        # Kruskal on Sortino
        if len(df_sc) >= 3:
            h, p = stats.kruskal(*[[v] for v in df_sc['sortino'].values])
            res['kruskal_sortino'] = p
        # Friedman on hit rate
        if len(df_sc) >= 2:
            chi2, p = stats.friedmanchisquare(*[[v] for v in df_sc['hit_rate'].values])
            res['friedman_hitrate'] = p

        stats_res[scenario] = res

        # 3) Plot per metric
        os.makedirs(results_dir, exist_ok=True)
        for metric in df_sc.columns:
            plt.figure()
            df_sc[metric].plot.bar(title=f"{scenario}: {metric} by model")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{scenario}_{metric}_bar.png"))
            plt.close()

    comparisons['rankings'] = rankings
    comparisons['percentage_improvement'] = pct_improve
    comparisons['stats_res'] = stats_res

    # 4) Cross-scenario paired-t tests for comparison between autoencoder on and off
    if len(csv_files) == 2:
        scen = list(csv_files.keys())
        a, b = scen
        df_a = summary_df.loc[a]
        df_b = summary_df.loc[b]
        common = df_a.index.intersection(df_b.index)
        inter = {}
        for metric in ['sharpe', 'mse', 'sortino', 'hit_rate']:
            t, p = stats.ttest_rel(df_b.loc[common, metric],
                                   df_a.loc[common, metric])
            inter[f'ttest_{metric}_{b}_vs_{a}'] = p
        comparisons['scenario_paired_t_tests'] = inter

        # Combined scatter matrix
        plt.figure()
        pd.plotting.scatter_matrix(
            combined.pivot(index='model', columns='scenario')[['sharpe', 'mse', 'sortino', 'hit_rate']],
            diagonal='kde', alpha=0.7)
        plt.suptitle('Metrics: AE_off vs AE_on')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'scenario_scatter_matrix.png'))
        plt.close()

    return summary_df, comparisons
