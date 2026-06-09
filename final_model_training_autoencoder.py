import pandas as pd
import numpy as np
from environment import sp500_training_pipeline
from pathlib import Path
from TM_models import TemporalConvNet, TemporalFusionTransformer
from Simple_models import SimpleConvolutional, SimpleFeedForward, SimpleLSTM, SimpleTransformer

from saving import save_model, load_model, save_predictions, load_predictions

all_models_with_kwargs = {
    'temporalconvnet': [TemporalConvNet, {
        'num_channels': [32, 64, 32],
        'kernel_size': 5,
        'dropout': 0.1
    }],
        'feedforward': [SimpleFeedForward, {
        'hidden_dim': 300,
        'dropout': 0.1}],
     'cnn': [SimpleConvolutional, {
        'num_channels': [32, 64, 32],
        'kernel_size': 5,
        'dropout': 0.1
    }]
    ,
    'lstm': [SimpleLSTM, {
        'hidden_dim': 200,
        'dropout': 0.1,
        'num_layers': 3}],
    'transformer': [SimpleTransformer, {
        'nhead': 8,
        'model_dim': 256,
        'dropout': 0.1,
        'num_layers': 3}],
    'temporalfusiontransformer': [TemporalFusionTransformer, {
        'hidden_dim': 128,
        'num_heads': 4,
        'dropout': 0.1,
        'num_layers': 2}],}
    
FIXED_PARAMS = {
    'window_strategy': 'rolling',
    'train_window_years': 3,
    'test_window_years': 1,
    'use_autoencoder': True,
    'encoding_dim': 10,
    'seq_length': 24,
    'epochs': 120,
    'lr': 0.0001,
    'batch_size': 128,
    'device': 'cpu',
    'plot_results': True,
    'do_print': False
}

df = pd.read_csv('data_non_std.csv', parse_dates=['Unnamed: 0'])
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
target = df['returns'].values.astype(np.float32)
dates = pd.to_datetime(df['Date'])
tbill3m = df['tbill3m'].values.astype(np.float32)

out_dir_models = Path("models_autoencoder")
out_dir_models.mkdir(parents=True, exist_ok=True)

for model_type, (model_class, model_kwargs) in all_models_with_kwargs.items():
    print(f"Training {model_type} model...")
    results = sp500_training_pipeline(
        X            = features,
        y            = target,
        dates        = dates,
        tbill3m      = tbill3m,
        model_class  = model_class,
        model_type   = model_type,
        model_kwargs = model_kwargs,
        **FIXED_PARAMS,
    )
    for k, fold_model in enumerate(results["models"], 1):
        fname = out_dir_models / f"{model_type}_fold_{k:02d}.joblib"
        save_model(fold_model, fname)

    save_predictions(
        dates        = results["all_test_dates"],
        predictions  = results["all_test_predictions"],
        csv_path     = f"results_autoencoder/{model_type}_predictions.csv",
    )
    print(f"{model_type} model training complete. Results saved.")
