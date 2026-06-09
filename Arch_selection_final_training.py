import ast
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from environment import (select_best_architectures, prepare_for_tuning, hyperparameter_tuning, train_final_models,
                         compare_models, PARAMS, sp500_training_pipeline)
from Simple_models import SimpleConvolutional, SimpleLSTM, SimpleTransformer, SimpleFeedForward
from TM_models import TemporalConvNet, TemporalFusionTransformer

from saving import save_model, save_predictions

RESULTS_DIR = 'results'
if PARAMS['use_autoencoder']:
    ARCHITECTURE_RESULTS_FILE = 'architecture_results_autoencoder_on.csv'
else:
    ARCHITECTURE_RESULTS_FILE = 'architecture_results_autoencoder_off.csv'
BEST_ARCHITECTURES_FILE = 'best_architectures.csv'
HYPERPARAM_RESULTS_FILE_CSV = 'hyperparam_results.csv'
HYPERPARAM_RESULTS_FILE_PICKLE = 'hyperparam_results.pickle'
FINAL_PICKLE = 'final_models.pickle'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

MODEL_MAPPING = {
    'TCN':        TemporalConvNet,
    'LSTM':       SimpleLSTM,
    'FeedForward':SimpleFeedForward,
    'TFT':        TemporalFusionTransformer,
    'Transformer':SimpleTransformer,
    'CNN':        SimpleConvolutional,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


df = pd.read_csv('data_non_std.csv', parse_dates=['date'])
df.rename(columns={'date': 'Date'}, inplace=True)
features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
target = df['returns'].values.astype(np.float32)
dates = pd.to_datetime(df['Date'])
tbill3m = df['tbill3m'].values.astype(np.float32)


print('Running architecture selection')
arch_path = os.path.join(RESULTS_DIR, ARCHITECTURE_RESULTS_FILE)
if os.path.exists(arch_path):
    df_arch = pd.read_csv(arch_path)
    print('Loaded existing architecture results.')
else:
    print('Selecting best architectures...')
    df_arch = select_best_architectures(
        X=features,
        y=target,
        dates=dates,
        tbill3m=tbill3m,
    )
    df_arch.to_csv(arch_path, index=False)
    print(f'Saved architecture results to {arch_path}')

all_results = []
out_dir = Path('trained_outputs')
out_dir.mkdir(exist_ok=True, parents=True)

for use_ae in (False, True):
    PARAMS['use_autoencoder'] = use_ae

    # csv_file = (
    #     'architecture_results_autoencoder_on.csv'
    #     if use_ae else
    #     'architecture_results_autoencoder_off.csv'
    # )
    csv_file = 'architecture_results.csv'

    df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file))
    for idx, row in df.iterrows():
        model_type = row['model_type']
        model_kwargs = ast.literal_eval(row['best_params'])
        model_class = MODEL_MAPPING[model_type]

        print(f'→ Training {model_type} (AE={use_ae}) with params={model_kwargs}')
        results = sp500_training_pipeline(
            X=features,
            y=target,
            tbill3m=tbill3m,
            dates=dates,
            model_class=model_class,
            model_kwargs=model_kwargs,
            model_type=model_type.lower(),
            **PARAMS
        )

        for fold_idx, fold_model in enumerate(results['models'], start=1):
            model_path = out_dir / f'{model_type}_ae_{use_ae}_fold_{fold_idx:02d}.joblib'
            save_model(fold_model, model_path)
            print(f'Saved model → {model_path}')

        preds_path = out_dir / f'{model_type}_ae_{use_ae}_predictions.csv'
        save_predictions(
            dates=results['all_test_dates'],
            predictions=results['all_test_predictions'],
            csv_path=preds_path
        )
        print(f'Saved predictions → {preds_path}')

        all_results.append({
            'model': model_type,
            'use_autoencoder': use_ae,
            'params': model_kwargs,
            'results': results
        })

all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv('trained_model_summary_results.csv')

