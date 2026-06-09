import pandas as pd
import glob
import os
import numpy as np
from scipy import stats
from typing import List, Tuple

returns_df = pd.read_csv('data_non_std.csv', parse_dates=['date'])


def load_and_label_csvs(folders: List[Tuple[str, bool]], returns_df: pd.DataFrame) -> pd.DataFrame:
    """Load all csvs from specified folders, label each by autoencoder flag, and combine them into a single dataframe"""
    all_records = []
    for folder_path, ae_flag in folders:
        pattern = os.path.join(folder_path, '*.csv')
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            model = os.path.splitext(filename)[0]

            df_pred = pd.read_csv(filepath)
            if 'date' not in df_pred.colums:
                df_pred.rename(columns={df_pred.columns[0]: 'date'}, inplace=True)
            df_pred['date'] = pd.to_datetime(df_pred['date'])
            df = df_pred.merge(returns_df, on='date', how='inner')
            df['model'] = model
            df['autoencoder'] = ae_flag
            df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')
            df['return'] = pd.to_numeric(df['return'], errors='coerce')
            all_records.append(df.dropna(subset=['prediction', 'return']))

    combined = pd.concat(all_records, ignore_index=True)
    return combined

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """"Compute mean, standard deviation, and count of prediction errors by model and autoencoder flag"""
    df_copy = df.copy()
    df_copy['error'] = df_copy['prediction'] - df_copy['actual']
    return 0


folder_off = 'results_autoencoder_off'
folder_on  = 'results_autoencoder_on'
df_all = load_and_label_csvs(
    [(folder_off, False), (folder_on, True)],
    returns_df)

































folder_ae_off = 'results_autoencoder_off'
folder_ae_on = 'results_autoencoder_on'



