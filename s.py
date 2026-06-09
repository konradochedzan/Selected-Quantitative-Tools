import pandas as pd
import numpy as np
from environment import sp500_training_pipeline
from pathlib import Path
from TM_models import TemporalConvNet, TemporalFusionTransformer
from Simple_models import SimpleConvolutional, SimpleFeedForward, SimpleLSTM, SimpleTransformer
from saving import save_predictions, save_model, load_model, load_predictions

import pandas as pd
import ast
import ast
import json
import re
from pathlib import Path
import pandas as pd
import ast
import numpy as np

df = pd.read_csv('expanding/trained_model_summary_results.csv')

import re
dictionary = {}
# for k in range(df.shape[0]):

#     results_str = df.loc[k, 'results']
#     model_name = df.loc[k, 'model']
#     autoencoder = df.loc[k, 'use_autoencoder']
#     match = re.search(r"avg_test_mse':\s*np\.float64\(([^)]+)\),", results_str)
#     avg_test_mse = match.group(1)
#     avg_test_mse = float(avg_test_mse)
#     dictionary[model_name + str(autoencoder)] = {'avg_test_mse':avg_test_mse}


# print(dictionary)

for k in range(df.shape[0]):
    results_str = df.loc[k, 'results']
    model_name = df.loc[k, 'model']
    autoencoder = df.loc[k, 'use_autoencoder']
    
    # Extract avg_train_mse
    match = re.search(r"'avg_train_mse':\s*np\.float64\(([^)]+)\),", results_str)
    avg_train_mse = float(match.group(1))
    
    # Extract avg_test_mse
    match = re.search(r"'avg_test_mse':\s*np\.float64\(([^)]+)\),", results_str)
    avg_test_mse = float(match.group(1))
    
    # Extract avg_train_sharpe
    match = re.search(r"'avg_train_sharpe':\s*np\.float64\(([^)]+)\),", results_str)
    avg_train_sharpe = float(match.group(1))
    
    # Extract avg_test_sharpe
    match = re.search(r"'avg_test_sharpe':\s*np\.float64\(([^)]+)\),", results_str)
    avg_test_sharpe = float(match.group(1))
    
    # Extract std_train_mse
    match = re.search(r"'std_train_mse':\s*np\.float64\(([^)]+)\),", results_str)
    std_train_mse = float(match.group(1))
    
    # Extract std_test_mse
    match = re.search(r"'std_test_mse':\s*np\.float64\(([^)]+)\),", results_str)
    std_test_mse = float(match.group(1))
    
    # Extract std_train_sharpe
    match = re.search(r"'std_train_sharpe':\s*np\.float64\(([^)]+)\),", results_str)
    std_train_sharpe = float(match.group(1))
    
    # Extract std_test_sharpe
    match = re.search(r"'std_test_sharpe':\s*np\.float64\(([^)]+)\),", results_str)
    std_test_sharpe = float(match.group(1))
    
    # Extract avg_train_mae
    match = re.search(r"'avg_train_mae':\s*np\.float64\(([^)]+)\),", results_str)
    avg_train_mae = float(match.group(1))
    
    # Extract avg_test_mae
    match = re.search(r"'avg_test_mae':\s*np\.float64\(([^)]+)\),", results_str)
    avg_test_mae = float(match.group(1))
    
    # Extract std_train_mae
    match = re.search(r"'std_train_mae':\s*np\.float64\(([^)]+)\),", results_str)
    std_train_mae = float(match.group(1))
    
    # Extract std_test_mae
    match = re.search(r"'std_test_mae':\s*np\.float64\(([^)]+)\),", results_str)
    std_test_mae = float(match.group(1))
    
    # Extract avg_train_r2
    match = re.search(r"'avg_train_r2':\s*np\.float64\(([^)]+)\),", results_str)
    avg_train_r2 = float(match.group(1))
    
    # Extract avg_test_r2
    match = re.search(r"'avg_test_r2':\s*np\.float64\(([^)]+)\),", results_str)
    avg_test_r2 = float(match.group(1))
    
    # Extract std_train_r2
    match = re.search(r"'std_train_r2':\s*np\.float64\(([^)]+)\),", results_str)
    std_train_r2 = float(match.group(1))
    
    # Extract std_test_r2
    match = re.search(r"'std_test_r2':\s*np\.float64\(([^)]+)\),", results_str)
    std_test_r2 = float(match.group(1))
    
    # Extract avg_train_sortino
    match = re.search(r"'avg_train_sortino':\s*np\.float64\(([^)]+)\),", results_str)
    avg_train_sortino = float(match.group(1))
    
    # Extract avg_test_sortino
    match = re.search(r"'avg_test_sortino':\s*np\.float64\(([^)]+)\),", results_str)
    avg_test_sortino = float(match.group(1))
    
    # Extract std_train_sortino
    match = re.search(r"'std_train_sortino':\s*np\.float64\(([^)]+)\),", results_str)
    std_train_sortino = float(match.group(1))
    
    # Extract std_test_sortino
    match = re.search(r"'std_test_sortino':\s*np\.float64\(([^)]+)\),", results_str)
    std_test_sortino = float(match.group(1))
    
    # Extract avg_train_hit
    match = re.search(r"'avg_train_hit':\s*np\.float64\(([^)]+)\),", results_str)
    avg_train_hit = float(match.group(1))
    
    # Extract avg_test_hit
    match = re.search(r"'avg_test_hit':\s*np\.float64\(([^)]+)\),", results_str)
    avg_test_hit = float(match.group(1))
    
    # Extract std_train_hit
    match = re.search(r"'std_train_hit':\s*np\.float64\(([^)]+)\),", results_str)
    std_train_hit = float(match.group(1))
    
    # Extract std_test_hit
    match = re.search(r"'std_test_hit':\s*np\.float64\(([^)]+)\)", results_str)  # Note: no comma at the end for the last one
    std_test_hit = float(match.group(1))
    
    dictionary[model_name + str(autoencoder)] = {
        'avg_train_mse': avg_train_mse,
        'avg_test_mse': avg_test_mse,
        'avg_train_sharpe': avg_train_sharpe,
        'avg_test_sharpe': avg_test_sharpe,
        'std_train_mse': std_train_mse,
        'std_test_mse': std_test_mse,
        'std_train_sharpe': std_train_sharpe,
        'std_test_sharpe': std_test_sharpe,
        'avg_train_mae': avg_train_mae,
        'avg_test_mae': avg_test_mae,
        'std_train_mae': std_train_mae,
        'std_test_mae': std_test_mae,
        'avg_train_r2': avg_train_r2,
        'avg_test_r2': avg_test_r2,
        'std_train_r2': std_train_r2,
        'std_test_r2': std_test_r2,
        'avg_train_sortino': avg_train_sortino,
        'avg_test_sortino': avg_test_sortino,
        'std_train_sortino': std_train_sortino,
        'std_test_sortino': std_test_sortino,
        'avg_train_hit': avg_train_hit,
        'avg_test_hit': avg_test_hit,
        'std_train_hit': std_train_hit,
        'std_test_hit': std_test_hit
    }
    
df_results = pd.DataFrame.from_dict(dictionary, orient='index')

# Reset index to make the model names a regular column
df_results.reset_index(inplace=True)
df_results.rename(columns={'index': 'model'}, inplace=True)

# Save to CSV
df_results.to_csv('expanding/model_results.csv', index=False)

print("Results saved to model_results.csv")
print(df_results.head())