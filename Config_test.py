import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from TM_models import TemporalConvNet, TemporalFusionTransformer
from Simple_models import SimpleFeedForward, SimpleLSTM, SimpleConvolutional, SimpleTransformer
from environment import sp500_training_pipeline

RESULTS_DIR = 'results'
ARCHITECTURE_RESULTS_FILE = 'architecture_selection_results.csv'
HYPERPARAMETER_RESULTS_FILE = 'hyperparameter_optimization_results.csv'

# Create results directory if it does not exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed training parameters for architecture selection
FIXED_PARAMS = {
    'window_strategy': 'rolling',
    'train_window_years': 3,
    'test_window_years': 1,
    'use_autoencoder': True,
    'encoding_dim': 10,
    'seq_length': 18,
    'epochs': 20,
    'lr': 0.0001,
    'batch_size': 32,
    'device': 'cpu',
    'plot_results': False
}

def evaluate_architecture(model_name, model_class, model_params, features, target, dates):
    """Evaluate a single model architecture configuration"""
    print(f'Evaluating {model_name} with parameters: {model_params}')
    try:
        pipeline_res = sp500_training_pipeline(
            X=features,
            y=target,
            dates=dates,
            model_class=model_class,
            model_kwargs=model_params,
            model_type=model_name.lower(),
            **FIXED_PARAMS
        )

        # Extract metrics
        test_sharpe = pipeline_res['overall_metrics']['avg_test_sharpe']
        test_sharpe_std = pipeline_res['overall_metrics']['std_test_sharpe']
        test_mse = pipeline_res['overall_metrics']['avg_test_mse']
        test_mse_std = pipeline_res['overall_metrics']['std_test_mse']
        test_sortino = pipeline_res['overall_metrics']['avg_test_sortino']
        test_hit_rate = pipeline_res['overall_metrics']['avg_test_hit']
        test_r2 = pipeline_res['overall_metrics']['avg_test_r2']

        return {
            'model': model_name,
            **model_params,
            'avg_test_sharpe': test_sharpe,
            'std_test_sharpe': test_sharpe_std,
            'avg_test_mse': test_mse,
            'std_test_mse': test_mse_std,
            'avg_test_sortino': test_sortino,
            'avg_test_hit_rate': test_hit_rate,
            'avg_test_r2': test_r2,
            'success': True
        }

    except Exception as e:
        print(f"Error evaluating {model_name} with {model_params}: {str(e)}")
        return {
            'model': model_name,
            **model_params,
            'avg_test_sharpe': -999,  # Very low score for failed runs
            'std_test_sharpe': 999,
            'avg_test_mse': 999,
            'std_test_mse': 999,
            'avg_test_sortino': -999,
            'avg_test_hit_rate': 0,
            'avg_test_r2': -999,
            'success': False,
            'error': str(e)
        }


def run_architecture_selection():
    """Phase 1: Architecture Selection with fixed hyperparameters"""
    print("PHASE 1: ARCHITECTURE SELECTION")

    # Load data
    df = pd.read_csv('data_non_std.csv', parse_dates=['Unnamed: 0']).rename(columns={'Unnamed: 0': 'Date'})
    features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
    target = df['returns'].values.astype(np.float32)
    dates = pd.to_datetime(df['Date'])

    # Define architecture variants with minimal hyperparameter exploration
    architecture_configs = {
        'TCN_basic': {
            'class': TemporalConvNet,
            'param_grid': {
                'num_channels': [[64, 128, 64]],
                'pool': ['last']
            }
        },
        'TCN_deep': {
            'class': TemporalConvNet,
            'param_grid': {
                'num_channels': [[32, 64, 128, 64, 32]],  # Deeper
                'pool': ['last']
            }
        },
        'TCN_wide': {
            'class': TemporalConvNet,
            'param_grid': {
                'num_channels': [[128, 256, 128]],  # Wider
                'pool': ['last']
            }
        },

        'TFT_basic': {
            'class': TemporalFusionTransformer,
            'param_grid': {
                'hidden_dim': [64],
                'num_heads': [4],
                'num_layers': [2],
                'dropout': [0.1]
            }
        },
        'TFT_large': {
            'class': TemporalFusionTransformer,
            'param_grid': {
                'hidden_dim': [128],
                'num_heads': [8],
                'num_layers': [3],
                'dropout': [0.1]
            }
        },

        'FeedForward_basic': {
            'class': SimpleFeedForward,
            'param_grid': {
                'hidden_dim': [200],
                'dropout': [0.1]
            }
        },
        'FeedForward_large': {
            'class': SimpleFeedForward,
            'param_grid': {
                'hidden_dim': [512],  # Larger
                'dropout': [0.1]
            }
        },

        'LSTM_basic': {
            'class': SimpleLSTM,
            'param_grid': {
                'hidden_dim': [200],
                'num_layers': [2],
                'dropout': [0.1]
            }
        },
        'LSTM_deep': {
            'class': SimpleLSTM,
            'param_grid': {
                'hidden_dim': [256],
                'num_layers': [3],    # Deeper
                'dropout': [0.1]
            }
        },

        'Transformer_basic': {
            'class': SimpleTransformer,
            'param_grid': {
                'model_dim': [128],
                'nhead': [8],
                'num_layers': [2],
                'dropout': [0.1],
                'max_seq_length': [500]
            }
        },
        'Transformer_large': {
            'class': SimpleTransformer,
            'param_grid': {
                'model_dim': [256],       # Larger
                'nhead': [8],
                'num_layers': [3],        # Deeper
                'dropout': [0.1],
                'max_seq_length': [500]
            }
        },

        'CNN_basic': {
            'class': SimpleConvolutional,
            'param_grid': {
                'num_channels': [[32, 64, 32]],
                'kernel_size': [5],
                'dropout': [0.25],
                'seq_length': [12]
            }
        },
        'CNN_deep': {
            'class': SimpleConvolutional,
            'param_grid': {
                'num_channels': [[32, 64, 128, 64]],  # Deeper
                'kernel_size': [5],
                'dropout': [0.25],
                'seq_length': [12]
            }
        },
        'CNN_wide': {
            'class': SimpleConvolutional,
            'param_grid': {
                'num_channels': [[64, 128, 256, 128, 64]],  # Wider
                'kernel_size': [5],
                'dropout': [0.25],
                'seq_length': [12]
            }
        }
    }

    # Check if architecture results already exist
    arch_results_path = os.path.join(RESULTS_DIR, ARCHITECTURE_RESULTS_FILE)
    if os.path.exists(arch_results_path):
        print(f"Loading existing architecture results from {arch_results_path}")
        df_arch_results = pd.read_csv(arch_results_path)
    else:
        print("Running architecture selection...")
        arch_results = []

        # Evaluate each architecture
        for arch_name, config in architecture_configs.items():
            ModelClass = config['class']
            keys, values = zip(*config['param_grid'].items())

            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                result = evaluate_architecture(arch_name, ModelClass, params, features, target, dates)
                arch_results.append(result)

        df_arch_results = pd.DataFrame(arch_results)
        df_arch_results.to_csv(arch_results_path, index=False)
        print(f"Architecture results saved to {arch_results_path}")

    # Filter successful runs and sort by Sharpe ratio
    successful_runs = df_arch_results[
        df_arch_results['success'] == True] if 'success' in df_arch_results.columns else df_arch_results
    df_arch_sorted = successful_runs.sort_values('avg_test_sharpe', ascending=False).reset_index(drop=True)

    print("\nArchitecture Selection Results (sorted by Test Sharpe Ratio):")
    print("=" * 80)
    display_cols = ['model', 'avg_test_sharpe', 'std_test_sharpe', 'avg_test_mse', 'avg_test_sortino',
                    'avg_test_hit_rate']
    for col in display_cols:
        if col in df_arch_sorted.columns:
            continue
        else:
            print(f"Warning: Column {col} not found in results")

    print(df_arch_sorted[display_cols].head(10))

    return df_arch_sorted



def run_hyperparameter_optimization(best_architecture_info):
    """Phase 2: Hyperparameter optimization for the best architecture"""
    print("PHASE 2: HYPERPARAMETER OPTIMIZATION")

    # Load data
    df = pd.read_csv('data_non_std.csv', parse_dates=['Unnamed: 0']).rename(columns={'Unnamed: 0': 'Date'})
    features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
    target = df['returns'].values.astype(np.float32)
    dates = pd.to_datetime(df['Date'])

    best_model = best_architecture_info['model']
    print(f"Optimizing hyperparameters for: {best_model}")

    # Define extensive hyperparameter grids for each model type
    hyperparameter_grids = {
        'TCN_basic': {
            'class': TemporalConvNet,
            'param_grid': {
                'num_channels': [[64, 128, 64], [32, 64, 32], [128, 256, 128], [64, 128, 256, 128, 64]],
                'kernel_size': [3, 5, 7],
                'dropout': [0.1, 0.2, 0.3],
                'pool': ['last', 'avg']
            }
        },
        'TCN_deep': {
            'class': TemporalConvNet,
            'param_grid': {
                'num_channels': [[32, 64, 128, 64, 32], [64, 128, 256, 128, 64], [32, 64, 128, 256, 128, 64, 32]],
                'kernel_size': [3, 5],
                'dropout': [0.1, 0.2, 0.3],
                'pool': ['last', 'avg']
            }
        },
        'TCN_wide': {
            'class': TemporalConvNet,
            'param_grid': {
                'num_channels': [[128, 256, 128], [256, 512, 256], [128, 256, 512, 256, 128]],
                'kernel_size': [3, 5],
                'dropout': [0.1, 0.2],
                'pool': ['last', 'avg']
            }
        },

        'TFT_basic': {
            'class': TemporalFusionTransformer,
            'param_grid': {
                'hidden_dim': [64, 128, 256],
                'num_heads': [4, 8],
                'num_layers': [1, 2, 3],
                'dropout': [0.1, 0.2, 0.3]
            }
        },
        'TFT_large': {
            'class': TemporalFusionTransformer,
            'param_grid': {
                'hidden_dim': [128, 256, 512],
                'num_heads': [8, 16],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2]
            }
        },

        'FeedForward_basic': {
            'class': SimpleFeedForward,
            'param_grid': {
                'hidden_dim': [64, 128, 200, 256],
                'dropout': [0.0, 0.1, 0.2, 0.3]
            }
        },
        'FeedForward_large': {
            'class': SimpleFeedForward,
            'param_grid': {
                'hidden_dim': [256, 512, 768],
                'dropout': [0.1, 0.2, 0.3]
            }
        },

        'LSTM_basic': {
            'class': SimpleLSTM,
            'param_grid': {
                'hidden_dim': [64, 128, 200, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.0, 0.1, 0.2, 0.3]
            }
        },
        'LSTM_deep': {
            'class': SimpleLSTM,
            'param_grid': {
                'hidden_dim': [128, 256, 512],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3]
            }
        },

        'Transformer_basic': {
            'class': SimpleTransformer,
            'param_grid': {
                'model_dim': [64, 128, 256],
                'nhead': [4, 8],
                'num_layers': [1, 2, 3],
                'dropout': [0.1, 0.2, 0.3],
                'max_seq_length': [100, 200, 500]
            }
        },
        'Transformer_large': {
            'class': SimpleTransformer,
            'param_grid': {
                'model_dim': [256, 512, 768],
                'nhead': [8, 16],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2],
                'max_seq_length': [200, 500]
            }
        },

        'CNN_basic': {
            'class': SimpleConvolutional,
            'param_grid': {
                'num_channels': [[16, 32, 16], [32, 64, 32], [64, 128, 64]],
                'kernel_size': [3, 5, 7],
                'dropout': [0.1, 0.2, 0.25, 0.3],
                'seq_length': [12, 24, 50]
            }
        },
        'CNN_deep': {
            'class': SimpleConvolutional,
            'param_grid': {
                'num_channels': [[32, 64, 128, 64], [64, 128, 256, 128], [32, 64, 128, 256, 128, 64]],
                'kernel_size': [3, 5],
                'dropout': [0.2, 0.25, 0.3],
                'seq_length': [12, 24, 50]
            }
        },
        'CNN_wide': {
            'class': SimpleConvolutional,
            'param_grid': {
                'num_channels': [[64, 128, 256, 128, 64], [128, 256, 512, 256, 128]],
                'kernel_size': [3, 5],
                'dropout': [0.2, 0.25],
                'seq_length': [24, 50]
            }
        }
    }

    # Check if hyperparameter results already exist
    hyperparam_results_path = os.path.join(RESULTS_DIR, HYPERPARAMETER_RESULTS_FILE)
    if os.path.exists(hyperparam_results_path):
        print(f"Loading existing hyperparameter results from {hyperparam_results_path}")
        df_hyperparam_results = pd.read_csv(hyperparam_results_path)
    else:
        if best_model not in hyperparameter_grids:
            print(f"No hyperparameter grid defined for {best_model}")
            return None

        config = hyperparameter_grids[best_model]
        ModelClass = config['class']

        # Use more epochs for final optimization
        optimization_params = FIXED_PARAMS.copy()
        optimization_params['epochs'] = 50
        optimization_params['plot_results'] = False

        print("Running hyperparameter optimization...")
        hyperparam_results = []

        keys, values = zip(*config['param_grid'].items())
        total_combinations = np.prod([len(v) for v in values])
        print(f"Total combinations to evaluate: {total_combinations}")

        for i, combo in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combo))
            print(f"Progress: {i + 1}/{total_combinations}")

            try:
                pipeline_res = sp500_training_pipeline(
                    X=features,
                    y=target,
                    dates=dates,
                    model_class=ModelClass,
                    model_kwargs=params,
                    model_type=best_model.lower(),
                    **optimization_params
                )

                result = {
                    'model': best_model,
                    **params,
                    'avg_test_sharpe': pipeline_res['overall_metrics']['avg_test_sharpe'],
                    'std_test_sharpe': pipeline_res['overall_metrics']['std_test_sharpe'],
                    'avg_test_mse': pipeline_res['overall_metrics']['avg_test_mse'],
                    'std_test_mse': pipeline_res['overall_metrics']['std_test_mse'],
                    'avg_test_sortino': pipeline_res['overall_metrics']['avg_test_sortino'],
                    'avg_test_hit_rate': pipeline_res['overall_metrics']['avg_test_hit'],
                    'avg_test_r2': pipeline_res['overall_metrics']['avg_test_r2'],
                    'success': True
                }

            except Exception as e:
                print(f"Error with params {params}: {str(e)}")
                result = {
                    'model': best_model,
                    **params,
                    'avg_test_sharpe': -999,
                    'std_test_sharpe': 999,
                    'avg_test_mse': 999,
                    'std_test_mse': 999,
                    'avg_test_sortino': -999,
                    'avg_test_hit_rate': 0,
                    'avg_test_r2': -999,
                    'success': False,
                    'error': str(e)
                }

            hyperparam_results.append(result)

        df_hyperparam_results = pd.DataFrame(hyperparam_results)

    df_hyperparam_results.to_csv(hyperparam_results_path, index=False)
    print(f"Hyperparameter results saved to {hyperparam_results_path}")

    # Filter successful runs and sort by Sharpe ratio
    successful_runs = df_hyperparam_results[df_hyperparam_results['success'] == True] if 'success' in df_hyperparam_results.columns else df_hyperparam_results
    df_hyperparam_sorted = successful_runs.sort_values('avg_test_sharpe', ascending=False).reset_index(drop=True)

    print("\nHyperparameter Optimization Results (sorted by Test Sharpe Ratio):")
    print("=" * 100)
    display_cols = ['model', 'avg_test_sharpe', 'std_test_sharpe', 'avg_test_mse', 'avg_test_sortino',
                    'avg_test_hit_rate']
    print(df_hyperparam_sorted[display_cols].head(10))

    return df_hyperparam_sorted


def create_plots(df_arch_results, df_hyperparam_results=None):
    """Create plots for the results"""

    # Architecture comparison plot
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    models = df_arch_results['model'].values
    sharpe_ratios = df_arch_results['avg_test_sharpe'].values
    sharpe_stds = df_arch_results['std_test_sharpe'].values

    bars = plt.bar(range(len(models)), sharpe_ratios, yerr=sharpe_stds, capsize=5, alpha=0.7)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Average Test Sharpe Ratio')
    plt.title('Architecture Comparison: Test Sharpe Ratio')
    plt.grid(True, alpha=0.3)

    for i, (bar, sharpe) in enumerate(zip(bars, sharpe_ratios)):
        if sharpe > 0:
            bar.set_color('green')
        elif sharpe < 0:
            bar.set_color('red')
        else:
            bar.set_color('gray')

    plt.subplot(2, 2, 2)
    plt.scatter(df_arch_results['avg_test_mse'], df_arch_results['avg_test_sharpe'], alpha=0.7)
    for i, model in enumerate(df_arch_results['model']):
        plt.annotate(model, (df_arch_results['avg_test_mse'].iloc[i], df_arch_results['avg_test_sharpe'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Average Test MSE')
    plt.ylabel('Average Test Sharpe Ratio')
    plt.title('Sharpe Ratio vs MSE Trade-off')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    metrics = ['avg_test_sharpe', 'avg_test_sortino', 'avg_test_hit_rate']
    x = np.arange(len(models))
    width = 0.25

    for i, metric in enumerate(metrics):
        if metric in df_arch_results.columns:
            values = df_arch_results[metric].values
            plt.bar(x + i * width, values, width, label=metric.replace('avg_test_', '').title(), alpha=0.7)

    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.ylabel('Metric Value')
    plt.title('Multiple Metrics Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if df_hyperparam_results is not None and len(df_hyperparam_results) > 1:
        plt.subplot(2, 2, 4)
        top_configs = df_hyperparam_results.head(10)
        config_labels = [f"Config {i + 1}" for i in range(len(top_configs))]

        plt.bar(config_labels, top_configs['avg_test_sharpe'],
                yerr=top_configs['std_test_sharpe'], capsize=5, alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Test Sharpe Ratio')
        plt.title('Top 10 Hyperparameter Configurations')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("Starting Two-Phase Model Selection Process")

    # Phase 1: Architecture Selection
    df_arch_results = run_architecture_selection()

    if len(df_arch_results) == 0:
        print("No successful architecture evaluations. Exiting.")
        return

    # Get the best architecture
    best_architecture = df_arch_results.iloc[0].to_dict()
    print(f"\nBest Architecture: {best_architecture['model']}")
    print(f"Test Sharpe Ratio: {best_architecture['avg_test_sharpe']:.4f} ± {best_architecture['std_test_sharpe']:.4f}")

    # Phase 2: Hyperparameter Optimization
    df_hyperparam_results = run_hyperparameter_optimization(best_architecture)

    # Create plots
    create_plots(df_arch_results, df_hyperparam_results)

    # Final summary
    print("FINAL SUMMARY")

    print("Top 3 Architectures:")
    display_cols = ['model', 'avg_test_sharpe', 'std_test_sharpe', 'avg_test_mse']
    print(df_arch_results[display_cols].head(3).to_string(index=False))

    if df_hyperparam_results is not None and len(df_hyperparam_results) > 0:
        print(f"Best Hyperparameter Configuration for {best_architecture['model']}:")
        best_config = df_hyperparam_results.iloc[0]
        print(f"Test Sharpe Ratio: {best_config['avg_test_sharpe']:.4f} ± {best_config['std_test_sharpe']:.4f}")
        print(f"Configuration: {best_config.to_dict()}")

        # Save the best configuration
        best_config_path = os.path.join(RESULTS_DIR, 'best_configuration.csv')
        pd.DataFrame([best_config]).to_csv(best_config_path, index=False)
        print(f"Best configuration saved to {best_config_path}")


if __name__ == "__main__":
    main()