import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_real_vs_predictions(
    real_csv_path,
    model_names,
    predictions_dir=".",
    output_dir="plots",
    real_col="returns",
    date_col="date",
    plot_title="Real vs Predictions",
    y_label="Value"
):

    real_csv_path = Path(real_csv_path)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real data
    real_df = pd.read_csv(real_csv_path, parse_dates=[date_col])
    real_df = real_df[[date_col, real_col]].rename(columns={real_col: "real"})
    real_df = real_df.sort_values(date_col)

    # For each model, plot and save
    for model in model_names:
        pred_path = predictions_dir / f"{model}_predictions.csv"
        if not pred_path.exists():
            print(f"Prediction file for model '{model}' not found at {pred_path}. Skipping.")
            continue

        pred_df = pd.read_csv(pred_path, parse_dates=[date_col])
        pred_df = pred_df[[date_col, "prediction"]].rename(columns={"prediction": model})
        pred_df = pred_df.sort_values(date_col)

        # Merge on date
        merged = pd.merge(real_df, pred_df, on=date_col, how="inner")

        # Plot
        plt.figure(figsize=(13, 7))
        real_line, = plt.plot(
            merged[date_col], merged["real"], 
            label="Real", color="black", linewidth=1.0
        )
        pred_line, = plt.plot(
            merged[date_col], merged[model], 
            label="Prediction", linestyle="--", linewidth=2, color="red"
        )

        # X-axis: only mark the beginning of each year
        years = merged[date_col].dt.year.unique()
        year_start_dates = [merged[merged[date_col].dt.year == y][date_col].iloc[0] for y in years]
        plt.xticks(
            year_start_dates, 
            [str(y) for y in years], 
            rotation=45, 
            fontsize=14, 
            fontweight='bold'
        )
        plt.yticks(fontsize=14, fontweight='bold')

        plt.title(f"{plot_title} - {model}", fontsize=18, fontweight='bold')
        plt.xlabel("Year", fontsize=15, fontweight='bold')
        plt.ylabel(y_label, fontsize=15, fontweight='bold')
        plt.legend(
            [real_line, pred_line],
            ["Real", "Prediction"],
            fontsize=16,
            frameon=True,
            facecolor='white',
            edgecolor='black',
            loc='upper left'
        )
        plt.tight_layout()

        # Save
        out_path = output_dir / f"{model}_vs_real.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot for model '{model}' to {out_path.resolve()}")


plot_real_vs_predictions(
    real_csv_path="data_non_std.csv",
    model_names=['tft_ex_nae', 'lstm_ex_nae', 'feedforward_ex_nae', 'transformer_ex_nae', 'tcn_ex_nae', 'cnn_ex_nae'],
    predictions_dir="expanding/trained_outputs_no_ae",
    output_dir="plots"
)