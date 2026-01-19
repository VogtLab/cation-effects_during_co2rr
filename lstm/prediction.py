import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from model import CationLSTMModel, get_device


def setup_prediction_plotting():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)


def predict_peak(
    input_data_path,
    model_path,
    target_peak,
    cation,
    time_steps=5,
    output_path=None,
    cations_list=None,
    batches_list=None
):
    setup_prediction_plotting()
    device = get_device()
    print(f"Using device: {device}")

    if output_path:
        os.makedirs(output_path, exist_ok=True)

    print(f"Loading time-series data from {input_data_path}...")

    data = pd.read_csv(input_data_path)

    peak_columns = [col for col in data.columns if col.startswith('Mean')]

    if target_peak not in peak_columns:
        print(f"Warning: {target_peak} not found in data. Will only generate predictions without comparison.")
        has_target = False
    else:
        has_target = True

    feature_peaks = [peak for peak in peak_columns if peak != target_peak]
    print(f"Using {len(feature_peaks)} peaks as features")

    X = data[feature_peaks].values

    if has_target:
        y = data[target_peak].values

    if cations_list is None:
        cations_list = ['Li', 'Na', 'K', 'Cs']
    if batches_list is None:
        batches_list = ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5']

    cation_size = len(cations_list)
    batch_size = len(batches_list)

    cation_encoding = np.zeros(cation_size)
    if cation in cations_list:
        cation_idx = cations_list.index(cation)
        cation_encoding[cation_idx] = 1

    batch_encoding = np.zeros(batch_size)

    combined_encoding = np.concatenate([cation_encoding, batch_encoding])

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    if has_target:
        scaler_y = MinMaxScaler()
        _ = scaler_y.fit_transform(y.reshape(-1, 1))
    else:
        scaler_y = MinMaxScaler()
        dummy_data = np.array([0, 1]).reshape(-1, 1)
        scaler_y.fit(dummy_data)

    X_seq = []
    timestamps = []

    for i in range(len(X_scaled) - time_steps):
        seq = X_scaled[i:i+time_steps]
        X_seq.append(seq)
        timestamps.append(i + time_steps)

    X_seq = np.array(X_seq)

    print(f"Loading model from {model_path}...")

    input_size = X_seq.shape[2]
    combined_size = cation_size + batch_size
    model = CationLSTMModel(input_size, combined_size, dropout_rate=0.2).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    X_seq_tensor = torch.FloatTensor(X_seq).to(device)
    combined_tensor = torch.FloatTensor(combined_encoding).to(device)

    print("Making predictions...")
    with torch.no_grad():
        predictions = []

        batch_size_inference = 32
        for i in range(0, len(X_seq_tensor), batch_size_inference):
            batch_X = X_seq_tensor[i:i+batch_size_inference]

            batch_combined = combined_tensor.repeat(len(batch_X), 1)

            batch_pred = model(batch_X, batch_combined)
            predictions.append(batch_pred.cpu().numpy())

        y_pred = np.vstack(predictions).flatten()

    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    results = pd.DataFrame({
        'Timestamp': timestamps,
        f'Predicted_{target_peak.replace(" ", "_")}': y_pred_original
    })

    if has_target:
        actual_values = y[time_steps:]
        if len(actual_values) > len(results):
            actual_values = actual_values[:len(results)]
        results[f'Actual_{target_peak.replace(" ", "_")}'] = actual_values
        results['Error'] = results[f'Actual_{target_peak.replace(" ", "_")}'] - results[f'Predicted_{target_peak.replace(" ", "_")}']

    if output_path:
        results.to_csv(os.path.join(output_path, f'predicted_{target_peak.replace(" ", "_")}.csv'), index=False)

    if output_path:
        _plot_time_evolution(results, target_peak, cation, has_target, output_path)

        if has_target:
            _plot_correlation(results, target_peak, output_path)

    print("Prediction complete!")
    return results


def _plot_time_evolution(results, target_peak, cation, has_target, output_path):
    plt.figure(figsize=(15, 8), dpi=300)

    pred_col = f'Predicted_{target_peak.replace(" ", "_")}'
    actual_col = f'Actual_{target_peak.replace(" ", "_")}'

    plt.plot(results['Timestamp'], results[pred_col],
             label='Predicted', color='#1f77b4', linewidth=2)

    if has_target:
        plt.plot(results['Timestamp'], results[actual_col],
                 label='Actual', color='#d62728', linewidth=2, alpha=0.8)

        mse = mean_squared_error(results[actual_col], results[pred_col])
        r2 = r2_score(results[actual_col], results[pred_col])

        textstr = f'MSE = {mse:.4f}\nR² = {r2:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=14, bbox=props, verticalalignment='top')

    plt.title(f'Time Evolution of {target_peak} ({cation} Cation)', fontsize=18, pad=20)
    plt.xlabel('Time Step', fontsize=16)
    plt.ylabel('Peak Intensity', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()

    base_name = f'{target_peak.replace(" ", "_")}_time_evolution'
    plt.savefig(os.path.join(output_path, f'{base_name}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, f'{base_name}.svg'), bbox_inches='tight', format='svg')
    plt.close()


def _plot_correlation(results, target_peak, output_path):
    plt.figure(figsize=(10, 9), dpi=300)

    pred_col = f'Predicted_{target_peak.replace(" ", "_")}'
    actual_col = f'Actual_{target_peak.replace(" ", "_")}'

    scatter = plt.scatter(results[actual_col], results[pred_col],
                         c=results['Timestamp'], cmap='Blues',
                         s=80, alpha=0.7, edgecolor='white', linewidth=0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Step', fontsize=14)

    min_val = min(results[actual_col].min(), results[pred_col].min())
    max_val = max(results[actual_col].max(), results[pred_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)

    plt.title(f'Actual vs Predicted {target_peak} Values', fontsize=18, pad=20)
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    base_name = f'{target_peak.replace(" ", "_")}_correlation'
    plt.savefig(os.path.join(output_path, f'{base_name}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, f'{base_name}.svg'), bbox_inches='tight', format='svg')
    plt.close()
