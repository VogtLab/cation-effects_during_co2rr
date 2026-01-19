import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

from data_loading import load_experiment_data
from data_preparation import create_sequences_from_experiments, encode_categorical
from model import CationLSTMModel, get_device
from training import train_model, evaluate_model, calculate_feature_importance
from visualization import (
    setup_plotting,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_predictions,
    plot_cation_predictions,
    plot_batch_predictions,
    plot_training_history
)


def analyze_leave_one_out(experiment_data, output_dir, time_steps=5, epochs=100, batch_size=16, dropout_rate=0.2):
    setup_plotting()

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    data = load_experiment_data(experiment_data)

    all_peak_data = data['peak_data']
    all_cations = data['cations']
    all_batches = data['batches']
    all_experiment_ids = data['experiment_ids']
    valid_peak_names = data['valid_peak_names']

    print(f"Using {len(valid_peak_names)} peaks that are common across all batches:")
    print(valid_peak_names)

    print(f"\nBatch distribution:")
    batch_counts = pd.Series(all_batches).value_counts()
    print(batch_counts)

    print(f"\nCation distribution:")
    cation_counts = pd.Series(all_cations).value_counts()
    print(cation_counts)

    results = []

    cation_encoded, batch_encoded, cation_encoder, batch_encoder = encode_categorical(all_cations, all_batches)

    combined_peaks = np.vstack([
        d[:, [valid_peak_names.index(peak) if peak in valid_peak_names else -1
              for peak in valid_peak_names]]
        for d in all_peak_data
    ])

    combined_df = pd.DataFrame(combined_peaks, columns=valid_peak_names)
    combined_df['Cation'] = all_cations
    combined_df['Batch'] = all_batches

    plot_correlation_matrix(
        combined_df.drop(['Cation', 'Batch'], axis=1),
        "Correlation Matrix of All FTIR Peaks",
        output_dir
    )

    for batch in batch_encoder.categories_[0]:
        batch_mask = np.array(all_batches) == batch
        batch_df = combined_df[batch_mask]
        if len(batch_df) > 1:
            plot_correlation_matrix(
                batch_df.drop(['Cation', 'Batch'], axis=1),
                f"Correlation Matrix - {batch}",
                output_dir
            )

    cation_size = cation_encoded.shape[1]
    batch_size_enc = batch_encoded.shape[1]

    for target_idx, target_peak in enumerate(valid_peak_names):
        print(f"\n{'='*80}\nTraining model for target peak: {target_peak} ({target_idx+1}/{len(valid_peak_names)})\n{'='*80}")

        peak_dir = os.path.join(output_dir, f"peak_{target_peak.replace(' ', '_')}")
        os.makedirs(peak_dir, exist_ok=True)

        seq_data = create_sequences_from_experiments(
            combined_peaks,
            valid_peak_names,
            all_experiment_ids,
            all_cations,
            all_batches,
            cation_encoded,
            batch_encoded,
            target_peak,
            time_steps
        )

        X_seq = seq_data['X_seq']
        y_seq = seq_data['y_seq']
        cation_seq = seq_data['cation_seq']
        batch_seq = seq_data['batch_seq']
        all_cation_labels = seq_data['cation_labels']
        all_batch_labels = seq_data['batch_labels']
        feature_peaks = seq_data['feature_peaks']
        scaler_y = seq_data['scaler_y']

        print(f"Prepared {len(X_seq)} sequences with {X_seq.shape[2]} features each")
        print(f"Batch distribution in sequences: {pd.Series(all_batch_labels).value_counts().to_dict()}")

        combined_labels = [f"{cation}_{batch}" for cation, batch in zip(all_cation_labels, all_batch_labels)]

        try:
            X_train, X_test, y_train, y_test, cation_train, cation_test, batch_train, batch_test, \
            cation_labels_train, cation_labels_test, batch_labels_train, batch_labels_test = train_test_split(
                X_seq, y_seq, cation_seq, batch_seq, all_cation_labels, all_batch_labels,
                test_size=0.2, random_state=42, stratify=combined_labels
            )
        except ValueError:
            print("Warning: Stratification failed, using regular train-test split")
            X_train, X_test, y_train, y_test, cation_train, cation_test, batch_train, batch_test, \
            cation_labels_train, cation_labels_test, batch_labels_train, batch_labels_test = train_test_split(
                X_seq, y_seq, cation_seq, batch_seq, all_cation_labels, all_batch_labels,
                test_size=0.2, random_state=42
            )

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        cation_train_tensor = torch.FloatTensor(cation_train)
        batch_train_tensor = torch.FloatTensor(batch_train)

        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
        cation_test_tensor = torch.FloatTensor(cation_test)
        batch_test_tensor = torch.FloatTensor(batch_test)

        train_size = int(0.8 * len(X_train_tensor))

        combined_train_tensor = torch.cat([cation_train_tensor, batch_train_tensor], dim=1)
        combined_test_tensor = torch.cat([cation_test_tensor, batch_test_tensor], dim=1)

        train_dataset = TensorDataset(
            X_train_tensor[:train_size],
            combined_train_tensor[:train_size],
            y_train_tensor[:train_size]
        )
        val_dataset = TensorDataset(
            X_train_tensor[train_size:],
            combined_train_tensor[train_size:],
            y_train_tensor[train_size:]
        )
        test_dataset = TensorDataset(
            X_test_tensor,
            combined_test_tensor,
            y_test_tensor
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        input_size = X_train.shape[2]
        combined_encoding_size = cation_size + batch_size_enc
        model = CationLSTMModel(input_size, combined_encoding_size, dropout_rate=dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        print(f"Training LSTM model for {target_peak}...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device
        )

        plot_training_history(train_losses, val_losses, target_peak, peak_dir)

        y_pred, y_true = evaluate_model(model, test_loader, device)

        y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_original = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()

        r2, mse = plot_predictions(
            y_true_original,
            y_pred_original,
            f"Predictions for {target_peak}",
            peak_dir
        )

        cation_r2, cation_mse = plot_cation_predictions(
            y_true_original,
            y_pred_original,
            cation_labels_test,
            f"Predictions by Cation: {target_peak}",
            peak_dir
        )

        batch_r2, batch_mse = plot_batch_predictions(
            y_true_original,
            y_pred_original,
            batch_labels_test,
            f"Predictions by Batch: {target_peak}",
            peak_dir
        )

        print(f"Calculating feature importance for {target_peak}...")
        feature_importance = calculate_feature_importance(
            model, X_test, combined_test_tensor, y_true, feature_peaks, time_steps, device
        )

        plot_feature_importance(
            feature_importance,
            feature_peaks,
            f"Feature Importance for {target_peak}",
            peak_dir
        )

        top_n = min(5, len(feature_peaks))
        top_indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_peaks[i] for i in top_indices]

        torch.save(model.state_dict(), os.path.join(peak_dir, f'LSTM_model_{target_peak.replace(" ", "_")}.pt'))

        feature_importance_df = pd.DataFrame({
            'Feature': feature_peaks,
            'Importance': feature_importance
        })
        feature_importance_df.to_csv(os.path.join(peak_dir, 'feature_importance.csv'), index=False)

        results_df = pd.DataFrame({
            'Actual': y_true_original,
            'Predicted': y_pred_original,
            'Error': y_true_original - y_pred_original,
            'Cation': cation_labels_test,
            'Batch': batch_labels_test
        })
        results_df.to_csv(os.path.join(peak_dir, 'prediction_results.csv'), index=False)

        cation_r2_dict = {}
        for cation in set(cation_labels_test):
            mask = np.array(cation_labels_test) == cation
            if sum(mask) > 0:
                cation_r2_dict[cation] = r2_score(y_true_original[mask], y_pred_original[mask])

        batch_r2_dict = {}
        for batch in set(batch_labels_test):
            mask = np.array(batch_labels_test) == batch
            if sum(mask) > 0:
                batch_r2_dict[batch] = r2_score(y_true_original[mask], y_pred_original[mask])

        with open(os.path.join(peak_dir, 'summary_statistics.txt'), 'w') as f:
            f.write(f"Model for predicting '{target_peak}' using other peaks\n")
            f.write(f"Overall MSE: {mse:.4f}\n")
            f.write(f"Overall R²: {r2:.4f}\n")
            f.write("\nCation-Specific R² Values:\n")
            for cation, r2_val in cation_r2_dict.items():
                f.write(f"{cation}: {r2_val:.4f}\n")

            f.write("\nBatch-Specific R² Values:\n")
            for batch, r2_val in batch_r2_dict.items():
                f.write(f"{batch}: {r2_val:.4f}\n")

            f.write(f"\nTop {top_n} important features:\n")
            for i, feature in enumerate(top_features):
                idx = feature_peaks.index(feature)
                f.write(f"{i+1}. {feature}: {feature_importance[idx]:.6f}\n")

        results.append({
            'target_peak': target_peak,
            'r2': r2,
            'mse': mse,
            'cation_r2': cation_r2_dict,
            'batch_r2': batch_r2_dict,
            'top_features': top_features,
            'feature_importance': {feat: imp for feat, imp in zip(feature_peaks, feature_importance)}
        })

    summary_df = pd.DataFrame({
        'Peak': [r['target_peak'] for r in results],
        'R²': [r['r2'] for r in results],
        'MSE': [r['mse'] for r in results]
    })

    summary_df = summary_df.sort_values('R²', ascending=False)

    summary_df.to_csv(os.path.join(output_dir, 'peak_prediction_summary.csv'), index=False)

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return summary_df, results
