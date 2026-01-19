import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def prepare_data_for_lstm_with_cation(X, y, cation_encoding, time_steps=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        seq = X[i:i+time_steps].copy()
        X_seq.append(seq)
        y_seq.append(y[i+time_steps])

    X_seq_np = np.array(X_seq)
    y_seq_np = np.array(y_seq)

    cation_repeated = np.tile(cation_encoding, (len(X_seq_np), 1))

    return X_seq_np, y_seq_np, cation_repeated


def create_sequences_from_experiments(
    combined_peaks,
    valid_peak_names,
    all_experiment_ids,
    all_cations,
    all_batches,
    cation_encoded,
    batch_encoded,
    target_peak,
    time_steps=5
):
    feature_peaks = [peak for peak in valid_peak_names if peak != target_peak]

    X_indices = [valid_peak_names.index(peak) for peak in feature_peaks]
    y_index = valid_peak_names.index(target_peak)

    X = combined_peaks[:, X_indices]
    y = combined_peaks[:, y_index]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    all_X_seq = []
    all_y_seq = []
    all_cation_seq = []
    all_batch_seq = []
    all_cation_labels = []
    all_batch_labels = []

    for exp_id in sorted(set(all_experiment_ids)):
        exp_indices = [i for i, eid in enumerate(all_experiment_ids) if eid == exp_id]

        exp_X = X_scaled[exp_indices]
        exp_y = y_scaled[exp_indices]
        exp_cation = cation_encoded[exp_indices[0]]
        exp_batch = batch_encoded[exp_indices[0]]
        exp_cation_label = all_cations[exp_indices[0]]
        exp_batch_label = all_batches[exp_indices[0]]

        if len(exp_indices) > time_steps:
            X_seq, y_seq, _ = prepare_data_for_lstm_with_cation(exp_X, exp_y, exp_cation, time_steps)

            all_X_seq.append(X_seq)
            all_y_seq.append(y_seq)
            all_cation_seq.extend([exp_cation] * len(X_seq))
            all_batch_seq.extend([exp_batch] * len(X_seq))
            all_cation_labels.extend([exp_cation_label] * len(X_seq))
            all_batch_labels.extend([exp_batch_label] * len(X_seq))

    X_seq = np.vstack(all_X_seq)
    y_seq = np.concatenate(all_y_seq)
    cation_seq = np.array(all_cation_seq)
    batch_seq = np.array(all_batch_seq)

    return {
        'X_seq': X_seq,
        'y_seq': y_seq,
        'cation_seq': cation_seq,
        'batch_seq': batch_seq,
        'cation_labels': all_cation_labels,
        'batch_labels': all_batch_labels,
        'feature_peaks': feature_peaks,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }


def encode_categorical(cations, batches):
    cation_encoder = OneHotEncoder(sparse_output=False)
    batch_encoder = OneHotEncoder(sparse_output=False)

    cation_encoded = cation_encoder.fit_transform(np.array(cations).reshape(-1, 1))
    batch_encoded = batch_encoder.fit_transform(np.array(batches).reshape(-1, 1))

    return cation_encoded, batch_encoded, cation_encoder, batch_encoder
