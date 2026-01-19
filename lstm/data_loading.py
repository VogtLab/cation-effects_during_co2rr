import pandas as pd


def load_integrated_data(file_path, cation_label, batch_boundaries=None, batch_labels=None):
    if batch_boundaries is None:
        batch_boundaries = [0, 181, 363, 545, 727]
    if batch_labels is None:
        batch_labels = ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5']

    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        print(f"Shape: {data.shape}")

        batch_boundaries = batch_boundaries + [data.shape[0]]

        batches = []
        for i in range(len(batch_boundaries) - 1):
            start_idx = batch_boundaries[i]
            end_idx = min(batch_boundaries[i + 1], data.shape[0])

            if start_idx < data.shape[0]:
                batch_data = data.iloc[start_idx:end_idx]
                if len(batch_data) > 0:
                    batch_label = batch_labels[i]
                    print(f"  {batch_label}: rows {start_idx}-{end_idx-1} ({len(batch_data)} rows)")
                    batches.append((batch_data, batch_label, cation_label))

        return batches

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []


def load_experiment_data(experiment_data, batch_boundaries=None, batch_labels=None):
    all_peak_data = []
    all_cations = []
    all_batches = []
    all_experiment_ids = []
    valid_peak_names = []

    experiment_counter = 0
    for exp in experiment_data:
        batches = load_integrated_data(
            exp['path'],
            exp['cation'],
            batch_boundaries,
            batch_labels
        )

        if not batches:
            continue

        for batch_data, batch_label, cation_label in batches:
            peak_columns = [col for col in batch_data.columns if col.startswith('Mean')]

            if experiment_counter == 0:
                valid_peak_names = peak_columns
            else:
                valid_peak_names = [peak for peak in valid_peak_names if peak in peak_columns]

            all_peak_data.append(batch_data[peak_columns].values)
            all_cations.extend([cation_label] * len(batch_data))
            all_batches.extend([batch_label] * len(batch_data))
            all_experiment_ids.extend([experiment_counter] * len(batch_data))

            experiment_counter += 1

    return {
        'peak_data': all_peak_data,
        'cations': all_cations,
        'batches': all_batches,
        'experiment_ids': all_experiment_ids,
        'valid_peak_names': valid_peak_names
    }
