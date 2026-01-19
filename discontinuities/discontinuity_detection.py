import numpy as np
import os
from scipy.signal import find_peaks, savgol_filter


def find_integrated_areas_file(file_path, subfolder, file_ending):
    base_folder = os.path.dirname(file_path)
    folder_path = os.path.join(base_folder, subfolder)

    if not os.path.exists(folder_path):
        print(f"WARNING: Subfolder not found: {folder_path}")
        return None

    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_ending):
            return os.path.join(folder_path, file_name)
    return None


def calculate_discontinuity_magnitudes(time, pc1_smoothed, discontinuities):
    magnitudes = []
    for i in discontinuities:
        if i > 0 and i < len(pc1_smoothed) - 1:
            magnitude_before = abs(pc1_smoothed[i] - pc1_smoothed[i-1])
            magnitude_after = abs(pc1_smoothed[i] - pc1_smoothed[i+1])
            magnitude = max(magnitude_before, magnitude_after)
            magnitudes.append(magnitude)
    return np.array(magnitudes)


def analyze_discontinuities(time, pc1_smoothed, peak_threshold=15, trough_threshold=15,
                            potential_change_times=None, magnitude_threshold=2):
    time_adjusted = time

    peaks, _ = find_peaks(pc1_smoothed, prominence=peak_threshold)
    troughs, _ = find_peaks(-pc1_smoothed, prominence=trough_threshold)

    discontinuities = np.sort(np.concatenate((peaks, troughs)))

    custom_discontinuities = []
    for i in range(1, len(pc1_smoothed) - 1):
        diff_before = abs(pc1_smoothed[i] - pc1_smoothed[i-1])
        diff_after = abs(pc1_smoothed[i+1] - pc1_smoothed[i])
        if diff_after >= 3 * diff_before:
            custom_discontinuities.append(i)

    discontinuities = np.unique(np.concatenate((discontinuities, custom_discontinuities)))

    if potential_change_times is not None:
        potential_change_indices = [np.argmin(np.abs(time_adjusted - pct)) for pct in potential_change_times]
        mask = np.ones(len(discontinuities), dtype=bool)
        for i in potential_change_indices:
            mask &= (np.abs(discontinuities - i) > 10)
        discontinuities = discontinuities[mask]

    magnitudes = calculate_discontinuity_magnitudes(time, pc1_smoothed, discontinuities)

    valid_indices = magnitudes >= magnitude_threshold
    discontinuities = discontinuities[valid_indices]
    magnitudes = magnitudes[valid_indices]

    if len(discontinuities) == 0:
        return np.array([]), np.array([]), np.array([]), potential_change_times

    return time_adjusted[discontinuities], magnitudes, pc1_smoothed[discontinuities], potential_change_times


def get_surrounding_points(time, pc1_smoothed, discontinuities, window=5):
    surrounding_times = []
    surrounding_values = []
    surrounding_indices = []

    discontinuity_indices = np.searchsorted(time, discontinuities)

    for i in discontinuity_indices:
        start = max(i - window, 0)
        end = min(i + window + 1, len(pc1_smoothed))

        surrounding_times.append(time[start:end])
        surrounding_values.append(pc1_smoothed[start:end])
        surrounding_indices.append(np.arange(start, end))

    return surrounding_times, surrounding_values, surrounding_indices


def smooth_signal(pc1, window_length, polyorder):
    return savgol_filter(pc1, window_length=window_length, polyorder=polyorder)
