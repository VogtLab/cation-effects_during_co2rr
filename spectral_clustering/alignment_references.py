import numpy as np


def normalize_to_magnitude_one(curve):
    max_val = np.max(np.abs(curve))
    return curve / max_val if max_val != 0 else curve


def smooth_transition(x, midpoint, width):
    return 1 / (1 + np.exp(-(x - midpoint) / width))


def create_reference_curves(n_points=100, n_flat=9, step_size=2):
    reference_1 = np.zeros(n_points)

    exp_decay = np.exp(-np.arange(n_points - n_flat) / 10)
    reference_2 = np.concatenate((np.zeros(n_flat), exp_decay - exp_decay[0]))

    exp_increase = 1 - np.exp(-np.arange(n_points - n_flat) / 10)
    reference_3 = np.concatenate((np.zeros(n_flat), exp_increase - exp_increase[0]))

    reference_4 = np.concatenate((np.zeros(n_flat), np.full(n_points - n_flat, step_size)))
    reference_5 = np.concatenate((np.zeros(n_flat), np.full(n_points - n_flat, -step_size)))

    decreasing_linear = np.concatenate((np.zeros(n_flat), np.linspace(step_size, -step_size, n_points - n_flat)))
    increasing_linear = np.concatenate((np.zeros(n_flat), np.linspace(-step_size, step_size, n_points - n_flat)))

    sinusoidal_curve = np.concatenate((np.zeros(n_flat), np.sin(np.linspace(0, 2 * np.pi, n_points - n_flat))))
    inverse_sinusoidal_curve = np.concatenate((np.zeros(n_flat), -np.sin(np.linspace(0, 2 * np.pi, n_points - n_flat))))

    increase_duration = 10
    decay_rate = 0.1
    baseline_shift = 2.0
    flat_section = np.zeros(n_flat)

    x_increase = np.linspace(-6, 6, increase_duration)
    smooth_increase = 1 / (1 + np.exp(-x_increase))
    smooth_increase = (smooth_increase - smooth_increase[0]) / (smooth_increase[-1] - smooth_increase[0])

    x_combined = np.linspace(0, n_points - n_flat, n_points - n_flat)
    smooth_transition_curve = (
        smooth_increase[-1] * (1 - smooth_transition(x_combined, increase_duration, 2))
        + (np.exp(-decay_rate * x_combined) - baseline_shift) * smooth_transition(x_combined, increase_duration, 2)
    )
    smooth_transition_curve_final = np.concatenate((flat_section, smooth_increase, smooth_transition_curve))
    smooth_transition_curve_flipped = -smooth_transition_curve_final

    decreasing_linear_no_flat = np.linspace(step_size, -step_size, n_points)
    increasing_linear_no_flat = np.linspace(-step_size, step_size, n_points)

    references = {
        "Flat horizontal line": normalize_to_magnitude_one(reference_1),
        "Exponential decay": normalize_to_magnitude_one(reference_2),
        "Exponential increase": normalize_to_magnitude_one(reference_3),
        "Step up": normalize_to_magnitude_one(reference_4),
        "Step down": normalize_to_magnitude_one(reference_5),
        "Decreasing linear curve with flat start": normalize_to_magnitude_one(decreasing_linear),
        "Increasing linear curve with flat start": normalize_to_magnitude_one(increasing_linear),
        "Decreasing linear curve": normalize_to_magnitude_one(decreasing_linear_no_flat),
        "Increasing linear curve": normalize_to_magnitude_one(increasing_linear_no_flat),
        "Sinusoidal curve": normalize_to_magnitude_one(sinusoidal_curve),
        "Inverse sinusoidal curve": normalize_to_magnitude_one(inverse_sinusoidal_curve),
        "Smooth transition (width=2)": normalize_to_magnitude_one(smooth_transition_curve_final),
        "Inverse smooth transition (width=2)": normalize_to_magnitude_one(smooth_transition_curve_flipped)
    }

    return references
