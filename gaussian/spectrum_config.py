import numpy as np


DEFAULT_CONDITIONS = {
    "start_reciprocal_cm": 1101,
    "end_reciprocal_cm": 3999,
    "start_reciprocal_cm_bkg": 2500,
    "end_reciprocal_cm_bkg": 3997,
    "spectrum_to_plot_as_example": 170,
    "liquid": "H2O"
}


def get_peak_parameters(liquid='H2O'):
    if liquid == 'D2O':
        initial_amplitudes = [1] * 24

        mean_values = [
            3400, 2950, 2658, 2600, 2555, 2470, 2390, 2367,
            2320, 2290, 2071, 2050, 2000, 1860, 1700, 1625,
            1560, 1508, 1460, 1430, 1365, 1310, 1204, 1140
        ]

        sigma_values = [
            100, 80, 35, 30, 70, 40, 50, 10,
            10, 75, 10, 20, 30, 80, 30, 23,
            30, 30, 20, 30, 10, 25, 23, 10
        ]

    elif liquid == 'H2O':
        initial_amplitudes = [1] * 32

        mean_values = [
            3680, 3520, 3360, 3210, 3100, 2870, 2800, 2367,
            2350, 2335, 2320, 2127, 2084, 2077, 2070, 2055,
            1836, 1800, 1700, 1639, 1610, 1541, 1508, 1430,
            1400, 1368, 1270, 1227, 1218, 1160, 1155, 1110
        ]

        sigma_values = [
            70, 100, 100, 80, 100, 80, 80, 10,
            10, 3, 10, 100, 3, 3, 5, 20,
            6, 100, 30, 40, 40, 40, 40, 30,
            20, 40, 40, 15, 15, 10, 40, 15
        ]

    else:
        raise ValueError(f"Unknown liquid type: {liquid}")

    return {
        'initial_amplitudes': initial_amplitudes,
        'mean_values': mean_values,
        'sigma_values': sigma_values
    }


def get_voltage_labels(experiment_classification):
    labels_map = {
        '_07': ["-0.05 V", "-0.4 V"],
        '_08': ["-0.4 V", "-0.8 V"],
        '_09': ["-0.8 V", "-1.1 V"]
    }
    return labels_map.get(experiment_classification, ["-0.4 V", "-0.8 V"])
