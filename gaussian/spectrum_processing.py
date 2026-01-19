import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.stats import pearsonr
import os

from spectrum_config import get_peak_parameters


def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))


def create_combined_function(mean_values, sigma_values):
    def combined_function(x, *amps):
        return sum(gaussian(x, amp, mean, sigma)
                   for amp, mean, sigma in zip(amps, mean_values, sigma_values))
    return combined_function


def load_spectrum_data(file_path):
    file_name = os.path.basename(file_path)

    if file_name.startswith("DS") and file_name.endswith(".csv") and file_name[-5].isdigit():
        df = pd.read_csv(file_path, header=None, skiprows=1)
        df = df.iloc[:, :911]
        df = df[::-1]
        reciprocal_cm = df.iloc[:, 0]
        return df, reciprocal_cm

    elif file_name.startswith("Reconstructed"):
        df = pd.read_csv(file_path, header=None, skiprows=0)
        reciprocal_cm = df.iloc[:, 0]
        return df, reciprocal_cm

    else:
        return None, None


def apply_background_correction(spectrum, wavenumbers, start_bkg, end_bkg):
    index_start_bkg = np.where(wavenumbers >= start_bkg)[0][0]
    index_end_bkg = np.where(wavenumbers <= end_bkg)[0][-1]

    slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / \
            (wavenumbers[index_end_bkg] - wavenumbers[index_start_bkg])

    intercept = spectrum[index_start_bkg] - slope * wavenumbers[index_start_bkg]

    background_array = slope * wavenumbers + intercept

    spectrum_corrected = spectrum - background_array

    return spectrum_corrected


def fit_gaussians(wavenumbers, spectrum_corrected, peak_params):
    mean_values = peak_params['mean_values']
    sigma_values = peak_params['sigma_values']
    initial_amplitudes = peak_params['initial_amplitudes']

    combined_function = create_combined_function(mean_values, sigma_values)

    popt, _ = curve_fit(combined_function, wavenumbers, spectrum_corrected,
                        p0=initial_amplitudes, maxfev=10000)

    amps = popt

    gaussians = [gaussian(wavenumbers, amp, mean, sigma)
                 for amp, mean, sigma in zip(amps, mean_values, sigma_values)]

    fitted_curve = combined_function(wavenumbers, *popt)

    correlation_coefficient, _ = pearsonr(spectrum_corrected, fitted_curve)
    r_squared = correlation_coefficient ** 2

    return {
        'gaussians': gaussians,
        'fitted_curve': fitted_curve,
        'amplitudes': amps,
        'r_squared': r_squared,
        'popt': popt
    }


def calculate_integrated_areas(df, reciprocal_cm, peak_params,
                                start_reciprocal_cm, end_reciprocal_cm,
                                start_reciprocal_cm_bkg, end_reciprocal_cm_bkg):
    mean_values = peak_params['mean_values']
    sigma_values = peak_params['sigma_values']
    initial_amplitudes = peak_params['initial_amplitudes']
    num_peaks = len(mean_values)

    combined_function = create_combined_function(mean_values, sigma_values)

    experiment_numbers = df.columns[1:]
    experiment_time = np.arange(len(experiment_numbers)) * 1.1

    integrated_areas = {f'Peak {i}': [] for i in range(num_peaks)}

    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]

    for experiment_number in experiment_numbers:
        trimmed_wavenumbers = df.iloc[start_index:end_index+1, 0].values
        spectrum = df.iloc[start_index:end_index+1, experiment_number].values

        spectrum_corrected = apply_background_correction(
            spectrum, trimmed_wavenumbers,
            start_reciprocal_cm_bkg, end_reciprocal_cm_bkg
        )

        popt, _ = curve_fit(combined_function, trimmed_wavenumbers, spectrum_corrected,
                            p0=initial_amplitudes, maxfev=10000)

        amps = popt

        gaussians = [gaussian(trimmed_wavenumbers, amp, mean, sigma)
                     for amp, mean, sigma in zip(amps, mean_values, sigma_values)]

        for i in range(num_peaks):
            area = trapezoid(gaussians[i], trimmed_wavenumbers)
            integrated_areas[f'Peak {i}'].append(area)

    integrated_areas_df = pd.DataFrame(integrated_areas)
    integrated_areas_df['Time (s)'] = experiment_time

    header_mapping = {f'Peak {i}': f'Mean {mean_values[i]}' for i in range(num_peaks)}
    integrated_areas_df.rename(columns=header_mapping, inplace=True)

    cols = ['Time (s)'] + [col for col in integrated_areas_df.columns if col != 'Time (s)']
    integrated_areas_df = integrated_areas_df[cols]

    return integrated_areas_df, experiment_time, integrated_areas
