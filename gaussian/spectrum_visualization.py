import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import magma
import mpld3
import os

from spectrum_config import get_voltage_labels


def setup_spectrum_plotting():
    mpl.use('SVG')
    mpl.rcParams['svg.fonttype'] = 'none'


def plot_fitted_spectrum(wavenumbers, spectrum_corrected, gaussians, fitted_curve,
                         mean_values, start_reciprocal_cm, end_reciprocal_cm,
                         output_dir, base_filename):
    num_peaks = len(mean_values)
    labels = mean_values
    colors = magma(np.linspace(0, 1, num_peaks))

    fig, ax = plt.subplots(figsize=(18, 8))

    for i, (gaussian_component, label, color) in enumerate(zip(gaussians, labels, colors)):
        plt.plot(wavenumbers, gaussian_component, label=label, color=color)
        plt.fill_between(wavenumbers, gaussian_component, color=color, alpha=0.6)

    plt.plot(wavenumbers, spectrum_corrected, label='Original Data')
    plt.plot(wavenumbers, fitted_curve, label='Fitted Curve')

    plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize=18)
    plt.ylabel('Intensity (a.u.)', fontsize=18)
    plt.xlim(start_reciprocal_cm, end_reciprocal_cm)
    plt.gca().invert_xaxis()
    plt.legend(fontsize=18, ncol=7)
    plt.title('Raw data', fontsize=18)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{base_filename}_withoutbackground_correction"

    png_path = os.path.join(output_dir, f"{filename}.png")
    svg_path = os.path.join(output_dir, f"{filename}.svg")

    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)

    html_path = os.path.join(output_dir, f"{filename}.html")
    html_content = mpld3.fig_to_html(plt.gcf())
    with open(html_path, "w") as html_file:
        html_file.write(html_content)

    plt.close()

    return filename


def plot_integrated_areas(experiment_time, integrated_areas, mean_values,
                          experiment_classification, output_dir, base_filename):
    num_peaks = len(mean_values)
    colors = magma(np.linspace(0, 1, num_peaks))

    fig, ax = plt.subplots(figsize=(18, 8))

    for i, peak_label in enumerate(integrated_areas.keys()):
        plt.plot(experiment_time, integrated_areas[peak_label],
                 label=mean_values[i], color=colors[i])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18, ncol=3)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Integrated Area (a.u.)', fontsize=18)
    plt.xlim(0, 1000)

    intersections = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)

    text_labels = get_voltage_labels(experiment_classification)

    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2]
        text_x = (x_start + x_end) / 2
        plt.text(text_x, plt.ylim()[1], text_label, rotation=45,
                 va='bottom', ha='center', fontsize=16)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{base_filename}_integration of deconvoluted peak"

    plt.subplots_adjust(top=0.9)

    png_path = os.path.join(output_dir, f"{filename}.png")
    eps_path = os.path.join(output_dir, f"{filename}.eps")
    svg_path = os.path.join(output_dir, f"{filename}.svg")

    plt.savefig(png_path)
    plt.savefig(eps_path)
    plt.savefig(svg_path, format='svg', transparent=True)

    html_path = os.path.join(output_dir, f"{filename}.html")
    html_content = mpld3.fig_to_html(plt.gcf())
    with open(html_path, "w") as html_file:
        html_file.write(html_content)

    plt.close()


def save_fitted_values(wavenumbers, popt, gaussians, mean_values,
                       output_dir, base_filename, combined_function):
    fitted_values = combined_function(wavenumbers, *popt)

    fit_df = {
        'Reciprocal_cm': wavenumbers,
        'Fitted_Values': fitted_values
    }

    for i, (gaussian_component, label) in enumerate(zip(gaussians, mean_values)):
        fit_df[f'Gaussian_{label}'] = gaussian_component

    import pandas as pd
    fit_df = pd.DataFrame(fit_df)

    filename = f"{base_filename}_withoutbackground_correction_fitted_values.csv"
    fit_csv_path = os.path.join(output_dir, filename)
    fit_df.to_csv(fit_csv_path, index=False)
