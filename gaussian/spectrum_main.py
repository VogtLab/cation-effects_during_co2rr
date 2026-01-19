import os
import pandas as pd

from spectrum_config import DEFAULT_CONDITIONS, get_peak_parameters
from spectrum_processing import (
    load_spectrum_data,
    apply_background_correction,
    fit_gaussians,
    calculate_integrated_areas,
    create_combined_function
)
from spectrum_visualization import (
    setup_spectrum_plotting,
    plot_fitted_spectrum,
    plot_integrated_areas,
    save_fitted_values
)
from spectrum_extraction import (
    extract_integrated_areas_at_time,
    save_consolidated_results
)


r2_results = []


def process_spectrum_file(file_path, start_reciprocal_cm=1101,
                          end_reciprocal_cm=3999,
                          start_reciprocal_cm_bkg=2500,
                          end_reciprocal_cm_bkg=3997,
                          spectrum_to_plot_as_example=600,
                          experiment_classification='_08',
                          liquid='H2O'):
    setup_spectrum_plotting()

    df, reciprocal_cm = load_spectrum_data(file_path)

    if df is None:
        print(f"Skipping file (doesn't match expected pattern): {os.path.basename(file_path)}")
        return

    peak_params = get_peak_parameters(liquid)
    mean_values = peak_params['mean_values']

    folder = os.path.dirname(file_path)
    folder_name = f"{start_reciprocal_cm}_to_{end_reciprocal_cm}"
    folder_path = os.path.join(folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    base_filename = os.path.basename(folder)

    start_index = (reciprocal_cm >= start_reciprocal_cm).idxmax()
    end_index = (reciprocal_cm <= end_reciprocal_cm)[::-1].idxmax()

    trimmed_wavenumbers = df.iloc[start_index:end_index+1, 0].values
    spectrum = df.iloc[start_index:end_index+1, spectrum_to_plot_as_example].values

    spectrum_corrected = apply_background_correction(
        spectrum, trimmed_wavenumbers,
        start_reciprocal_cm_bkg, end_reciprocal_cm_bkg
    )

    fit_result = fit_gaussians(trimmed_wavenumbers, spectrum_corrected, peak_params)

    print(f"R-squared: {fit_result['r_squared']}")

    r2_results.append({
        'Experiment': base_filename,
        'Folder': folder,
        'File': os.path.basename(file_path),
        'R_squared': fit_result['r_squared']
    })

    plot_fitted_spectrum(
        trimmed_wavenumbers, spectrum_corrected,
        fit_result['gaussians'], fit_result['fitted_curve'],
        mean_values, start_reciprocal_cm, end_reciprocal_cm,
        folder_path, base_filename
    )

    combined_function = create_combined_function(
        peak_params['mean_values'],
        peak_params['sigma_values']
    )
    save_fitted_values(
        trimmed_wavenumbers, fit_result['popt'],
        fit_result['gaussians'], mean_values,
        folder_path, base_filename, combined_function
    )

    integrated_areas_df, experiment_time, integrated_areas = calculate_integrated_areas(
        df, reciprocal_cm, peak_params,
        start_reciprocal_cm, end_reciprocal_cm,
        start_reciprocal_cm_bkg, end_reciprocal_cm_bkg
    )

    csv_filename = os.path.join(folder_path, f"{base_filename}_withoutbackground_correction_integrated_areas.csv")
    integrated_areas_df.to_csv(csv_filename, index=False)

    plot_integrated_areas(
        experiment_time, integrated_areas, mean_values,
        experiment_classification, folder_path, base_filename
    )


base_dir = "./data"

include_folders = set()

conditions = DEFAULT_CONDITIONS


if __name__ == "__main__":
    for folder_name in include_folders:
        reconstruction_path = os.path.join(
            base_dir, folder_name,
            "Reconstruction_based_on_CO_peak_in_eigenspectra"
        )

        if not os.path.exists(reconstruction_path):
            print(f"Folder not found: {reconstruction_path}")
            continue

        csv_files = []
        for root, _, files in os.walk(reconstruction_path):
            csv_files.extend([
                os.path.join(root, file)
                for file in files
                if file.endswith(".csv") and
                   (file.startswith("DS") or file.startswith("ReconstructedData"))
            ])

        if not csv_files:
            print(f"No matching CSV files found in {reconstruction_path}")
            continue

        for csv_file in csv_files:
            print(f"Processing file: {csv_file}")

            experiment_classification = '_unknown'
            for suffix in ["_07", "_08", "_09"]:
                if f"{folder_name}{suffix}.csv" in csv_file:
                    experiment_classification = suffix
                    break

            process_spectrum_file(
                file_path=csv_file,
                experiment_classification=experiment_classification,
                **conditions
            )

    if r2_results:
        r2_df = pd.DataFrame(r2_results)
        r2_csv_path = os.path.join(base_dir, "r2_values.csv")
        r2_df.to_csv(r2_csv_path, index=False)
        print(f"R^2 values saved to: {r2_csv_path}")
    else:
        print("No R^2 values to save.")

    target_peaks = [3680, 3520, 3360, 3210, 3100, 2870]
    target_time = 195.8

    results = extract_integrated_areas_at_time(
        base_dir=base_dir,
        include_folders=include_folders,
        target_peaks=target_peaks,
        target_time=target_time
    )

    consolidated_csv_path = os.path.join(base_dir, "consolidated_integrated_areas.csv")
    save_consolidated_results(results, consolidated_csv_path)
