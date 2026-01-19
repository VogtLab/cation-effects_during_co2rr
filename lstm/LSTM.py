'''
change where to save in output_dir under def analyze_oneout():


change under experiment_data = what to use as input



'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import os
from datetime import datetime
import matplotlib as mpl

mpl.use('SVG')

# Set font settings for better compatibility with SVG text
mpl.rcParams['svg.fonttype'] = 'none' 

# Set plotting style for prettier figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define experiment paths and cation identities
experiment_data = [
    
    # Raw data CO2
    
# =============================================================================
#     
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_07'},
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_08'},
# 
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_09'},
# 
# =============================================================================


# Raw data Ar

# =============================================================================
#     
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_07_Ar'},
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_08_Ar'},
# 
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00134/Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_09_Ar'},
# 
# =============================================================================


# 1-15 CO2
    
# =============================================================================
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_07'},
# 
# =============================================================================
#    {'path': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_08'},
# =============================================================================
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_08'},
# 
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/R1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_09'},
# 
# =============================================================================




# =============================================================================
# # 1-15 Ar
#     
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_07_Ar'},
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_08_Ar'},
# 
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/R1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_raw_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_raw_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00134/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_raw_09_Ar'},
# 
# =============================================================================

# =============================================================================
# # 1-15 NaCl Ar
#     
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00035/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00036/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00037/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_09_Ar'},
# 
# =============================================================================


# =============================================================================
# # 1-15 NaCl CO2
#     
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00058/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00059/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00060/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_09'},
# 
# =============================================================================

# =============================================================================
# # 1-15 NaHCO3 Au CO2
#     
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00068/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_07'},
     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00178/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/CZ_00070/Reconstruction_based_on_CO_peak_in_eigenspectra/1-15/1101_to_3999/1-15_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_raw_09'},
# 
# =============================================================================




# Interfacial Layer CO2

# =============================================================================
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_07'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_07'},
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_08'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_08'},
# 
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_09'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_09'},
# 
# 
# =============================================================================


# Interfacial Layer Ar

# =============================================================================
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_07_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_07_Ar'},
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_08_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_08_Ar'},
# 
# 
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'K_09_Ar'},
#     {'path': '/Users/danielsinausia/Documents/Experiments/DS_00134/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs_09_Ar'},
# 
# 
# =============================================================================


]

# Function to load and process the integrated area data
def load_integrated_data(file_path, cation_label):
    """
    Load data and fragment it into batches based on column ranges
    Returns list of tuples: (batch_data, batch_label, cation_label)
    """
    try:
        # Load with header
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        print(f"Shape: {data.shape}")
        
        # Define batch boundaries (1-indexed, so subtract 1 for 0-indexed)
        batch_boundaries = [0, 181, 363, 545, 727, data.shape[0]]
        batch_labels = ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5']
        
        batches = []
        for i in range(len(batch_boundaries) - 1):
            start_idx = batch_boundaries[i]
            end_idx = min(batch_boundaries[i + 1], data.shape[0])
            
            if start_idx < data.shape[0]:  # Only if we have data for this batch
                batch_data = data.iloc[start_idx:end_idx]
                if len(batch_data) > 0:  # Only add non-empty batches
                    batch_label = batch_labels[i]
                    print(f"  {batch_label}: rows {start_idx}-{end_idx-1} ({len(batch_data)} rows)")
                    batches.append((batch_data, batch_label, cation_label))
        
        return batches
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

# Function to prepare data for LSTM with cation identity
def prepare_data_for_lstm_with_cation(X, y, cation_encoding, time_steps=5):
    """
    Prepare data for LSTM by creating sequences of time_steps length
    X: input features (all peaks except the target peak)
    y: target values (the peak being predicted)
    cation_encoding: one-hot encoded cation identity
    time_steps: number of time steps to include in each sequence
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        # Create a sequence with all peaks
        seq = X[i:i+time_steps].copy()
        
        # Add sequence to collection
        X_seq.append(seq)
        y_seq.append(y[i+time_steps])
    
    # Convert to numpy arrays
    X_seq_np = np.array(X_seq)
    y_seq_np = np.array(y_seq)
    
    # Create array of cation encodings repeated for each sequence
    cation_repeated = np.tile(cation_encoding, (len(X_seq_np), 1))
    
    return X_seq_np, y_seq_np, cation_repeated

# Define LSTM model with cation identity and dropout for regularization
class CationLSTMModel(nn.Module):
    def __init__(self, input_size, cation_size, hidden_size1=64, hidden_size2=32, output_size=1, dropout_rate=0.3):
        super(CationLSTMModel, self).__init__()
        # Add dropout for regularization
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Combine LSTM output with cation encoding
        self.fc1 = nn.Linear(hidden_size2 + cation_size, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate/2)  # Less dropout on final layers
        self.fc2 = nn.Linear(16, output_size)
        
    def forward(self, x, cation):
        # Process sequence with LSTM
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Get the output from the last time step
        lstm_out = lstm2_out[:, -1, :]
        
        # Concatenate LSTM output with cation encoding
        combined = torch.cat((lstm_out, cation), dim=1)
        
        # Process through fully connected layers
        x = self.relu(self.fc1(combined))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

# Function to plot correlation matrix
def plot_correlation_matrix(df, title, output_dir=None):
    plt.figure(figsize=(14, 12), dpi=300)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.color_palette("Blues", as_cmap=True)
    ax = sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt='.2f', 
        cmap=cmap, 
        vmin=-1, 
        vmax=1,
        annot_kws={"size": 10},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    plt.title(title, fontsize=18, pad=20)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()  # Close the figure to free memory

# Function to plot feature importance
def plot_feature_importance(importance, feature_names, title, output_dir=None):
    plt.figure(figsize=(14, 10), dpi=300)
    
    # Sort indices and data
    sorted_idx = np.argsort(importance)
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(sorted_idx)))
    bars = plt.barh(range(len(sorted_idx)), sorted_importance, color=colors)
    plt.yticks(range(len(sorted_idx)), sorted_names, fontsize=14)
    plt.xlabel('Feature Importance', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    
    # Add values next to the bars
    for i, v in enumerate(sorted_importance):
        plt.text(v + 0.01*max(importance), i, f"{v:.2e}", va='center', fontsize=12)
    
    # Add grid for horizontal lines only
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()  # Close the figure to free memory

# Function to plot predictions vs actual
def plot_predictions(y_true, y_pred, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    # Create a scatter plot with Blues-colored points
    scatter = plt.scatter(y_true, y_pred, c=y_true, cmap='Blues', 
                         s=80, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Actual Values', fontsize=14)
    
    # Plot the perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    
    # Add labels and title with better formatting
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    
    # Add metrics as text box
    textstr = f'$R^2 = {r2:.4f}$\n$MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=14, bbox=props, verticalalignment='top')    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()  # Close the figure to free memory
    return r2, mse

# Function to plot cation-specific predictions
def plot_cation_predictions(y_true, y_pred, cations, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Define colors for each cation
    cation_colors = {'Na': '#1f77b4', 'K': '#2ca02c', 'Cs': '#d62728', 'Li': '#9467bd'}
    
    # Plot points with colors by cation
    for cation in set(cations):
        mask = np.array(cations) == cation
        if sum(mask) > 0:  # Only if we have data for this cation
            plt.scatter(
                y_true[mask], 
                y_pred[mask], 
                color=cation_colors.get(cation, 'gray'),
                label=f"{cation} (R²={r2_score(y_true[mask], y_pred[mask]):.4f})",
                alpha=0.7,
                s=80,
                edgecolor='white',
                linewidth=0.5
            )
    
    # Add diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    # Add overall R² value
    overall_r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    textstr = f'Overall $R^2 = {overall_r2:.4f}$\nOverall $MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=14, bbox=props, verticalalignment='top')
    
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()  # Close the figure to free memory
    return overall_r2, mse

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, device=device, verbose=True):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, cations, targets in train_loader:
            inputs, cations, targets = inputs.to(device), cations.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, cations)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, cations, targets in val_loader:
                inputs, cations, targets = inputs.to(device), cations.to(device), targets.to(device)
                outputs = model(inputs, cations)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Main function to analyze with leave-one-out approach
def analyze_leave_one_out():
    # Create a timestamp for the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = '/Users/danielsinausia/Documents/Paper Cations/LSTM/1-15_NaCl_08_splitinbatchessothatyouonlyaccountforthetransitionfromlessnegativetomorenegativeoverpotentials'
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to hold data from all experiments
    all_peak_data = []
    all_cations = []
    all_batches = []  # New: track batch information
    all_experiment_ids = []
    valid_peak_names = []  # Will store the intersection of available peaks across all experiments
    
    # First, load data and determine which peaks are available in all experiments
    experiment_counter = 0
    for exp in experiment_data:
        batches = load_integrated_data(exp['path'], exp['cation'])
        
        if not batches:
            continue
            
        for batch_data, batch_label, cation_label in batches:
            # Find all peak columns (assumed to start with 'Mean')
            peak_columns = [col for col in batch_data.columns if col.startswith('Mean')]
            
            # For the first batch, initialize valid_peak_names
            if experiment_counter == 0:
                valid_peak_names = peak_columns
            else:
                # For subsequent batches, keep only peaks that exist in this dataset
                valid_peak_names = [peak for peak in valid_peak_names if peak in peak_columns]
            
            # Store data
            all_peak_data.append(batch_data[peak_columns].values)
            all_cations.extend([cation_label] * len(batch_data))
            all_batches.extend([batch_label] * len(batch_data))  # New: store batch labels
            all_experiment_ids.extend([experiment_counter] * len(batch_data))
            
            experiment_counter += 1
    
    # Print the list of peaks that will be used
    print(f"Using {len(valid_peak_names)} peaks that are common across all batches:")
    print(valid_peak_names)
    
    # Print batch and cation distribution
    print(f"\nBatch distribution:")
    batch_counts = pd.Series(all_batches).value_counts()
    print(batch_counts)
    
    print(f"\nCation distribution:")
    cation_counts = pd.Series(all_cations).value_counts()
    print(cation_counts)
    
    # Create a list to store results for each peak
    results = []
    
    # One-hot encode the cation information
    cation_encoder = OneHotEncoder(sparse_output=False)
    cation_encoded = cation_encoder.fit_transform(np.array(all_cations).reshape(-1, 1))
    
    # One-hot encode the batch information
    batch_encoder = OneHotEncoder(sparse_output=False)
    batch_encoded = batch_encoder.fit_transform(np.array(all_batches).reshape(-1, 1))
    
    # Combined data for all peaks
    combined_peaks = np.vstack([data[:, [valid_peak_names.index(peak) if peak in valid_peak_names else -1 
                                        for peak in valid_peak_names]] 
                               for data in all_peak_data])
    
    # Create a DataFrame for correlation analysis
    combined_df = pd.DataFrame(combined_peaks, columns=valid_peak_names)
    combined_df['Cation'] = all_cations
    combined_df['Batch'] = all_batches  # New: add batch information
    
    # Plot overall correlation matrix
    plot_correlation_matrix(
        combined_df.drop(['Cation', 'Batch'], axis=1), 
        "Correlation Matrix of All FTIR Peaks", 
        output_dir
    )
    
    # Plot correlation matrices by batch
    for batch in batch_encoder.categories_[0]:
        batch_mask = np.array(all_batches) == batch
        batch_df = combined_df[batch_mask]
        if len(batch_df) > 1:  # Only plot if we have enough data
            plot_correlation_matrix(
                batch_df.drop(['Cation', 'Batch'], axis=1), 
                f"Correlation Matrix - {batch}", 
                output_dir
            )
    
    # Define common parameters
    time_steps = 5
    epochs = 100
    cation_size = cation_encoded.shape[1]
    batch_size = batch_encoded.shape[1]
    
    # For each peak, train a model to predict it from all other peaks
    for target_idx, target_peak in enumerate(valid_peak_names):
        print(f"\n{'='*80}\nTraining model for target peak: {target_peak} ({target_idx+1}/{len(valid_peak_names)})\n{'='*80}")
        
        # Create a subdirectory for this peak
        peak_dir = os.path.join(output_dir, f"peak_{target_peak.replace(' ', '_')}")
        os.makedirs(peak_dir, exist_ok=True)
        
        # Separate target peak and feature peaks
        feature_peaks = [peak for peak in valid_peak_names if peak != target_peak]
        
        # Create X (features) and y (target) datasets
        X_indices = [valid_peak_names.index(peak) for peak in feature_peaks]
        y_index = valid_peak_names.index(target_peak)
        
        X = combined_peaks[:, X_indices]
        y = combined_peaks[:, y_index]
        
        # Scale the data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Prepare sequences for each experiment
        all_X_seq = []
        all_y_seq = []
        all_cation_seq = []
        all_batch_seq = []  # New: store batch sequences
        all_cation_labels = []
        all_batch_labels = []  # New: store batch labels
        
        # Create sequences for each experiment separately to avoid crossing experiment boundaries
        for exp_id in sorted(set(all_experiment_ids)):
            # Get indices for this experiment
            exp_indices = [i for i, eid in enumerate(all_experiment_ids) if eid == exp_id]
            
            # Get data for this experiment
            exp_X = X_scaled[exp_indices]
            exp_y = y_scaled[exp_indices]
            exp_cation = cation_encoded[exp_indices[0]]  # All same cation in experiment
            exp_batch = batch_encoded[exp_indices[0]]    # All same batch in experiment
            exp_cation_label = all_cations[exp_indices[0]]
            exp_batch_label = all_batches[exp_indices[0]]
            
            # Create sequences
            if len(exp_indices) > time_steps:  # Only if we have enough data points
                X_seq, y_seq, _ = prepare_data_for_lstm_with_cation(exp_X, exp_y, exp_cation, time_steps)
                
                # Store
                all_X_seq.append(X_seq)
                all_y_seq.append(y_seq)
                all_cation_seq.extend([exp_cation] * len(X_seq))
                all_batch_seq.extend([exp_batch] * len(X_seq))  # New: store batch sequences
                all_cation_labels.extend([exp_cation_label] * len(X_seq))
                all_batch_labels.extend([exp_batch_label] * len(X_seq))  # New: store batch labels
        
        # Combine sequences from all experiments
        X_seq = np.vstack(all_X_seq)
        y_seq = np.concatenate(all_y_seq)
        cation_seq = np.array(all_cation_seq)
        batch_seq = np.array(all_batch_seq)  # New: batch sequences
        
        print(f"Prepared {len(X_seq)} sequences with {X_seq.shape[2]} features each")
        print(f"Batch distribution in sequences: {pd.Series(all_batch_labels).value_counts().to_dict()}")
        
        # Create combined labels for stratification (cation + batch)
        combined_labels = [f"{cation}_{batch}" for cation, batch in zip(all_cation_labels, all_batch_labels)]
        
        # Split into training and testing sets (stratified by combined cation+batch)
        try:
            X_train, X_test, y_train, y_test, cation_train, cation_test, batch_train, batch_test, \
            cation_labels_train, cation_labels_test, batch_labels_train, batch_labels_test = train_test_split(
                X_seq, y_seq, cation_seq, batch_seq, all_cation_labels, all_batch_labels, 
                test_size=0.2, random_state=42, stratify=combined_labels
            )
        except ValueError:
            # If stratification fails due to insufficient samples, use regular split
            print("Warning: Stratification failed, using regular train-test split")
            X_train, X_test, y_train, y_test, cation_train, cation_test, batch_train, batch_test, \
            cation_labels_train, cation_labels_test, batch_labels_train, batch_labels_test = train_test_split(
                X_seq, y_seq, cation_seq, batch_seq, all_cation_labels, all_batch_labels, 
                test_size=0.2, random_state=42
            )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        cation_train_tensor = torch.FloatTensor(cation_train)
        batch_train_tensor = torch.FloatTensor(batch_train)  # New: batch tensors
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
        cation_test_tensor = torch.FloatTensor(cation_test)
        batch_test_tensor = torch.FloatTensor(batch_test)  # New: batch tensors
        
        # Create DataLoader for training and validation
        train_size = int(0.8 * len(X_train_tensor))
        val_size = len(X_train_tensor) - train_size
        
        # Combine cation and batch encodings
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
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Initialize model with combined cation+batch size
        input_size = X_train.shape[2]  # Number of features
        combined_encoding_size = cation_size + batch_size  # Combined size
        model = CationLSTMModel(input_size, combined_encoding_size, dropout_rate=0.2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Train the model
        print(f"Training LSTM model for {target_peak}...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device
        )
        
        # Plot training history (same as before)
        plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(train_losses, color='#9467bd', linewidth=2.5, label='Training Loss')
        plt.plot(val_losses, color='#e377c2', linewidth=2.5, label='Validation Loss')
        
        plt.fill_between(range(len(train_losses)), train_losses, alpha=0.2, color='#9467bd')
        plt.fill_between(range(len(val_losses)), val_losses, alpha=0.2, color='#e377c2')
        
        plt.title(f'Training History: {target_peak}', fontsize=18, pad=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(peak_dir, 'Training_History.png'), bbox_inches='tight')
        plt.savefig(os.path.join(peak_dir, 'Training_History.svg'), bbox_inches='tight', format='svg')
        plt.close()
        
        # Evaluate the model (same as before)
        model.eval()
        with torch.no_grad():
            all_y_pred = []
            all_y_true = []
            
            for inputs, combined_encodings, targets in test_loader:
                inputs, combined_encodings, targets = inputs.to(device), combined_encodings.to(device), targets.to(device)
                outputs = model(inputs, combined_encodings)
                
                all_y_pred.append(outputs.cpu().numpy())
                all_y_true.append(targets.cpu().numpy())
            
            y_pred = np.vstack(all_y_pred).flatten()
            y_true = np.vstack(all_y_true).flatten()
        
        # Convert predictions back to original scale
        y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_original = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        
        # Plot and evaluate overall predictions
        r2, mse = plot_predictions(
            y_true_original, 
            y_pred_original, 
            f"Predictions for {target_peak}", 
            peak_dir
        )
        
        # Plot and evaluate cation-specific predictions
        cation_r2, cation_mse = plot_cation_predictions(
            y_true_original, 
            y_pred_original, 
            cation_labels_test,
            f"Predictions by Cation: {target_peak}",
            peak_dir
        )
        
        # NEW: Plot and evaluate batch-specific predictions
        batch_r2, batch_mse = plot_batch_predictions(
            y_true_original, 
            y_pred_original, 
            batch_labels_test,
            f"Predictions by Batch: {target_peak}",
            peak_dir
        )
        
        # Calculate feature importance (same as before, but with combined encodings)
        print(f"Calculating feature importance for {target_peak}...")
        
        feature_importance = np.zeros(len(feature_peaks))
        baseline_mse = mean_squared_error(y_true, y_pred)
        
        # Calculate feature importance by permuting each feature
        for feature_idx, feature_name in enumerate(feature_peaks):
            X_permuted = X_test.copy()
            
            # Permute the feature across all time steps
            for t in range(time_steps):
                # Save original values
                orig_values = X_permuted[:, t, feature_idx].copy()
                
                # Permute the values
                np.random.shuffle(X_permuted[:, t, feature_idx])
                
                # Convert to tensor and predict
                X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
                
                with torch.no_grad():
                    y_permuted_pred = model(X_permuted_tensor, combined_test_tensor.to(device))
                    y_permuted_pred = y_permuted_pred.cpu().numpy().flatten()
                
                # Calculate new MSE with permuted feature
                permuted_mse = mean_squared_error(y_true, y_permuted_pred)
                
                # Feature importance is the increase in error
                feature_importance[feature_idx] += (permuted_mse - baseline_mse)
                
                # Restore original values
                X_permuted[:, t, feature_idx] = orig_values
        
        # Normalize feature importance
        feature_importance /= time_steps
        
        # Plot feature importance
        plot_feature_importance(
            feature_importance, 
            feature_peaks, 
            f"Feature Importance for {target_peak}",
            peak_dir
        )
        
        # Identify top features
        top_n = min(5, len(feature_peaks))
        top_indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_peaks[i] for i in top_indices]
        
        # Save model
        torch.save(model.state_dict(), os.path.join(peak_dir, f'LSTM_model_{target_peak.replace(" ", "_")}.pt'))
        
        # Save feature importance data
        feature_importance_df = pd.DataFrame({
            'Feature': feature_peaks,
            'Importance': feature_importance
        })
        feature_importance_df.to_csv(os.path.join(peak_dir, 'feature_importance.csv'), index=False)
        
        # Save predictions and actual values with batch information
        results_df = pd.DataFrame({
            'Actual': y_true_original,
            'Predicted': y_pred_original,
            'Error': y_true_original - y_pred_original,
            'Cation': cation_labels_test,
            'Batch': batch_labels_test  # New: include batch labels
        })
        results_df.to_csv(os.path.join(peak_dir, 'prediction_results.csv'), index=False)
        
        # Calculate R² for each cation
        cation_r2_dict = {}
        for cation in set(cation_labels_test):
            mask = np.array(cation_labels_test) == cation
            if sum(mask) > 0:  # Only if we have test data for this cation
                cation_r2_dict[cation] = r2_score(y_true_original[mask], y_pred_original[mask])
        
        # NEW: Calculate R² for each batch
        batch_r2_dict = {}
        for batch in set(batch_labels_test):
            mask = np.array(batch_labels_test) == batch
            if sum(mask) > 0:  # Only if we have test data for this batch
                batch_r2_dict[batch] = r2_score(y_true_original[mask], y_pred_original[mask])
        
        # Save summary statistics
        with open(os.path.join(peak_dir, 'summary_statistics.txt'), 'w') as f:
            f.write(f"Model for predicting '{target_peak}' using other peaks\n")
            f.write(f"Overall MSE: {mse:.4f}\n")
            f.write(f"Overall R²: {r2:.4f}\n")
            f.write("\nCation-Specific R² Values:\n")
            for cation, r2_val in cation_r2_dict.items():
                f.write(f"{cation}: {r2_val:.4f}\n")
            
            f.write("\nBatch-Specific R² Values:\n")  # New: batch-specific results
            for batch, r2_val in batch_r2_dict.items():
                f.write(f"{batch}: {r2_val:.4f}\n")
            
            f.write(f"\nTop {top_n} important features:\n")
            for i, feature in enumerate(top_features):
                idx = feature_peaks.index(feature)
                f.write(f"{i+1}. {feature}: {feature_importance[idx]:.6f}\n")
        
        # Store results for summary
        results.append({
            'target_peak': target_peak,
            'r2': r2,
            'mse': mse,
            'cation_r2': cation_r2_dict,
            'batch_r2': batch_r2_dict,  # New: batch-specific results
            'top_features': top_features,
            'feature_importance': {feat: imp for feat, imp in zip(feature_peaks, feature_importance)}
        })
    
    # Create summary results DataFrame and visualizations (same as before)
    summary_df = pd.DataFrame({
        'Peak': [r['target_peak'] for r in results],
        'R²': [r['r2'] for r in results],
        'MSE': [r['mse'] for r in results]
    })
    
    # Sort by R² to see which peaks can be best predicted
    summary_df = summary_df.sort_values('R²', ascending=False)
    
    # Save summary results
    summary_df.to_csv(os.path.join(output_dir, 'peak_prediction_summary.csv'), index=False)
    
    # [Rest of the plotting code remains the same...]
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return summary_df, results

def plot_batch_predictions(y_true, y_pred, batches, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Define colors for each batch
    batch_colors = {'batch_1': '#1f77b4', 'batch_2': '#ff7f0e', 'batch_3': '#2ca02c', 
                    'batch_4': '#d62728', 'batch_5': '#9467bd'}
    
    # Plot points with colors by batch
    for batch in set(batches):
        mask = np.array(batches) == batch
        if sum(mask) > 0:  # Only if we have data for this batch
            plt.scatter(
                y_true[mask], 
                y_pred[mask], 
                color=batch_colors.get(batch, 'gray'),
                label=f"{batch} (R²={r2_score(y_true[mask], y_pred[mask]):.4f})",
                alpha=0.7,
                s=80,
                edgecolor='white',
                linewidth=0.5
            )
    
    # Add diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    # Add overall R² value
    overall_r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    textstr = f'Overall $R^2 = {overall_r2:.4f}$\nOverall $MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=14, bbox=props, verticalalignment='top')
    
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()  # Close the figure to free memory
    return overall_r2, mse

# Run the leave-one-out analysis
if __name__ == "__main__":
    print("Starting leave-one-out analysis for all FTIR peaks...")
    summary, results = analyze_leave_one_out()
    print("Analysis complete!")
