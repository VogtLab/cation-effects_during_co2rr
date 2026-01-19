import numpy as np
import torch

from prediction import predict_peak

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


input_data_path = "path/to/your/data.csv"
model_path = "path/to/your/model.pt"
output_path = "./prediction_output"
target_peak = "Mean 3520"
cation_type = "Na"


if __name__ == "__main__":
    results = predict_peak(
        input_data_path=input_data_path,
        model_path=model_path,
        target_peak=target_peak,
        cation=cation_type,
        output_path=output_path,
        time_steps=5
    )
