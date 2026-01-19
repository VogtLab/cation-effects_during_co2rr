import numpy as np
import torch

from analysis import analyze_leave_one_out

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


experiment_data = [
    {'path': 'path/to/your/data.csv', 'cation': 'Na'},
]

output_dir = './output'


if __name__ == "__main__":
    print("Starting leave-one-out analysis for all FTIR peaks...")
    summary, results = analyze_leave_one_out(
        experiment_data=experiment_data,
        output_dir=output_dir,
        time_steps=5,
        epochs=100,
        batch_size=16,
        dropout_rate=0.2
    )
    print("Analysis complete!")
