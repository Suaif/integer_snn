import re
import os
import pandas as pd
from itertools import product

def extract_max_validation_accuracy_from_file(log_filename):
    # Read the log file content
    try:
        with open(log_filename, 'r') as file:
            log_content = file.read()
    except FileNotFoundError:
        print(f"File not found: {log_filename}")
        return None, None

    # Pattern to capture the best validation accuracy and its epoch
    pattern = r"New best validation accuracy at epoch (\d+): ([\d\.]+)"

    # Find all matches in the log file
    matches = re.findall(pattern, log_content)

    if not matches:
        print("No validation accuracy data found.")
        return None, None

    # Find the maximum validation accuracy and the corresponding epoch
    best_epoch, best_accuracy = max(matches, key=lambda x: float(x[1]))

    return int(best_epoch), float(best_accuracy)

def process_model_logs(base_folder, exp_name):
    model_pattern = re.compile(rf"_{exp_name}_\d+$")
    results = []

    # Iterate through subfolders in the base folder
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path) and model_pattern.search(subfolder):
            log_file_path = os.path.join(subfolder_path, "training.log")
            best_epoch, best_accuracy = extract_max_validation_accuracy_from_file(log_file_path)
            model_id = subfolder.split("_")[-1]
            if best_epoch is not None and best_accuracy is not None:
                results.append((subfolder, model_id, best_epoch, best_accuracy))

    return results

def results_to_dataframe(results):
    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results, columns=["Model", "Model_id", "Best Epoch", "Best val_acc"])
    return df

# Example usage
base_folder = "save_models"
exp_name = "exp_th_lr"

model_results = process_model_logs(base_folder, exp_name)
if model_results:
    results_df = results_to_dataframe(model_results)
    results_df.sort_values("Best val_acc", ascending=False, inplace=True)
else:
    print("No results found.")

#  Hyperparameter grid
grid = {
        'exp_name': ['th_lr'],
        'epochs': [50],
        'seed': [100364571],
        'float_mode': [False],
        'feedback_weights': [False],
        'loss_mode': ['final'],
        'num_bits': [16],
        'weight_init_bits': [12],
        'hidden_dim_list': [[100]],
        'hidden_param': [[16000, 16000], [16000, 32000], [32000, 32000]],
        'output_param': [[3000, 64000, 31], [16000, 32000, 31], [32000, 32000, 31],
                         [3000, 64000, 1], [16000, 32000, 1], [32000, 32000, 1]],
        'lr_hidden': [-6, 7, 8],
        'lr_output': [4, 5],
    }
def grid_search(grid):
    configs = [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
    for idx, config in enumerate(configs):
        config["id"] = idx
    return configs

configs = grid_search(grid)
configs_df = pd.DataFrame(configs)

# Merge the results with the hyperparameter configurations
results_df["Model_id"] = results_df["Model_id"].astype(int)
merged_df = pd.merge(results_df, configs_df, left_on="Model_id", right_on="id")
merged_df.to_csv(f"grid_results_logs_{exp_name}.csv", index=False)
print(merged_df)