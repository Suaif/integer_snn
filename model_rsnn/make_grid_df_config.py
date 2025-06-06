import re
import os
import pickle
import pandas as pd

def extract_config_from_subfolders(base_folder, exp_name):
    model_pattern = re.compile(rf"_{exp_name}_\d+$")
    results = []

    # Iterate through subfolders in the base folder
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)

        if os.path.isdir(subfolder_path):

            config_file_path = os.path.join(subfolder_path, "config_results.p")
            try:
                with open(config_file_path, 'rb') as file:
                    config_data = pickle.load(file)
                    results.append(config_data)
            except FileNotFoundError:
                print(f"Config file not found in: {subfolder_path}")
            except Exception as e:
                print(f"Error reading config file in {subfolder_path}: {e}")

    return results

def configs_to_dataframe(configs):
    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(configs)
    return df

exp_name = "mnist_rsnn_float_v4"
base_folder = "./save_models/mnist/grid_search_" + exp_name
server = "snellius" # snellius or dacs
save_folder = "./grid_results_mnist_" + server

configs = extract_config_from_subfolders(base_folder, exp_name)
if configs:
    configs_df = configs_to_dataframe(configs)
    configs_df.to_csv(f"{save_folder}/grid_results_{exp_name}.csv", index=False)
    print(f"Saved {len(configs)} configurations to {save_folder}/grid_results_{exp_name}.csv")
    print(f"Total configurations in {base_folder}: {len(os.listdir(base_folder))}")
else:
    print(f"No configurations found in {base_folder}")
