import pickle
import torch
import numpy as np
from mnist_exp.training_biograd import biograd_snn_training
from biograd_snn.network_w_biograd import BioGradNetworkWithSleep
from biograd_snn.online_error_functions import cross_entropy_loss_error_function, l2_loss
import plot_training
from sklearn.model_selection import ParameterGrid
import time
import os
import gc
import sys

import torch.multiprocessing as mp
from itertools import product
import pandas as pd
from torch.utils.data import TensorDataset

def train_biograd_mnist(snn_param, hidden_dim_list, float_mode, feedback_weights, loss_mode, seed_list=0, 
                        num_bits=16, weight_init_bits=12, spike_stats=False, batch_stats=False, epochs=100, 
                        exp_name='', data=0):
    """
    Train a BioGrad SNN on MNIST dataset with given parameters.

    Args:
    - snn_param: dict, parameters for the SNN
    - float_mode: False: Int, True: Float
    - feedback_weights: True: Aligned feedback weights, False: Transposed feedforward weights
    - loss_mode:
        'feedback': traditional (accumulated loss over all time steps)
        'final': loss only on final step
        'direct': loss only on final step not processed in OutputLayer.feedbackstep
    - seed_list: list of int seeds to train the SNN with
    - num_bits: int, number of bits to quantize weights to
    - weight_init_bits: int, number of bits to quantize initial weights to

    """

    # Define SNN parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_shape = 1 * 28 * 28
    out_dim = 10

    soft_error_start = 5
    spike_ts = 20
    sleep_spike_ts = 50

    # Define Training parameters
    val_size = 10000
    train_batch_size = 128
    sleep_batch_size = 128
    test_batch_size = 256
    epoch = epochs
    save_epoch = 1
    sleep_oja_power = 2.0
    sleep_lr = 1.0e-4 / 3.

    if float_mode:
        loss_precision = 1
        lr = 1e-3
    else:
        loss_precision = 32
        lr = 1

    # Define SNN and start training
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        session_name = "grid_search_" + "_".join(exp_name.split("_")[:-1]) + "/mnist_biograd_"
        session_name += "float_" if float_mode else "int_"
        session_name += "aligned_" if feedback_weights else "transposed_"
        session_name += "loss_" + loss_mode + "_"
        session_name += str(seed)
        session_name += "_exp_" + exp_name

        plot_title = session_name + "\n MNIST with BioGrad"

        online_snn = BioGradNetworkWithSleep(in_shape, out_dim, hidden_dim_list,
                                            snn_param, l2_loss, device, float_mode=float_mode, 
                                            feedback_weights = feedback_weights, loss_mode=loss_mode, terminal_stats=False)
        if not float_mode:
            online_snn.quantize_network(num_bits, weight_init_bits)
        train_acc, val_acc, test_acc, fb_angle, fb_ratio, best_stats = biograd_snn_training(
            online_snn, spike_ts, sleep_spike_ts, device, soft_error_start, session_name,
            validation_size=val_size, batch_size=train_batch_size, sleep_batch_size=sleep_batch_size,
            test_batch_size=test_batch_size, epoch=epoch, save_epoch=save_epoch, lr=lr,
            sleep_oja_power=sleep_oja_power, sleep_lr=sleep_lr, loss_precision=loss_precision, 
            stats=(spike_stats, batch_stats), float_mode=float_mode, train_stats=False, data=data)

        pickle.dump(train_acc, open("./save_models/" + session_name + "/train_accuracy_list.p", "wb+"))
        pickle.dump(val_acc, open("./save_models/" + session_name + "/val_accuracy_list.p", "wb+"))
        pickle.dump(test_acc, open("./save_models/" + session_name + "/test_accuracy_list.p", "wb+"))
        pickle.dump(fb_angle, open("./save_models/" + session_name + "/feedback_angle_list.p", "wb+"))
        pickle.dump(fb_ratio, open("./save_models/" + session_name + "/feedback_ratio_list.p", "wb+"))

        plot_training.plot_training(train_acc, val_acc, test_acc, "./save_models/" + session_name + "/training_plot.png", title=plot_title)
        plot_training.plot_fb(fb_angle, fb_ratio, "./save_models/" + session_name + "/feedback_plot.png", title=plot_title)

        return train_acc, val_acc, test_acc, fb_angle, fb_ratio, best_stats, session_name
        

def train_and_evaluate(config):
    """
    Train a model with given hyperparameters and evaluate its performance
    
    :param params: Dictionary of hyperparameters
    :return: Dictionary with params and performance metrics
    """

    try:
        # torch.cuda.set_per_process_memory_fraction(0.05)
        params = config
        exp_name = config['exp_name'] + "_" + str(config['id'])
        snn_params = {
            'out_layer': (params['vdecay'], params['output_param'][0], params['output_param'][1], 1., params['output_param'][2], params['lr_output']),
            'hidden_layer': [(params['vdecay'], params['hidden_param'][0][0], params['hidden_param'][0][1], 1., params['lr_hidden'][0]),
                             (params['vdecay'], params['hidden_param'][1][0], params['hidden_param'][1][1], 1., params['lr_hidden'][1])]
        }
        train_acc, val_acc, test_acc, _, _, best_stats, name = train_biograd_mnist(snn_params, params['hidden_dim_list'], params['float_mode'], 
                                                                                params['feedback_weights'], params['loss_mode'], seed_list=[params['seed']], 
                                                                                num_bits=params['num_bits'], weight_init_bits=params['weight_init_bits'], 
                                                                                spike_stats=False, batch_stats=False, epochs=params['epochs'], exp_name=exp_name,
                                                                                data=params['data'])

        fb_angle, fb_ratio, train_activations_fb, train_activations, val_activations, test_activations = best_stats

        best_model = np.argmax(val_acc)
        
        config_results = {
            'id': params['id'],
            'snn_params': snn_params,
            'seed': params['seed'],
            'epochs': params['epochs'],
            'train_acc': train_acc[best_model],
            'val_acc': val_acc[best_model],
            'test_acc': test_acc[best_model],
            'best_epoch': best_model,
            'train_activations_fb': train_activations_fb,
            'train_activations': train_activations,
            'val_activations': val_activations,
            'test_activations': test_activations,
            'fb_angle': fb_angle,
            'fb_ratio': fb_ratio,
            'session_name': name
        }

        pickle.dump(config_results, open(f"./save_models/{name}/config_results.p", "wb+"))
        return config_results
    
    except Exception as e:
            print(f"Error in configuration {config['id']}: {e}")
            log_file_name = f"./save_models/grid_search_{config['exp_name']}/exception_{config['id']}.txt"
            with open(log_file_name, "w") as log_file:
                log_file.write(f"Error in configuration {config['id']}: {e}")
            return {
                'id': config['id'],
                'snn_params': 0,
                'seed': params['seed'],
                'epochs': params['epochs'],
                'train_acc': 0,
                'val_acc': 0,
                'test_acc': 0,
                'best_epoch': 0,
                'train_activations_fb': 0,
                'train_activations': 0,
                'val_activations': 0,
                'test_activations': 0,
                'fb_angle': 0,
                'fb_ratio': 0,
                'session_name': ''
            }

def grid_search(grid):
    configs = [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
    for idx, config in enumerate(configs):
        config["id"] = idx
    return configs

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, sampler, SubsetRandomSampler

def preload_data(batch_size, validation_size, test_batch_size, device="cuda"):
    # Data path and transformations
    data_path = './data/'
    transform = transforms.ToTensor()

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    # Split into train and validation
    train_idx = list(range(len(train_dataset) - validation_size))
    val_idx = list(range(len(train_idx), len(train_dataset)))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, sampler=val_sampler, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Preload data to GPU if necessary
    train_data = [(x.to(device), y.to(device)) for x, y in train_dataloader]
    val_data = [(x.to(device), y.to(device)) for x, y in val_dataloader]
    test_data = [(x.to(device), y.to(device)) for x, y in test_dataloader]

    return train_data, val_data, test_data
   

if __name__ == "__main__":
    start = time.time()
    mp.set_start_method("spawn")  # Use spawn method for multiprocessing

    # Hyperparameter grid
    grid = {
        'exp_name': ['2_layers'],
        'epochs': [50],
        'seed': [100364571],
        'float_mode': [False],
        'feedback_weights': [False],
        'loss_mode': ['direct', 'feedback'],
        'num_bits': [16],
        'weight_init_bits': [12],
        'hidden_dim_list': [[500, 100]],
        'hidden_params': [[[0.3, 0.6], [0.3, 0.3]],[[0.3, 0.6], [0.3, 0.3]], [[0.3, 0.6], [0.3, 0.3]]],
        'output_param': [[0.3, 0.6, 1.], [0.3, 0.6, 1.]],
        'vdecay': [0.5, 1.],
        'lr_hidden': [[-6, -6]],
        'lr_output': [3],
    }
    
    print("Total combinations: ", len(list(ParameterGrid(grid))))
    sys.stdout = sys.stderr = open(f"grid_search_{grid['exp_name'][0]}_logs.txt", 'w')
    
    # Preload the data
    train_batch_size = 256
    validation_size = 10000
    test_batch_size = 512

    # Preload data
    train_data, val_data, test_data = preload_data(train_batch_size, validation_size, test_batch_size)
    # grid['data'] = [[train_data, val_data, test_data]]
    grid['data'] = [0]
    
    # Generate all configurations
    configs = grid_search(grid)
    os.makedirs(f"save_models/grid_search_{grid['exp_name'][0]}/", exist_ok=True)
    
    # Set the maximum number of parallel processes
    max_processes = 4 # ADJUST torch.cuda.set_per_process_memory_fraction(0.15)
    
    # Use a multiprocessing Pool to run configurations in parallel
    with mp.Pool(max_processes) as pool:
        results = pool.map(train_and_evaluate, configs)

    # Collect results into a DataFrame
    results_df = pd.DataFrame({
        "id": [r["id"] for r in results],
        "snn_params": [r["snn_params"] for r in results],
        "seed": [r["seed"] for r in results],
        "epochs": [r["epochs"] for r in results],
        "train_acc": [r["train_acc"] for r in results],
        "val_acc": [r["val_acc"] for r in results],
        "test_acc": [r["test_acc"] for r in results],
        "best_epoch": [r["best_epoch"] for r in results],
        "train_activations_fb": [r["train_activations_fb"] for r in results],
        "train_activations": [r["train_activations"] for r in results],
        "val_activations": [r["val_activations"] for r in results],
        "test_activations": [r["test_activations"] for r in results],
        "fb_angle": [r["fb_angle"] for r in results],
        "fb_ratio": [r["fb_ratio"] for r in results],
        "session_name": [r["session_name"] for r in results]
    })

    # Save results to a CSV file
    results_df.to_csv(f"grid_results/grid_search_results_exp_{grid['exp_name'][0]}.csv", index=False)

    sorted_results = results_df.sort_values(by="val_acc", ascending=False)

    # Write results to log file
    log_file_name = f"grid_results/grid_search_log_{grid['exp_name'][0]}.txt"

    with open(log_file_name, "w") as log_file:
        log_file.write("Grid Search Results sorted by validation accuracy:\n")
        log_file.write(f"Experiment Name: {grid['exp_name'][0]}\n")
        log_file.write(f"Total combinations: {str(len(list(ParameterGrid(grid))))} \n")
        log_file.write(f"Finished grid search in {str(time.time() - start)} seconds\n")
        log_file.write(f"Average training time: {str((time.time() - start) / len(sorted_results))} seconds\n")
        log_file.write(f"Seed: {grid['seed']}\n")
        log_file.write(f"Epochs: {grid['epochs']}\n")
        log_file.write(f"Float Mode: {grid['float_mode']}\n")
        log_file.write(f"Feedback Weights: {grid['feedback_weights']}\n")
        log_file.write(f"Loss Mode: {grid['loss_mode']}\n")
        log_file.write(f"Number of Bits: {grid['num_bits']}\n")
        log_file.write(f"Weight Initialization Bits: {grid['weight_init_bits']}\n\n")

        for idx, row in sorted_results.iterrows():
            log_file.write(f"ID: {row['id']}\n")
            log_file.write(f"snn_params: {row['snn_params']}\n")
            log_file.write(f"Session Name: {row['session_name']}\n")
            # log_file.write(f"Seed: {row['seed']}\n")
            # log_file.write(f"Epochs: {row['epochs']}\n")
            # log_file.write(f"Float Mode: {row['float_mode']}\n")
            # log_file.write(f"Feedback Weights: {row['feedback_weights']}\n")
            # log_file.write(f"Loss Mode: {row['loss_mode']}\n")
            # log_file.write(f"Number of Bits: {row['num_bits']}\n")
            # log_file.write(f"Weight Initialization Bits: {row['weight_init_bits']}\n")
            log_file.write(f"Train Accuracy: {row['train_acc']}\n")
            log_file.write(f"Validation Accuracy: {row['val_acc']}\n")
            log_file.write(f"Test Accuracy: {row['test_acc']}\n")
            log_file.write(f"Best Epoch: {row['best_epoch']}\n")
            log_file.write(f"Train Activations FB: {row['train_activations_fb']}\n")
            log_file.write(f"Train Activations: {row['train_activations']}\n")
            log_file.write(f"Validation Activations: {row['val_activations']}\n")
            log_file.write(f"Test Activations: {row['test_activations']}\n")
            log_file.write(f"Feedback Angle: {row['fb_angle']}\n")
            log_file.write(f"Feedback Ratio: {row['fb_ratio']}\n")
            log_file.write("\n")