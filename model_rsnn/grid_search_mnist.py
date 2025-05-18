import pickle
import torch
import numpy as np
from mnist_exp.training_biograd import biograd_snn_training
from biograd_snn.network_w_biograd import BioGradNetworkWithSleep
from biograd_snn.online_error_functions import cross_entropy_loss_error_function, LossFunction
from lr_schedulers import ExponentialDecay, CosineAnnealingWarmRestarts
from sklearn.model_selection import ParameterGrid
import time
import os
import gc
import sys
import re
import traceback

import torch.multiprocessing as mp
from itertools import product
import pandas as pd

def train_biograd_shd(snn_param, hidden_dim_list, float_mode, aligned_weights, loss_mode, seed_list=0, 
                        num_bits=16, weight_init_bits=12, low_precision_bits=8, activation_bits=16, spike_stats=False, batch_stats=False, plot_batch=False, 
                        epochs=100, exp_name='', data_params=0, train_batch_size=128, test_batch_size=256, lr_scheduler=0, 
                        loss_function=LossFunction, gradient_clip=0, loss_precision=256, tf_folder='./runs', save_folder='./save_models'):
    """
    Train a BioGrad SNN on MNIST dataset with given parameters.

    Args:
    - snn_param: dict, parameters for the SNN
    - float_mode: False: Int, True: Float
    - aligned_weights: True: Aligned feedback weights, False: Transposed feedforward weights
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
    in_shape = 28*28
    out_dim = 10

    # soft_error_start = 5
    spike_ts = 20
    # sleep_spike_ts = 50

    # Define Training parameters
    val_size = 10000
    train_batch_size = train_batch_size
    sleep_batch_size = 128
    test_batch_size = test_batch_size
    epoch = epochs
    sleep_oja_power = 2.0
    sleep_lr = 1.0e-4 / 3.

    loss_precision = 1 if float_mode else loss_precision

    # Define SNN and start training
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # session_name = "grid_search_" + "_".join(exp_name.split("_")[:-1])
        session_name = f"mnist_biograd_rsnn_{str(len(snn_param['hidden_layer']))}_layers_"
        session_name += "float_" if float_mode else "int_"
        if not float_mode:
            session_name += f"{num_bits}_{weight_init_bits}_{low_precision_bits}_"
        session_name += "aligned_" if aligned_weights else "transposed_"
        session_name += "loss_" + loss_mode + "_"
        # if data_params['duration'] is None:
        #     session_name += f"frames_{data_params['frames']}_{data_params['split_by']}_"
        # else:
        #     session_name += f"duration_{data_params['duration']}_"
        session_name += str(seed)
        session_name += "_exp_id_" + exp_name.split("_")[-1]

        online_snn = BioGradNetworkWithSleep(in_shape, out_dim, hidden_dim_list,
                                            snn_param, loss_function, device, float_mode=float_mode,
                                            aligned_weights = aligned_weights, loss_mode=loss_mode, bias=False, gradient_clip=gradient_clip,
                                            num_bits=num_bits, weight_init_bits=weight_init_bits, low_precision_bits=low_precision_bits, activation_bits=activation_bits,
                                            terminal_stats=False)
        
        train_acc, val_acc, test_acc, fb_angle, fb_ratio, best_stats = biograd_snn_training(
            online_snn, spike_ts, device, session_name,
            batch_size=train_batch_size, sleep_batch_size=sleep_batch_size,
            test_batch_size=test_batch_size, epoch=epoch, train_stats=False,
            sleep_oja_power=sleep_oja_power, sleep_lr=sleep_lr, loss_precision=loss_precision, 
            stats=(False, False, False), float_mode=float_mode, lr_scheduler=lr_scheduler, 
            writer=(False, False, False), tf_folder=tf_folder, save_folder=save_folder)
        
        pickle.dump(train_acc, open(save_folder + session_name + "/train_accuracy_list.p", "wb+"))
        pickle.dump(val_acc, open(save_folder + session_name + "/val_accuracy_list.p", "wb+"))
        pickle.dump(test_acc, open(save_folder + session_name + "/test_accuracy_list.p", "wb+"))
        pickle.dump(fb_angle, open(save_folder + session_name + "/feedback_angle_list.p", "wb+"))
        pickle.dump(fb_ratio, open(save_folder + session_name + "/feedback_ratio_list.p", "wb+"))

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
        f = params['param_factor']

        out_layer_dict = {'Vdecay': params['vdecay'], 'Vth': f*params['output_param'][0], 'Grad_win': f*params['output_param'][1], 'Grad_amp': 1., 
                          'Fb_th': 1., 'lr': params['lr_output'], 'Weight_decay': 0}
        hidden_layer_dict = {'Vdecay': params['vdecay'], 'Vth': f*params['hidden_param'][0], 'Grad_win': f*params['hidden_param'][1], 'Grad_amp': 1., 
                             'lr': params['lr_hidden'], 'Weight_decay': 0, 'lr_rec': params['lr_rec'], 'Vdecay_rec': params['vdecay_rec'],
                             'fw_factor': params['fw_factor'], 'rec_factor': params['rec_factor']}
    
        snn_params = {'out_layer': out_layer_dict, 'hidden_layer': [hidden_layer_dict]}

        # Loss function
        loss_function = LossFunction(params['loss_function'][0], parameters=params['loss_function'][1])

        # Learning rate scheduler
        if params['lr_scheduler'] == 'none':
            lr_scheduler = 0
        elif params['lr_scheduler'] == 'decay':
            lr_scheduler = ExponentialDecay(initial_lr=1, decay_factor=1.1)
        elif params['lr_scheduler'] == 'cosine':
            lr_scheduler = CosineAnnealingWarmRestarts(initial_lr=1, T_0=10, T_mult=2)
        
        # Define save folder and tensorboard folder
        experiment_group = "mnist/grid_search_" + config['exp_name'] + "/"
        tf_folder = "./runs/" + experiment_group
        save_folder = "./save_models/" + experiment_group

        train_acc, val_acc, test_acc, _, _, best_stats, name = train_biograd_shd(snn_params, params['hidden_dim_list'], params['float_mode'], 
                                                                                params['aligned_weights'], params['loss_mode'], seed_list=[params['seed']], 
                                                                                num_bits=params['num_bits'], weight_init_bits=params['weight_init_bits'], low_precision_bits=params['low_precision_weights'], activation_bits=params['activation_bits'],
                                                                                spike_stats=False, batch_stats=False, epochs=params['epochs'], exp_name=exp_name,
                                                                                data_params=params['data_params'], train_batch_size=params['train_batch_size'], test_batch_size=params['test_batch_size'],
                                                                                lr_scheduler=lr_scheduler, loss_function=loss_function, gradient_clip=params['gradient_clip'],
                                                                                loss_precision=params['loss_precision'],
                                                                                tf_folder=tf_folder, save_folder=save_folder)

        fb_angle, fb_ratio, train_activations_fb, train_activations, val_activations, test_activations = best_stats

        best_model = np.argmax(val_acc)
        
        config_results = {
            'id': params['id'],
            'train_acc': train_acc[best_model],
            'val_acc': val_acc[best_model],
            # 'test_acc': test_acc[best_model],
            'best_epoch': best_model,
            'Neurons': params['hidden_dim_list'],
            "loss_function": params['loss_function'],
            "loss_precision": params['loss_precision'],
            'num_bits': params['num_bits'],
            'weight_init_bits': params['weight_init_bits'],
            'low_precision_weights': params['low_precision_weights'],
            'activation_bits': params['activation_bits'],
            'vdecay': params['vdecay'],
            'vedecay_rec': params['vdecay_rec'],
            'hidden_layer': [f*params['hidden_param'][0], f*params['hidden_param'][1]],
            'out_layer': [f*params['output_param'][0], f*params['output_param'][1]],
            'lr_hidden': params['lr_hidden'],
            'lr_hidden_rec': params['lr_rec'],
            'lr_output': params['lr_output'],
            'lr_scheduler': params['lr_scheduler'],
            "gradient_clip": params['gradient_clip'],
            'fw_factor': params['fw_factor'],
            'rec_factor': params['rec_factor'],
            'data_params': params['data_params'],
            'train_activations': train_activations,
            'val_activations': val_activations,
            # 'test_activations': test_activations,
            'train_activations_fb': train_activations_fb,
            'train_batch_size': params['train_batch_size'],
            'test_batch_size': params['test_batch_size'],
            'fb_angle': fb_angle,
            'fb_ratio': fb_ratio,
            'session_name': name,
            'seed': params['seed'],
            'epochs': params['epochs'],
        }

        pickle.dump(config_results, open("./save_models/mnist/grid_search_" + config['exp_name'] + f"/{name}/config_results.p", "wb+"))
        return config_results
    
    except Exception as e:
            print(f"Error in configuration {config['id']}: {e}")
            traceback.print_exc()
            log_file_folder = f"./save_models/mnist/grid_search_{config['exp_name']}/exceptions/"
            if not os.path.exists(log_file_folder):
                os.makedirs(log_file_folder)
            log_file_name = f"{log_file_folder}config_{config['id']}.txt"
            with open(log_file_name, "w") as log_file:
                log_file.write(f"Error in configuration {config['id']}: {e}\n")
                log_file.write(traceback.format_exc())
            return {
                'id': params['id'],
                'train_acc': 0,
                'val_acc': 0,
                # 'test_acc': 0,
                'best_epoch': 0,
                'Neurons': params['hidden_dim_list'],
                "loss_function": params['loss_function'],
                "loss_precision": params['loss_precision'],
                'num_bits': params['num_bits'],
                'weight_init_bits': params['weight_init_bits'],
                'low_precision_weights': params['low_precision_weights'],
                'activation_bits': params['activation_bits'],
                'vdecay': params['vdecay'],
                'vedecay_rec': params['vdecay_rec'],
                'hidden_layer': [params['param_factor']*params['hidden_param'][0], params['param_factor']*params['hidden_param'][1]],
                'out_layer': [params['param_factor']*params['output_param'][0], params['param_factor']*params['output_param'][1]],
                'lr_hidden': params['lr_hidden'],
                'lr_hidden_rec': params['lr_rec'],
                'lr_output': params['lr_output'],
                'lr_scheduler': params['lr_scheduler'],
                "gradient_clip": params['gradient_clip'],
                'fw_factor': params['fw_factor'],
                'rec_factor': params['rec_factor'],
                'data_params': params['data_params'],
                'train_activations': 0,
                'val_activations': 0,
                # 'test_activations': 0,
                'train_activations_fb': 0,
                'train_batch_size': params['train_batch_size'],
                'test_batch_size': params['test_batch_size'],
                'fb_angle': 0,
                'fb_ratio': 0,
                'session_name': '',
                'seed': params['seed'],
                'epochs': params['epochs'],
            }

def grid_search(grid):
    configs = [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
    for idx, config in enumerate(configs):
        config["id"] = idx
    return configs

def extract_config_from_subfolders(base_folder, exp_name):
    """
    Extract the configuration files from each experiment
    """
    model_pattern = re.compile(rf"_{exp_name}_\d+$")
    results = []

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
    """
    Convert list of dictionaries to a pandas DataFrame
    """
    df = pd.DataFrame(configs)
    return df

if __name__ == "__main__":
    start = time.time()
    mp.set_start_method("spawn")  # Use spawn method for multiprocessing

    # Hyperparameter grid
    grid = {
        'exp_name': ['mnist_rsnn_float_v4_feedback'],
        'epochs': [50],
        'seed': [100364571],
        'train_batch_size': [128],
        'test_batch_size': [256],
        'float_mode': [True],
        'aligned_weights': [False],
        'loss_mode': ['feedback'],
        'num_bits': [16],
        'weight_init_bits': [16], #
        'low_precision_weights': [11],
        'activation_bits': [15],
        'data_params': [0],
        'hidden_dim_list': [[100]],
        'hidden_param': [[1., 2.], [2., 4.], [4., 6.]], #
        'output_param': [[1., 2.], [4., 6.], [10., 10.]], #
        'vdecay': [0.5],
        'vdecay_rec': [0.5], #
        'param_factor': [1.],
        'lr_hidden': [3e-3, 3e-2], #
        'lr_output': [3e-3, 3e-2], #
        'lr_rec': [0], #
        'lr_scheduler': ['none'], #
        'fw_factor': [1.],
        'rec_factor': [1.],
        "gradient_clip": [0, 1e-4], #, 256, 512
        # "rec_scaling": [16],
        "loss_function": [('normal', None)],
        "loss_precision": [128], # 
    }

    print("Total combinations: ", len(list(ParameterGrid(grid))))

    # Define paths
    exp_name = grid['exp_name'][0]
    experiments_folder = "save_models/mnist/grid_search_" + exp_name
    server = "dacs" # snellius, dacs
    grid_results_folder = "grid_results_mnist_" + server

    os.makedirs(experiments_folder, exist_ok=True)
    sys.stdout = open(f"{experiments_folder}/{exp_name}_output.txt", 'w')
    sys.stderr = open(f"{experiments_folder}/{exp_name}_error.txt", 'w')
    
    # Generate all configurations
    configs = grid_search(grid)
    
    # Set the maximum number of parallel processes
    max_processes = 2 # ADJUST torch.cuda.set_per_process_memory_fraction(0.15)
    
    # Use a multiprocessing Pool to run configurations in parallel
    with mp.Pool(max_processes) as pool:
        results = pool.map(train_and_evaluate, configs)

    configs = extract_config_from_subfolders(experiments_folder, exp_name)

    # Save grid search information into a txt file
    with open(f"{grid_results_folder}/grid_search_info_{exp_name}.txt", "w") as file:
        file.write(f"Experiment Name: {grid['exp_name'][0]}\n")
        file.write(f"Total combinations: {str(len(list(ParameterGrid(grid))))} \n")
        file.write(f"Server: {server}\n")
        file.write(f"Finished grid search in {str(time.time() - start)} seconds\n")
        file.write(f"Average training time: {str((time.time() - start) / len(configs))} seconds\n\n")
        for key, value in grid.items():
            file.write(f"{key}: {value}\n")
    
    # Save results to a CSV file
    configs_df = configs_to_dataframe(configs)
    configs_df.to_csv(f"{grid_results_folder}/grid_results_{exp_name}.csv", index=False)
    print(f"Saved {len(configs)} configurations to {grid_results_folder}/grid_results_{exp_name}.csv")

    sorted_results = configs_df.sort_values(by="val_acc", ascending=False)

    # Write results to log file
    log_file_name = f"{experiments_folder}/{exp_name}_results_log.txt"

    with open(log_file_name, "w") as log_file:
        log_file.write("Grid Search Results sorted by validation accuracy:\n")
        log_file.write(f"Experiment Name: {grid['exp_name'][0]}\n")
        log_file.write(f"Server: {server}\n")
        log_file.write(f"Total combinations: {str(len(list(ParameterGrid(grid))))} \n")
        log_file.write(f"Finished grid search in {str(time.time() - start)} seconds\n")
        log_file.write(f"Average training time: {str((time.time() - start) / len(sorted_results))} seconds\n")
        log_file.write(f"Seed: {grid['seed']}\n")
        log_file.write(f"Epochs: {grid['epochs']}\n")
        log_file.write(f"Float Mode: {grid['float_mode']}\n")
        log_file.write(f"Aligned Weights: {grid['aligned_weights']}\n")
        log_file.write(f"Loss Mode: {grid['loss_mode']}\n")
        log_file.write(f"Number of Bits: {grid['num_bits']}\n")
        log_file.write(f"Weight Initialization Bits: {grid['weight_init_bits']}\n")
        log_file.write(f"Low Precision Weights: {grid['low_precision_weights']}\n\n")

        for idx, row in sorted_results.iterrows():
            for column, value in row.items():
                log_file.write(f"{column}: {value}\n")
            log_file.write("\n")
