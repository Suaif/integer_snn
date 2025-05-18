import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import pickle
import copy
from datetime import datetime
from mnist_exp.utility import img_2_event_img
from quantization import quantize_image, print_stats
import plot_training


def biograd_snn_training(network, spike_ts, device, soft_error_step, session_name,
                         validation_size=10000, batch_size=128, sleep_batch_size=128,
                         test_batch_size=128, epoch=100,
                         save_epoch=1,
                         sleep_oja_power=2, sleep_lr=1e-3, loss_precision=32,
                         stats=False, float_mode=False, train_stats=True, lr_scheduler=0, 
                         writer=(False, False, True), tf_folder="./runs/", save_folder="./save_models/", tf_name=None):
    """
    BioGrad SNN training with sleep

    Args:
        network (SNN): Online learning SNN
        spike_ts (int): Spike timestep
        device (torch.device): Device to run the training on
        soft_error_step (int): Soft start error step for feedback simulation
        session_name (str): Name of the training session
        validation_size (int): Size of validation set
        batch_size (int): Batch size for training
        sleep_batch_size (int): Batch size for sleep
        test_batch_size (int): Batch size for testing
        epoch (int): Number of epochs
        save_epoch (int): Save model every `save_epoch` epochs
        sleep_oja_power (float): Oja power for sleep phase
        sleep_lr (float): Learning rate for sleep phase
        loss_precision (int): Precision for loss computation
        stats (bool): Whether to collect statistics
        float_mode (bool): Whether to use floating-point mode
        train_stats (bool): Whether to print training statistics
        lr_scheduler (int): Learning rate scheduler
        writer (tuple): Tensorboard writer options
        tf_folder (str): Folder for tensorboard logs
        save_folder (str): Folder to save models
        tf_name (str): Tensorboard log name

    Returns:
        tuple: Training, validation, and test accuracy lists, feedback angle, feedback ratio, and best stats
    """
    # Create folder for saving models
    os.makedirs(save_folder, exist_ok=True)

    session_file_dir = os.path.join(save_folder, session_name)
    os.makedirs(session_file_dir, exist_ok=True)
    print(f"Directory {session_file_dir} created or already exists")

    data_path = './data/'
    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                               transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                              transform=transforms.ToTensor())

    # Train, validation, and test dataloaders
    train_idx = list(range(len(train_dataset) - validation_size))
    val_idx = list(range(len(train_idx), len(train_dataset)))
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  shuffle=False, num_workers=0)
    val_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, sampler=val_sampler,
                                shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 shuffle=False, num_workers=0)

    # Number of samples in train, validation, and test datasets
    train_num = len(train_idx)
    val_num = len(val_idx)
    test_num = len(test_dataset)

    # Quantize weights
    if not float_mode:
        network.quantize_network()
    network.prepare_inference_weights(train_stats)

    # Prepare learning rate
    if lr_scheduler != 0:
        network.set_initial_lr()

    # Initialize tensorboard writer
    log_dir = os.path.join(tf_folder, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + session_name) if tf_name is None else os.path.join(tf_folder, tf_name)
    tf_writer = SummaryWriter(log_dir)

    writer_spike, writer_batch, writer_epoch = writer
    network.writer_batch = writer_batch
    network.tf_writer = tf_writer
    for n_layer, layer in enumerate(network.hidden_cells):
        layer.writer_spike, layer.writer_batch = writer_spike, writer_batch
        layer.tf_writer = tf_writer
        layer.tf_name = f"layer_{n_layer}"
    network.output_cell.writer_spike, network.output_cell.writer_batch = writer_spike, writer_batch
    network.output_cell.tf_writer = tf_writer
    network.output_cell.tf_name = "output_layer"

    # Define hyperparameter dictionary
    hparams = {
        'batch_size': batch_size,
        'test_batch_size': test_batch_size,
        "bias": network.bias,
        'epoch': epoch,
        'loss_precision': loss_precision,
        "n_bits": network.n_bits,
        "weight_init_bits": network.weight_init_bits,
        "low_precision_bits": network.low_precision_bits,
        'hidden_params': f"{network.hidden_cells[0].vdecay}, {network.hidden_cells[0].vth}, {network.hidden_cells[0].grad_win}",
        "output_params": f"{network.output_cell.vdecay}, {network.output_cell.vth}, {network.output_cell.grad_win}",
        "hidden_lr": float(network.hidden_cells[0].lr),
        "output_lr": float(network.output_cell.lr),
        "gradient_clip": float(network.gradient_clip),
    }

    # Add hyperparameters to tensorboard
    markdown_text = f"## **{session_name}**\n\n"
    for key, value in hparams.items():
        markdown_text += f"**{key}**: {value}\n"
    tf_writer.add_text('Hyperparameters', markdown_text)

    # Initialize training stats
    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []
    feedback_angle, feedback_ratio = [0], [0]

    # Log network summary
    summary_message = "*** Network Summary *** \n"
    summary_message += "Gradient clip: " + str(network.gradient_clip) + "\n"
    for n_layer, layer in enumerate(network.hidden_cells):
        summary_message += (f"Hidden Layer {n_layer}\n"
                            f"Neurons:  {layer.forward_func.weight.shape[0]}, {layer.forward_func.weight.shape[1]}, "
                            f"Vdecay {layer.vdecay}, Vth {layer.vth}, Grad_win {layer.grad_win}, Grad_amp {layer.grad_amp}, "
                            f"Learning rate: {layer.lr}, Weight decay {layer.weight_decay} \n")
        summary_message += print_stats(layer.forward_func.weight.cpu().detach().numpy(),        f"FW   ", False) + '\n'
        summary_message += print_stats(layer.forward_func_quant.weight.cpu().detach().numpy(),  f"FW Q ", False) + '\n'
        summary_message += print_stats(layer.feedback_func.weight.cpu().detach().numpy(),       f"FB   ", False) + '\n'
        summary_message += print_stats(layer.feedback_func_quant.weight.cpu().detach().numpy(), f"FB Q ", False) + '\n'

    summary_message += (f"Output Layer\n"
                        f"Neurons:  {network.output_cell.forward_func.weight.shape[0]}, {network.output_cell.forward_func.weight.shape[1]}, "
                        f"Vdecay {network.output_cell.vdecay}, Vth {network.output_cell.vth}, "
                        f"Grad_win {network.output_cell.grad_win}, Grad_amp {network.output_cell.grad_amp}, Feedback Vth {network.output_cell.feedback_th}, "
                        f"Learning rate: {network.output_cell.lr}, Weight decay {network.output_cell.weight_decay} \n")

    summary_message += print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(),      "FW  ", False) + '\n'
    summary_message += print_stats(network.output_cell.forward_func_quant.weight.cpu().detach().numpy(),"FW Q", False) + '\n'

    if train_stats:
        print(summary_message)

    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write("\n*****\nSNN training with BioGrad\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(summary_message)

    return train_accuracy_list, val_accuracy_list, test_accuracy_list, feedback_angle, feedback_ratio, None