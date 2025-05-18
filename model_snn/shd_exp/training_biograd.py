import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import pickle
import copy
from datetime import datetime
from quantization import quantize_image, print_stats
import plot_training
from shd_exp.datasets import BinnedSpikingHeidelbergDigits
from spikingjelly.datasets import pad_sequence_collate


def biograd_snn_training(network, device, session_name,
                         validation_size=10000, batch_size=128, sleep_batch_size=128,
                         test_batch_size=128, epoch=100,
                         sleep_oja_power=2, sleep_lr=1e-3, loss_precision=32,
                         stats=False, float_mode=False, train_stats=True, lr_scheduler=0, data_params=0,
                         writer=(False, False, True), tf_folder="./runs/", save_folder="./save_models/", tf_name=None):
    """
    BioGrad SNN training with sleep

    Args:
        network (SNN): Online learning SNN
        device (torch.device): Device to run the training on
        session_name (str): Name of the training session
        validation_size (int): Size of validation set
        batch_size (int): Batch size for training
        sleep_batch_size (int): Batch size for sleep
        test_batch_size (int): Batch size for testing
        epoch (int): Number of epochs
        sleep_oja_power (float): Oja power for sleep phase
        sleep_lr (float): Learning rate for sleep phase
        loss_precision (int): Precision for loss computation
        stats (bool): Whether to collect statistics
        float_mode (bool): Whether to use floating-point mode
        train_stats (bool): Whether to print training statistics
        lr_scheduler (int): Learning rate scheduler
        data_params (dict): Parameters for data loading
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

    # Set data parameters
    if data_params == 0:
        data_type, duration, frames, split_by, n_bins = 'frame', None, 20, 'number', 1
    else:
        data_type = data_params['data_type']
        duration = data_params['duration']
        frames = data_params['frames']
        split_by = data_params['split_by']
        n_bins = data_params['n_bins']

    if duration is not None:
        frames, split_by = None, None
        print(f"Duration: {duration}, Frames: {frames}, Split by: {split_by}")

    # Load datasets
    train_dataset = BinnedSpikingHeidelbergDigits('data/SHD', n_bins=n_bins, train=True, data_type=data_type, frames_number=frames, split_by=split_by, duration=duration)
    test_dataset = BinnedSpikingHeidelbergDigits('data/SHD', n_bins=n_bins, train=False, data_type=data_type, frames_number=frames, split_by=split_by, duration=duration)

    train_dataloader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=test_batch_size, shuffle=True, num_workers=0)

    # Set soft error step
    soft_error_step = 1 if network.loss_mode != 'feedback' else 5
    if network.loss_mode == 'feedback':
        print("Feedback loss mode: soft_error_step NEEDS TO BE UPDATED")

    # Print dataset stats
    if train_stats:
        print(f"Train samples: {train_dataset.length}, Test samples: {test_dataset.length}")

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

    # Configure tensorboard writers for network layers
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
    for n_layer, layer in enumerate(network.hidden_cells):
        summary_message += (f"Hidden Layer {n_layer}\n"
                            f"Neurons:  {layer.forward_func.weight.shape[0]}, {layer.forward_func.weight.shape[1]}, "
                            f"Vdecay {layer.vdecay}, Vth {layer.vth}, Grad_win {layer.grad_win}, Grad_amp {layer.grad_amp}, "
                            f"Learning rate: {layer.lr}, Weight decay {layer.weight_decay} \n")
        summary_message += print_stats(layer.forward_func.weight.cpu().detach().numpy(), f"FW Weight", False) + '\n'

    summary_message += (f"Output Layer\n"
                        f"Neurons:  {network.output_cell.forward_func.weight.shape[0]}, {network.output_cell.forward_func.weight.shape[1]}, "
                        f"Vdecay {network.output_cell.vdecay}, Vth {network.output_cell.vth}, "
                        f"Grad_win {network.output_cell.grad_win}, Grad_amp {network.output_cell.grad_amp}, Feedback Vth {network.output_cell.feedback_th}, "
                        f"Learning rate: {network.output_cell.lr}, Weight decay {network.output_cell.weight_decay} \n")

    summary_message += print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(), "FW Weight", False) + '\n'

    if train_stats:
        print(summary_message)

    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write("SNN training with BioGrad\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(summary_message)

    # Training loop (simplified for clarity)
    for ee in range(epoch):
        train_correct = 0
        for n_batch, (x, labels, _) in enumerate(train_dataloader):
            labels_one_hot = nn.functional.one_hot(labels, num_classes=network.output_dim).float() * loss_precision
            x = x.permute(0, 2, 1).float().to(device)
            labels, labels_one_hot = labels.to(device), labels_one_hot.to(device)

            predict_label, _, _, _, _, _, _, _, _ = network.train_online(
                x, labels_one_hot, soft_error_step, loss_precision, stats, float_mode=float_mode)

            train_correct += ((predict_label == labels).sum().to("cpu")).item()

        train_accuracy_list.append(train_correct / train_dataset.length)

    return train_accuracy_list, val_accuracy_list, test_accuracy_list, feedback_angle, feedback_ratio, None