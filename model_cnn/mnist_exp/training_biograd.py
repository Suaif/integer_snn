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
        spike_ts (int): spike timestep
        sleep_spike_ts (int): spike timestep for skeep
        device (device):device
        soft_error_step (int): soft start error step for feedback simulation
        session_name (str): name of the training session
        validation_size (int): size of validation set
        batch_size (int): batch size for training
        sleep_batch_size (int): batch size for sleep
        test_batch_size (int): batch size for testing
        epoch (int): number of epoches
        save_epoch (int): every number of epoch to save model
        lr (float): learning rate
        sleep_oja_power (float): oja power for oja decay for sleep
        sleep_lr (float): learning rate for sleep

    Returns:
        train_accuracy_list: list of training accuracy for each epoch
        val_accuracy_list: list of validation accuracy for each epoch
        test_accuracy_list: list of test accuracy for each epoch
        feedback_angle: list of feedback angle of each hidden layer
        feedback_ratio: list of feedback ratio of each hidden layer

    """
    # Create folder for saving models
    try:
        os.mkdir(save_folder)
    except FileExistsError:
        pass

    session_file_dir = save_folder + session_name
    try:
        os.mkdir(session_file_dir)
        print("Directory " + session_file_dir + " Created")
    except FileExistsError:
        print("Directory " + session_file_dir + " already exists")

    data_path = './data/'
    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                            transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                            transform=transforms.ToTensor())

    # Train, validation, and test dataloader
    train_idx = [idx for idx in range(len(train_dataset) - validation_size)]
    val_idx = [(idx + len(train_idx)) for idx in range(validation_size)]
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                shuffle=False, num_workers=0)
    val_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, sampler=val_sampler,
                                shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
                                shuffle=False, num_workers=0)
    # train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=num_worker,
    #                           drop_last = True, pin_memory=True, persistent_workers = True)
    # val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=True, num_workers=int(num_worker/2),
    #                         drop_last = False, pin_memory=True, persistent_workers = True)

    # Number of samples in train, validation, and test dataset
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
    
    # Lists for training stats
    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []
    hidden_weights_list, output_weights_list = [], []
    hidden_weights_quant_list, output_weights_quant_list = [], []
    deltas_list = []
    deltas_abs_list = []
    activation_list = [[[], [], []]]
    train_activation_fb_list = [[], []]
    max_volts_hist = torch.zeros(epoch, network.n_layers)
    lr_lists = []

    # Define tensorboard
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

    # Add the markdown text to tensorboard
    markdown_text = f"## **{session_name}**\n\n"
    for key, value in hparams.items():
        markdown_text += f"**{key}**: {value}\n"
    
    tf_writer.add_text('Hyperparameters', markdown_text)

    # Compute init angle and ratio between feedback weight and forward weight
    # if len(network.hidden_cells) > 0:
    #     feedback_angle, feedback_ratio = [], []
    #     angle_list, ratio_list = network.compute_feedback_angle_ratio()
    #     feedback_angle.append(angle_list)
    #     feedback_ratio.append(ratio_list)
    #     for hh in range(len(angle_list)):
    #         tf_writer.add_scalar('mnist_exp/feedback_angle_hidden' + str(hh), angle_list[hh], len(feedback_angle))
    #         tf_writer.add_scalar('mnist_exp/feedback_ratio_hidden' + str(hh), ratio_list[hh], len(feedback_ratio))
    feedback_angle, feedback_ratio = [0], [0]

    # Network info
    summary_message = "*** Network Summary *** \n"
    for n_layer, layer in enumerate(network.hidden_cells):
        summary_message += (f"Hidden Layer {n_layer}\n"
                        f"Vdecay {layer.vdecay}, Vth {layer.vth}, Grad_win {layer.grad_win}, Grad_amp {layer.grad_amp}, "
                        f"n_filters {layer.n_filters}, kernel_size {layer.kernel_size}, reduce_dim {layer.reduce_dim}, "
                        f"Learning rate: {layer.lr}, Weight decay {layer.weight_decay} \n")
        
        summary_message += print_stats(layer.forward_func.weight.cpu().detach().numpy(), f"FW Weight", False) + '\n'
        summary_message += print_stats(layer.forward_func_quant.weight.cpu().detach().numpy(), f"FW Weight Q", False) + '\n'
        summary_message += print_stats(layer.feedback_func.weight.cpu().detach().numpy(), f"FB Weight", False) + '\n'
        summary_message += print_stats(layer.feedback_func_quant.weight.cpu().detach().numpy(), f"FB Weight Q", False) + '\n'

        activation_list.append([[], [], []])
        hidden_weights_list.append([layer.forward_func.weight.data])
        hidden_weights_quant_list.append([layer.forward_func_quant.weight.data])
        deltas_list.append([torch.zeros_like(layer.forward_func.weight)])
        deltas_abs_list.append([torch.zeros_like(layer.forward_func.weight)])
        lr_lists.append([])

    summary_message += (f"Output Layer\n"
                        f"Neurons:  {network.output_cell.forward_func.weight.shape[0]}, {network.output_cell.forward_func.weight.shape[1]}, "
                        f"Vdecay {network.output_cell.vdecay}, Vth {network.output_cell.vth}, "
                        f"Grad_win {network.output_cell.grad_win}, Grad_amp {network.output_cell.grad_amp}, Feedback Vth {network.output_cell.feedback_th}, "
                        f"Learning rate: {network.output_cell.lr}, Weight decay {network.output_cell.weight_decay} \n")

    summary_message += print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(), "FW Weight", False) + '\n'
    summary_message += print_stats(network.output_cell.forward_func_quant.weight.cpu().detach().numpy(), "FW Weight Q", False) + '\n'

    output_weights_list.append(network.output_cell.forward_func.weight.data)
    output_weights_quant_list.append(network.output_cell.forward_func_quant.weight.data)
    deltas_list.append([torch.zeros_like(network.output_cell.forward_func.weight)])
    deltas_abs_list.append([torch.zeros_like(network.output_cell.forward_func.weight)])
    lr_lists.append([])

     # Write in the log all the training parameters
    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write("SNN training with BioGrad\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("spike_ts: %d\n" % spike_ts)
        # log_file.write("sleep_spike_ts: %d\n" % sleep_spike_ts)
        log_file.write("soft_error_step: %d\n" % soft_error_step)
        log_file.write("validation_size: %d\n" % validation_size)
        log_file.write("batch_size: %d\n" % batch_size)
        log_file.write("sleep_batch_size: %d\n" % sleep_batch_size)
        log_file.write("test_batch_size: %d\n" % test_batch_size)
        log_file.write("epoch: %d\n" % epoch)
        log_file.write("sleep_oja_power: %f\n" % sleep_oja_power)
        log_file.write("sleep_lr: %f\n" % sleep_lr)
        log_file.write("train_num: %d\n" % train_num)
        log_file.write("val_num: %d\n" % val_num)
        log_file.write("test_num: %d\n" % test_num)
        log_file.write("device: %s\n" % str(device))
        log_file.write("session_name: %s\n" % session_name)
        log_file.write("session_file_dir: %s\n" % session_file_dir)
        log_file.write("Decay term: False \n")
        log_file.write("Sleep phase: False \n")
        log_file.write("Bias: %s\n" % network.bias)
        log_file.write("Aligned weights: %s\n" % network.aligned_weights)
        log_file.write("Loss mode: %s\n" % network.loss_mode)
        log_file.write("Loss precision: %d\n" % loss_precision)
        log_file.write("Loss function: %s\n" % network.error_func.name)
        log_file.write("Num_bits: %d\n" % network.n_bits)
        log_file.write("Weight_init_bits: %d\n" % network.weight_init_bits)
        log_file.write("Low_precision_bits: %d\n" % network.low_precision_bits)
        log_file.write("Activation_bits: %d\n" % network.activation_bits)
        log_file.write("Train samples: %d, Test samples: %d\n" % (train_num, val_num))
        log_file.write(summary_message)
        
    # Print network info
    if train_stats:
        print(summary_message)

    # Start training
    spike_stats, batch_stats, plot_batch = stats
    ask_plot = plot_batch
    best_val_acc = 0
    # with torch.no_grad():
    with torch.inference_mode():
        for ee in range(epoch):
            # Training
            train_correct = 0
            train_start = time.time()
            # Epoch stats
            max_volts_epoch = torch.zeros(network.n_layers)
            train_activations = torch.zeros(network.n_layers)
            train_activations_fb = torch.zeros(2)

            # Batch stats
            h_w_batch, output_w_batch = [], []
            h_w_q_batch, output_w_q_batch = [], []
            deltas_batch = []
            deltas_abs_batch = []
            activation_batch = [[[]]]
            train_activation_fb_batch = [[], []]
            max_volts_batch = [[] for _ in range(network.n_layers)]

            for layer in network.hidden_cells:
                h_w_batch.append([copy.deepcopy(layer.forward_func.weight.data)])
                h_w_q_batch.append([copy.deepcopy(layer.forward_func_quant.weight.data)])

                deltas_batch.append([torch.zeros_like(layer.forward_func.weight.data)])
                deltas_abs_batch.append([torch.zeros_like(layer.forward_func.weight.data)])

                activation_batch.append([[]])
                train_activation_fb_batch.append([0, 0])

            output_w_batch.append(copy.deepcopy(network.output_cell.forward_func.weight.data))
            output_w_q_batch.append(copy.deepcopy(network.output_cell.forward_func_quant.weight.data))
            deltas_batch.append([torch.zeros_like(network.output_cell.forward_func.weight.data)])
            deltas_abs_batch.append([torch.zeros_like(network.output_cell.forward_func.weight.data)])

            batch_to_show = 999999
            ask_batch_stats = batch_stats
            
            # Batch training
            for n_batch, data in enumerate(train_dataloader):
                # if i == 2:
                #     spike_stats = True
                network.n_batch = n_batch
                for layer in network.hidden_cells:
                    layer.n_batch = n_batch
                network.output_cell.n_batch = n_batch

                img, labels = data
                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float() * loss_precision
                img, labels, labels_one_hot = img.to(device), labels.to(device), labels_one_hot.to(device)
                event_img = img_2_event_img(img, device, spike_ts)
                # event_img = quantize_image(event_img, network.n_bits)
                predict_label, hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step, max_volts, act, act_fb = network.train_online(
                    event_img, labels_one_hot, soft_error_step, loss_precision, spike_stats, float_mode=float_mode)
                deltas_dict = network.train_update_parameter(hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step, spike_stats)
                network.update_inference_weights()

                train_correct += ((predict_label == labels).sum().to("cpu")).item()
                train_activations += act
                train_activations_fb += act_fb

                for n_volt, max_volt in enumerate(max_volts):
                    max_volts_epoch[n_volt] = max(max_volts_epoch[n_volt], max_volt)

                # Put network to sleep for feedback training
                # network.sleep_feedback_update(sleep_batch_size, sleep_spike_ts, sleep_oja_power, sleep_lr)
               
                # Compute angle and ratio between feedback weight and forward weight after each update
                # if len(network.hidden_cells) > 0:
                #     angle_list, ratio_list = network.compute_feedback_angle_ratio()
                #     feedback_angle.append(angle_list)
                #     feedback_ratio.append(ratio_list)
                #     for hh in range(len(angle_list)):
                #         tf_writer.add_scalar('mnist_exp/feedback_angle_hidden' + str(hh), angle_list[hh],
                #                             len(feedback_angle))
                #         tf_writer.add_scalar('mnist_exp/feedback_ratio_hidden' + str(hh), ratio_list[hh],
                #                             len(feedback_ratio))
                # Show batch stats
                if batch_stats:
                    print(f"\n*** Batch {n_batch} ***")
                    formatted_volts_batch = " ".join([f"{x:.1f}" for x in max_volts])
                    formatted_volts_epoch = " ".join([f"{x:.1f}" for x in max_volts_epoch])
                    print(f"Max Volts batch: {formatted_volts_batch}")
                    print(f"Max Volts epoch: {formatted_volts_epoch}")
                    print(f"Network activations FW: {act}")
                    print(f"Network activations FB: {act_fb}")
                    print("Thresholds: " + str([layer.vth for layer in network.hidden_cells]) + " " + str(network.output_cell.vth))
                    
                    print("Deltas: ")
                    for key, delta in deltas_dict.items():
                        print_stats(delta.cpu().detach().numpy(), key)
                    
                    print("Weights: ")
                    for n_layer, layer in enumerate(network.hidden_cells):
                        print_stats(layer.forward_func.weight.cpu().detach().numpy(),                f"Hidden Layer {n_layer} FW   ")
                        print_stats(layer.forward_func_quant.weight.cpu().detach().numpy(),          f"Hidden Layer {n_layer} FW Q ")
                        print_stats(layer.feedback_func.weight.cpu().detach().numpy(),               f"Hidden Layer {n_layer} FB   ")
                        print_stats(layer.feedback_func_quant.weight.cpu().detach().numpy(),         f"Hidden Layer {n_layer} FB Q ")
                    print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(),       "Output Layer FW  ")
                    print_stats(network.output_cell.forward_func_quant.weight.cpu().detach().numpy(), "Output Layer FW Q")
                    
                    # if len(network.hidden_cells) > 0:
                    #     print(f"\nFeedback Angle {angle_list}, Feedback Ratio {ratio_list}")
                
                # Update batch stats - Only needed for batch plotting
                if plot_batch:
                    for n_layer, layer in enumerate(network.hidden_cells):
                        
                        delta_abs = layer.forward_func.weight.data - h_w_batch[n_layer][-1]
                        deltas_abs_batch[n_layer].append(delta_abs)
                        delta = 100 * delta_abs / (h_w_batch[n_layer][-1])
                        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

                        h_w_batch[n_layer].append(layer.forward_func.weight.data)
                        h_w_q_batch[n_layer].append(layer.forward_func_quant.weight.data)
                        deltas_batch[n_layer].append(delta)

                        activation_batch[n_layer][0].append(act[n_layer])
                        max_volts_batch[n_layer].append(max_volts[n_layer])

                    # Output layer                
                    delta_abs = network.output_cell.forward_func.weight.data - output_w_batch[-1]
                    deltas_abs_batch[-1].append(delta_abs)
                    delta = 100 * delta_abs / (output_w_batch[-1])
                    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                    deltas_batch[-1].append(delta)

                    output_w_batch.append(network.output_cell.forward_func.weight.data)
                    output_w_q_batch.append(network.output_cell.forward_func_quant.weight.data)
                    activation_batch[-1][0].append(act[-1])
                    max_volts_batch[-1].append(max_volts[-1])
                
                if n_batch == batch_to_show:
                    spike_stats = True
                    batch_stats = True
                    ask_batch_stats = True
                
                if ask_batch_stats:
                    stop = input("Continue execution with batch_stats? (y/go/fast/number/other): ")
                    if stop.lower() == 'y':
                        continue
                    elif stop.lower() == 'go':
                        batch_stats = False
                        spike_stats = False
                        ask_batch_stats = False
                    elif stop.lower() == 'fast' or stop.lower() == 'f':
                        batch_stats = False
                        spike_stats = False
                        ask_batch_stats = False
                        for layer in network.hidden_cells:
                            layer.writer_batch = False
                            layer.writer_spike = False
                        network.output_cell.writer_batch = False
                        network.output_cell.writer_spike = False
                    elif stop.isdigit():
                        ask_batch_stats = False
                        spike_stats = False
                        # batch_stats = False
                        batch_to_show = int(copy.deepcopy(stop))
                    else:
                        return
             
            # Update training stats
            train_stats_message = "\n\nEpoch: " + str(ee) + " \n"
            deltas_epoch = {}
            for n_layer, layer in enumerate(network.hidden_cells):
                # Weight stats
                train_stats_message += print_stats(layer.forward_func.weight.cpu().detach().numpy(),        f"Hidden {n_layer} FW   ", False) + "\n"
                train_stats_message += print_stats(layer.forward_func_quant.weight.cpu().detach().numpy(),  f"Hidden {n_layer} FW Q ", False) + "\n"
                train_stats_message += print_stats(layer.feedback_func.weight.cpu().detach().numpy(),       f"Hidden {n_layer} FB   ", False) + "\n"
                train_stats_message += print_stats(layer.feedback_func_quant.weight.cpu().detach().numpy(), f"Hidden {n_layer} FB Q ", False) + "\n"

                # Weight deltas
                delta_abs = layer.forward_func.weight.data - h_w_batch[n_layer][0]
                deltas_abs_list[n_layer].append(delta_abs)
                delta = 100 * delta_abs / (h_w_batch[n_layer][0])
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                deltas_epoch[layer.name] = (delta, delta_abs, layer.lr)
                hidden_weights_list[n_layer].append(layer.forward_func.weight.data)
                hidden_weights_quant_list[n_layer].append(layer.forward_func_quant.weight.data)
                deltas_list[n_layer].append(delta)
                max_volts_hist[ee][n_layer] = max_volts_epoch[n_layer]
                lr_lists[n_layer].append(layer.lr)

            # Output layer weights
            train_stats_message += print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(),       "Output   FW  ", False) + "\n"
            train_stats_message += print_stats(network.output_cell.forward_func_quant.weight.cpu().detach().numpy(), "Output   FW Q", False) + "\n"
            output_weights_list.append(network.output_cell.forward_func.weight.data)
            output_weights_quant_list.append(network.output_cell.forward_func_quant.weight.data)
            
            # Output layer deltas
            delta_abs = network.output_cell.forward_func.weight.data - output_w_batch[0]
            deltas_abs_list[-1].append(delta_abs)
            delta = 100 * delta_abs / (output_w_batch[0])
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            deltas_epoch[network.output_cell.name] = (delta, delta_abs, network.output_cell.lr)
            deltas_list[-1].append(delta)
            lr_lists[-1].append(network.output_cell.lr)

            train_stats_message += "Deltas: \n"
            for key, delta in deltas_epoch.items():
                train_stats_message += f"{key} Learning rate: {delta[2]} \n"
                train_stats_message += print_stats(delta[0].cpu().detach().numpy().flatten(), f"{key} Delta %  ", False) + "\n"
                train_stats_message += print_stats(delta[1].cpu().detach().numpy().flatten(), f"{key} Delta Abs", False) + "\n"

            max_volts_hist[ee][-1] = max_volts_epoch[-1]
            thresholds = [layer.vth for layer in network.hidden_cells] + [network.output_cell.vth]
            train_stats_message += "Thresholds: " + str(thresholds) + "\n"
            train_stats_message += "Max Volts: " + " ".join([f"{x:.1f}" for x in max_volts_epoch]) + "\n"

            # if len(network.hidden_cells) > 0:
            #         train_stats_message += f"Feedback Angle {angle_list}, Feedback Ratio {ratio_list} \n"

            train_end = time.time()
            train_accuracy_list.append(train_correct / train_num)
            train_activations /= len(train_dataloader)
            train_activations_fb /= len(train_dataloader)
            train_activation_fb_list[0].append(train_activations_fb[0])
            train_activation_fb_list[1].append(train_activations_fb[1])
            
            performance_message = "Epoch %d Training Accuracy %.4f" % (ee, train_accuracy_list[-1])

            # Update learning rate
            if lr_scheduler != 0:
                lr_coef = lr_scheduler.step()
                network.update_lr(lr_coef)

            # Validation
            val_correct = 0
            val_activations = torch.zeros(network.n_layers)
            val_start = time.time()
            for data in val_dataloader:
                img, labels = data
                img, labels = img.to(device), labels.to(device)
                event_img = img_2_event_img(img, device, spike_ts)
                predict_label, act = network.test(event_img)
                val_activations += act
                val_correct += ((predict_label == labels).sum().to("cpu")).item()
            val_end = time.time()
            val_activations /= len(val_dataloader)
            val_accuracy_list.append(val_correct / val_num)
            performance_message += " Validate Accuracy %.4f" % val_accuracy_list[-1]

            # Testing
            test_correct = 0
            test_activations = torch.zeros(network.n_layers)
            test_start = time.time()
            for data in test_dataloader:
                img, labels = data
                img, labels = img.to(device), labels.to(device)
                event_img = img_2_event_img(img, device, spike_ts)
                predict_label, act = network.test(event_img)
                test_activations += act
                test_correct += ((predict_label == labels).sum().to("cpu")).item()
            test_end = time.time()
            test_activations /= len(test_dataloader)
            test_accuracy_list.append(test_correct / test_num)
            performance_message += " Test Accuracy %.4f " % (test_accuracy_list[-1])
            
            for n_layer in range(network.n_layers):
                activation_list[n_layer][0].append(train_activations[n_layer])
                activation_list[n_layer][1].append(val_activations[n_layer])
                activation_list[n_layer][2].append(test_activations[n_layer])
            
            # Tensorboard
            if writer_epoch:
                tf_writer.add_scalar('Performance/train_accuracy', train_accuracy_list[-1], ee)
                tf_writer.add_scalar('Performance/val_accuracy', val_accuracy_list[-1], ee)
                tf_writer.add_scalar('Performance/test_accuracy', test_accuracy_list[-1], ee)

                for n_layer, layer in enumerate(network.hidden_cells):
                    tf_writer.add_scalar(f"Activations/layer_{n_layer}_train", train_activations[n_layer].item(), ee)
                    tf_writer.add_scalar(f"Activations/layer_{n_layer}_val", val_activations[n_layer].item(), ee)
                    tf_writer.add_scalar(f"Activations/layer_{n_layer}_test", test_activations[n_layer].item(), ee)
                    tf_writer.add_scalar(f"Max_volts/layer_{n_layer}", max_volts_epoch[n_layer].item(), ee)
                    tf_writer.add_scalar(f"Learning_rate/layer_{n_layer}_lr", layer.lr, ee)

                    tf_writer.add_histogram(f"Weights_{n_layer}/forward_weights", layer.forward_func.weight, ee)
                    tf_writer.add_histogram(f"Weights_{n_layer}/forward_weights_quant", layer.forward_func_quant.weight, ee)
                    tf_writer.add_histogram(f"Weights_{n_layer}/feedback_weights", layer.feedback_func.weight, ee)
                    tf_writer.add_histogram(f"Weights_{n_layer}/feedback_weights_quant", layer.feedback_func_quant.weight, ee)

                tf_writer.add_scalar("Activations/output_layer_train", train_activations[-1].item(), ee)
                tf_writer.add_scalar("Activations/output_layer_val", val_activations[-1].item(), ee)
                tf_writer.add_scalar("Max_volts/output_layer", max_volts_epoch[-1].item(), ee)
                tf_writer.add_scalar("Learning_rate/output_layer_lr", network.output_cell.lr, ee)

                tf_writer.add_histogram("Weights_Out/forward_weights", network.output_cell.forward_func.weight, ee)
                tf_writer.add_histogram("Weights_Out/forward_weights_quant", network.output_cell.forward_func_quant.weight, ee)
                
            time_message = "Training Time: %.1f Val Time: %.1f Test Time: %.1f" % (
            train_end - train_start, val_end - val_start, test_end - test_start)
            train_stats_message += f"Train Activations FB: {train_activations_fb} \n"
            train_stats_message += f"Train Activations: {train_activations} \n"
            train_stats_message += f"Val Activations: {val_activations} \n"
            train_stats_message += f"Test Activations: {test_activations} \n"
            train_stats_message += performance_message
            train_stats_message += time_message
            if train_stats:
                print(train_stats_message)

            with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
                log_file.write(train_stats_message + '\n')

            # Save model if best validation accuracy
            best_val_acc = max(best_val_acc, val_accuracy_list[-1])
            if val_accuracy_list[-1] == best_val_acc:
                message = f"New best validation accuracy at epoch {ee}: {best_val_acc}"
                best_stats = (feedback_angle[-1], feedback_ratio[-1], train_activations_fb, train_activations, val_activations, test_activations)
                if train_stats:
                    print(message)
                with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
                    log_file.write(message)
                # pickle.dump(network, open(session_file_dir + "/model_best.p", "wb+"))
                pickle.dump(max_volts_epoch, open(session_file_dir + "/max_volts_best.p", "wb+"))
            
            # Plot epoch stats per batch
            if not plot_batch:
                continue
            plots_folder = "./dynamic_debug/"
            if not os.path.exists(plots_folder):
                os.makedirs(plots_folder)
            title = f"Epoch {ee} - Training acc {train_accuracy_list[-1]:.4f} - Test acc {val_accuracy_list[-1]:.4f} \n {session_name}"
            plot_training.boxplot_weights(h_w_batch, output_w_batch,
                                            h_w_q_batch, output_w_q_batch,
                                            deltas_batch, deltas_abs_batch,
                                            title, plots_folder, best_epoch=0)
            plot_training.plot_max_volts(max_volts_batch, title, plots_folder, best_epoch=0)
            plot_training.activations_plot(activation_batch, title, plots_folder, best_epoch=0)

            if ask_plot:
                print("Images saved at " + plots_folder)
                stop = input("Continue batch plotting? (y/go/auto/other): ")
                if stop.lower() == 'y':
                    continue
                elif stop.lower() == 'go':
                    plot_batch = False
                    ask_plot = False
                    for layer in network.hidden_cells:
                        layer.writer_batch = False
                        layer.writer_spike = False
                    network.output_cell.writer_batch = False
                    network.output_cell.writer_spike = False
                elif stop.lower() == 'auto':
                    plot_batch = True
                    ask_plot = False
                else:
                    return

    best_epoch = np.argmax(val_accuracy_list)
    best_epoch_message = f"End Training from session {session_name} \n {session_file_dir}\n\n"
    best_epoch_message += f"Best epoch: {best_epoch} \n"
    best_epoch_message += f"Validation Accuracy: {best_val_acc} \n"
    best_epoch_message += f"Feedback Angle: {best_stats[0]} \n"
    best_epoch_message += f"Feedback Ratio: {best_stats[1]} \n"
    best_epoch_message += f"Train Activations FB: {best_stats[2]} \n"
    best_epoch_message += f"Train Activations: {best_stats[3]} \n"
    best_epoch_message += f"Val Activations: {best_stats[4]} \n"
    best_epoch_message += f"Test Activations: {best_stats[5]} \n \n"

    if train_stats:
        print(best_epoch_message)

    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write(best_epoch_message)

    tf_writer.add_hparams(hparam_dict=hparams, metric_dict={"Performance/train_accuracy": max(train_accuracy_list),
                                                            "Performance/val_accuracy": max(val_accuracy_list),
                                                            "Performance/test_accuracy": max(test_accuracy_list)})
    tf_writer.close()

    if network.loss_mode == 'direct':
        train_activation_fb_list = []
    plot_training.plot_training(train_accuracy_list, val_accuracy_list, test_accuracy_list, session_name, session_file_dir, best_epoch=best_epoch)
    plot_training.plot_fb(feedback_angle, feedback_ratio, session_name, session_file_dir, best_epoch=best_epoch)
    plot_training.boxplot_weights(hidden_weights_list, output_weights_list, 
                                  hidden_weights_quant_list, output_weights_quant_list,
                                  deltas_list, deltas_abs_list,
                                  session_name, session_file_dir, best_epoch=best_epoch+1)
    plot_training.activations_plot(activation_list, session_name, session_file_dir, feedback_act=train_activation_fb_list, best_epoch=best_epoch)
    plot_training.plot_max_volts(max_volts_hist, session_name, session_file_dir, best_epoch=best_epoch)
    plot_training.plot_lr(lr_lists, session_name, session_file_dir, best_epoch=best_epoch)

    return train_accuracy_list, val_accuracy_list, test_accuracy_list, feedback_angle, feedback_ratio, best_stats