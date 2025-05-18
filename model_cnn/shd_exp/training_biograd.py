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
# from mnist_exp.utility import img_2_event_img
from quantization import quantize_image, print_stats
import plot_training
from shd_exp.datasets import BinnedSpikingHeidelbergDigits
from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets import pad_sequence_collate


def biograd_snn_training(network, device, session_name,
                         validation_size=10000, batch_size=128, sleep_batch_size=128,
                         test_batch_size=128, epoch=100,
                         save_epoch=1,
                         sleep_oja_power=2, sleep_lr=1e-3, loss_precision=32, 
                         stats=False, float_mode=False, train_stats=True, data_params=0, tf_folder="", save_folder=""):
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

    # train_dataloader, val_dataloader = SHD_dataloaders(config)
    if data_params == 0:
        data_type = 'frame'
        duration = None
        frames = 20
        split_by = 'number'
        n_bins = 1
    else:
        data_type = data_params['data_type']
        duration = data_params['duration']
        frames = data_params['frames']
        split_by = data_params['split_by']
        n_bins = data_params['n_bins']
    
    if duration is not None:
        frames, split_by = None, None
        print(f"Duration: {duration}, Frames: {frames}, Split by: {split_by}")

    # train_dataset = SpikingHeidelbergDigits('data/SHD', train=True, data_type=data_type, frames_number=frames, split_by=split_by)
    # test_dataset = SpikingHeidelbergDigits('data/SHD', train=False, data_type=data_type, frames_number=frames, split_by=split_by)
    train_dataset = BinnedSpikingHeidelbergDigits('data/SHD', n_bins=n_bins, train=True, data_type=data_type, frames_number=frames, split_by=split_by, duration=duration)
    test_dataset = BinnedSpikingHeidelbergDigits('data/SHD', n_bins=n_bins,  train=False, data_type=data_type, frames_number=frames, split_by=split_by, duration=duration)
    train_dataloader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=test_batch_size, shuffle=True, num_workers=0)

    if network.loss_mode != 'feedback':
        soft_error_step = 1
    else:
        print("Feedback loss mode: soft_error_step NEEDS TO BE UDPATED")

    # Number of samples in train, validation, and test dataset
    train_num = train_dataset.length
    val_num = test_dataset.length
    if train_stats:
        print(f"Train samples: {train_num}, Test samples: {val_num}")

    # Quantize weights
    if not float_mode:
        network.quantize_network()
    network.prepare_inference_weights(train_stats)
    
    # Lists for training stats
    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []
    hidden_weights_list, output_weights_list = [], []
    deltas_list = []
    activation_list = [[[], [], []]]
    train_activation_fb_list = [[], []]
    max_volts_hist = torch.zeros(epoch, network.n_layers)

    # Define tensorboard
    log_dir = os.path.join(tf_folder, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + session_name)
    tf_writer = SummaryWriter(log_dir)

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

        activation_list.append([[], []])
        hidden_weights_list.append([layer.forward_func.weight.cpu().detach().numpy().flatten()])
        deltas_list.append([np.zeros_like(layer.forward_func.weight.cpu().detach().numpy())])

    summary_message += (f"Output Layer\n"
                        f"Neurons:  {network.output_cell.forward_func.weight.shape[0]}, {network.output_cell.forward_func.weight.shape[1]}, "
                        f"Vdecay {network.output_cell.vdecay}, Vth {network.output_cell.vth}, "
                        f"Grad_win {network.output_cell.grad_win}, Grad_amp {network.output_cell.grad_amp}, Feedback Vth {network.output_cell.feedback_th}, "
                        f"Learning rate: {network.output_cell.lr}, Weight decay {network.output_cell.weight_decay} \n")

    summary_message += print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(), "FW Weight", False) + '\n'
    summary_message += print_stats(network.output_cell.forward_func_quant.weight.cpu().detach().numpy(), "FW Weight Q", False) + '\n'

    output_weights_list.append(network.output_cell.forward_func.weight.cpu().detach().numpy().flatten())
    deltas_list.append([np.zeros_like(network.output_cell.forward_func.weight.cpu().detach().numpy())])

    # Write in the log all the training parameters
    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write("SNN training with BioGrad\n")
        # log_file.write("spike_ts: %d\n" % spike_ts)
        # log_file.write("sleep_spike_ts: %d\n" % sleep_spike_ts)
        # log_file.write("soft_error_step: %d\n" % soft_error_step)
        log_file.write("validation_size: %d\n" % validation_size)
        log_file.write("batch_size: %d\n" % batch_size)
        log_file.write("sleep_batch_size: %d\n" % sleep_batch_size)
        log_file.write("test_batch_size: %d\n" % test_batch_size)
        log_file.write("epoch: %d\n" % epoch)
        log_file.write("sleep_oja_power: %f\n" % sleep_oja_power)
        log_file.write("sleep_lr: %f\n" % sleep_lr)
        log_file.write("train_num: %d\n" % train_num)
        log_file.write("val_num: %d\n" % val_num)
        log_file.write("device: %s\n" % str(device))
        log_file.write("session_name: %s\n" % session_name)
        log_file.write("session_file_dir: %s\n" % session_file_dir)
        log_file.write("Decay term: False \n")
        log_file.write("Sleep phase: False \n")
        log_file.write("Bias: False\n")
        log_file.write("Loss precision: %d\n" % loss_precision)
        log_file.write("Data params: %s\n" % str(data_params))
        log_file.write("Train samples: %d, Test samples: %d\n" % (train_num, val_num))
        log_file.write(summary_message)
        
    # Print network info
    if train_stats:
        print(summary_message)

    # Start training
    spike_stats, batch_stats = stats
    best_val_acc = 0
    # with torch.no_grad():
    with torch.inference_mode():
        for ee in range(epoch):
            # Training
            train_correct = 0
            train_start = time.time()
            # if ee == 5:
            #     stats = False
            # if train_stats:
            weights_start = []
            for layer in network.hidden_cells:
                weights_start.append(copy.deepcopy(layer.forward_func.weight))
            weights_start.append(copy.deepcopy(network.output_cell.forward_func.weight))
            
            max_volts_epoch = torch.zeros(network.n_layers)
            train_activations = torch.zeros(network.n_layers)
            train_activations_fb = torch.zeros(2)
            for n_batch, (x, labels, _) in enumerate(train_dataloader):
                # if i == 2:
                #     spike_stats = True
                labels_one_hot = nn.functional.one_hot(labels, num_classes=network.output_dim).float() * loss_precision
                x = x.permute(0,2,1).float().to(device)  #(time, batch, neurons)
                labels, labels_one_hot = labels.to(device), labels_one_hot.to(device)

                # event_img = img_2_event_img(img, device, spike_ts)
                # event_img = quantize_image(event_img, network.n_bits)
                predict_label, hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step, max_volts, act, act_fb = network.train_online(
                    x, labels_one_hot, soft_error_step, loss_precision, spike_stats, float_mode=float_mode)
                deltas = network.train_update_parameter(hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step, spike_stats)
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
                
                if batch_stats:
                    print(f"\n*** Batch {n_batch} ***")
                    formatted_volts_batch = " ".join([f"{x:.1f}" for x in max_volts])
                    formatted_volts_epoch = " ".join([f"{x:.1f}" for x in max_volts_epoch])
                    print(f"Max Volts batch: {formatted_volts_batch}")
                    print(f"Max Volts epoch: {formatted_volts_epoch}")
                    print(f"Network activations FW: {act}")
                    print(f"Network activations FB: {act_fb}")
                    print("Thresholds: " + str([layer.vth for layer in network.hidden_cells]) + " " + str(network.output_cell.vth))
                    
                    if len(deltas) > 1:
                        print("Deltas: ")
                    for n_delta, delta in enumerate(deltas):
                        print_stats(delta.cpu().detach().numpy(), f"Layer {n_delta} Delta %")
                    
                    if len(deltas) > 1:
                        print("Weights: ")
                    for n_layer, layer in enumerate(network.hidden_cells):
                        print_stats(layer.forward_func.weight.cpu().detach().numpy(), f"Hidden Layer {n_layer} FW Weight  ")
                        print_stats(layer.forward_func_quant.weight.cpu().detach().numpy(), f"Hidden Layer {n_layer} FW Weight Q")
                        print_stats(layer.feedback_func.weight.cpu().detach().numpy(), f"Hidden Layer {n_layer} FB Weight  ")
                        print_stats(layer.feedback_func_quant.weight.cpu().detach().numpy(), f"Hidden Layer {n_layer} FB Weight Q")
                    print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(), "Output Layer 1 FW Weight  ")
                    print_stats(network.output_cell.forward_func_quant.weight.cpu().detach().numpy(), "Output Layer 1 FW Weight Q")
                    
                    # if len(network.hidden_cells) > 0:
                    #     print(f"\nFeedback Angle {angle_list}, Feedback Ratio {ratio_list}")

                    stop = input("Continue execution? (y/n): ")
                    if stop.lower() != 'y':
                        return
            
            train_stats_message = "\n\nEpoch: " + str(ee) + " \n"
            deltas_epoch = {}
            for n_layer, layer in enumerate(network.hidden_cells):
                # Weight stats
                train_stats_message += print_stats(layer.forward_func.weight.cpu().detach().numpy(), f"Hidden {n_layer} FW Weight", False) + "\n"
                train_stats_message += print_stats(layer.forward_func_quant.weight.cpu().detach().numpy(), f"Hidden {n_layer} FW Weight Q", False) + "\n"
                train_stats_message += print_stats(layer.feedback_func.weight.cpu().detach().numpy(), f"Hidden {n_layer} FB Weight", False) + "\n"
                train_stats_message += print_stats(layer.feedback_func_quant.weight.cpu().detach().numpy(), f"Hidden {n_layer} FB Weight Q", False) + "\n"

                # Weight deltas
                delta = 100 * (layer.forward_func.weight - weights_start[n_layer])/ weights_start[n_layer]
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                deltas_epoch[layer.name] = (delta, layer.lr)
                hidden_weights_list[n_layer].append(layer.forward_func.weight.cpu().detach().numpy().flatten())
                deltas_list[n_layer].append(delta.cpu().detach().numpy())

                max_volts_hist[ee][n_layer] = max_volts_epoch[n_layer]

            # Output layer weights
            train_stats_message += print_stats(network.output_cell.forward_func.weight.cpu().detach().numpy(), "Output   FW Weight", False) + "\n"
            train_stats_message += print_stats(network.output_cell.forward_func_quant.weight.cpu().detach().numpy(), "Output   FW Weight Q", False) + "\n"
            output_weights_list.append(network.output_cell.forward_func.weight.cpu().detach().numpy().flatten())
            
            # Output layer deltas
            delta = 100 * (network.output_cell.forward_func.weight - weights_start[-1])/ weights_start[-1]
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            deltas_epoch[network.output_cell.name] = (delta, network.output_cell.lr)
            deltas_list[-1].append(delta.cpu().detach().numpy())

            train_stats_message += "Deltas: \n"
            for key, delta in deltas_epoch.items():
                train_stats_message += f"{key} Learning rate: {delta[1]} \n"
                train_stats_message += print_stats(delta[0].cpu().detach().numpy(), f"{key} Delta %", False) + "\n"

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
            
            tf_writer.add_scalar('train_accuracy', train_accuracy_list[-1], ee)
            performance_message = "Epoch %d Training Accuracy %.4f" % (ee, train_accuracy_list[-1])

            # Test
            val_correct = 0
            val_activations = torch.zeros(network.n_layers)
            val_start = time.time()
            for data in test_dataloader:
                img, labels, _ = data
                img, labels = img.to(device), labels.to(device)
                img = img.permute(0,2,1).float().to(device)
                # event_img = img_2_event_img(img, device, spike_ts)
                predict_label, act = network.test(img, spike_stats)
                val_activations += act
                val_correct += ((predict_label == labels).sum().to("cpu")).item()
            val_end = time.time()
            val_activations /= len(test_dataloader)
            val_accuracy_list.append(val_correct / val_num)
            tf_writer.add_scalar('val_accuracy', val_accuracy_list[-1], ee)
            performance_message += " Test Accuracy %.4f " % val_accuracy_list[-1]
            
            for n_layer in range(network.n_layers):
                activation_list[n_layer][0].append(train_activations[n_layer])
                activation_list[n_layer][1].append(val_activations[n_layer])
                
            time_message = "Training Time: %.1f Test Time: %.1f " % (
            train_end - train_start, val_end - val_start)
            train_stats_message += f"Train Activations FB: {train_activations_fb} \n"
            train_stats_message += f"Train Activations: {train_activations} \n"
            train_stats_message += f"Val Activations: {val_activations} \n"
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
                best_stats = (feedback_angle[-1], feedback_ratio[-1], train_activations_fb, train_activations, val_activations, torch.zeros(network.n_layers))
                if train_stats:
                    print(message)
                with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
                    log_file.write(message)
                pickle.dump(network, open(session_file_dir + "/model_best.p", "wb+"))
                pickle.dump(max_volts_epoch, open(session_file_dir + "/max_volts_best.p", "wb+"))
            
            # stop = input("Continue execution? (y/n): ")
            # if stop.lower() != 'y':
            #     return

    best_epoch = np.argmax(val_accuracy_list)
    best_epoch_message = f"End Training from session {session_name} \n {session_file_dir}\n\n"
    best_epoch_message += f"Best epoch: {best_epoch} \n"
    best_epoch_message += f"Validation Accuracy: {best_val_acc} \n"
    best_epoch_message += f"Feedback Angle: {best_stats[0]} \n"
    best_epoch_message += f"Feedback Ratio: {best_stats[1]} \n"
    best_epoch_message += f"Train Activations FB: {best_stats[2]} \n"
    best_epoch_message += f"Train Activations: {best_stats[3]} \n"
    best_epoch_message += f"Val Activations: {best_stats[4]} \n \n"

    if train_stats:
        print(best_epoch_message)

    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write(best_epoch_message)

    plot_training.plot_training(train_accuracy_list, val_accuracy_list, test_accuracy_list, session_name, session_file_dir, best_epoch=best_epoch)
    plot_training.plot_fb(feedback_angle, feedback_ratio, session_name, session_file_dir, best_epoch=best_epoch)
    plot_training.boxplot_weights(hidden_weights_list, output_weights_list, deltas_list, 
                                  session_name, session_file_dir, best_epoch=best_epoch+1)
    plot_training.activations_plot(activation_list, train_activation_fb_list, session_name, session_file_dir, best_epoch=best_epoch)
    plot_training.plot_max_volts(max_volts_hist, session_name, session_file_dir, best_epoch=best_epoch)

    return train_accuracy_list, val_accuracy_list, test_accuracy_list, feedback_angle, feedback_ratio, best_stats