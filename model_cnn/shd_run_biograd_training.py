import pickle
import torch
import numpy as np
from shd_exp.config import Config
from shd_exp.training_biograd import biograd_snn_training
from biograd_snn.network_w_biograd import BioGradNetworkWithSleep
from biograd_snn.online_error_functions import cross_entropy_loss_error_function, l2_loss

# Define SNN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dim = 20

# soft_error_start = 5
# spike_ts = 20
# sleep_spike_ts = 50

# Define Training parameters
train_batch_size = 256
sleep_batch_size = 128
test_batch_size = 512
epoch = 5
save_epoch = 1
# lr = 1
sleep_oja_power = 2.0
sleep_lr = 1.0e-4 / 3.

num_bits = 16
weight_init_bits = 16
low_precision_bits = 8

float_mode = False # False: Int, True: Float
aligned_weights = False # True: Aligned feedback weights, False: Transposed feedforward weights
loss_mode = 'direct'
# 'feedback': traditional (accumulated loss over all time steps)
# 'final': loss only on final step
# 'direct': loss only on final step not processed in OutputLayer.feedbackstep
softmax = True # True: Use softmax, False: Use simplified loss function -> HAS TO BE CHANGED MANUALLY (only for float_mode)

spike_stats = False
batch_stats = False

if float_mode:
    loss_precision = 1

    out_layer_dict = {'Vdecay': 0.5, 'Vth': 0.3, 'Grad_win': 0.6, 'Grad_amp': 1., 'Fb_th': 1., 
                      'lr': 3e-3, 'Weight_decay': 0}
    hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': 2., 'Grad_win': 4., 'Grad_amp': 1., 'n_filters': 32, 'kernel_size': 5, 'reduce_dim': True,
                          'lr': 3e-3, 'Weight_decay': 0}
    # hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': 2., 'Grad_win': 4., 'Grad_amp': 1., 'n_filters': 32, 'filter_size': 5,
    #                       'lr': 3e-3, 'Weight_decay': 0}

else:
    loss_precision = 32

    factor = 16
    out_layer_dict = {'Vdecay': 0.5, 'Vth': factor*4000, 'Grad_win': factor*8000, 'Grad_amp': 1., 'Fb_th': 1., 
                      'lr': 2**-4, 'Weight_decay': 1/2**20}
    hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': factor*4000., 'Grad_win': factor*8000., 'Grad_amp': 1., 'n_filters': 32, 'kernel_size': 5, 'reduce_dim': True,
                          'lr': 1/2**32, 'Weight_decay': 1/2**12}

# Define network architecture
hidden_dim_list = [256]
snn_param = {'out_layer': out_layer_dict, 'hidden_layer': [hidden_layer_dict1]}

data_params = {'data_type': 'frame', 'duration': None, 'n_bins': 4, 'frames': 10, 'split_by': 'time'}
in_shape = 700 // data_params['n_bins']

# Define save folder and tensorboard folder
experiment_group = "shd/sanity/"
tf_folder = "./runs/" + experiment_group
save_folder = "./save_models/" + experiment_group

# Define SNN and start training
seed_list = [65941348, 55674239, 3164977, 6334270]
seed_list = [100364571]
with torch.no_grad():
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        session_name = f"shd_biograd_cnn_quant_{str(len(snn_param['hidden_layer']))}_layers_"
        session_name += "float_" if float_mode else "int_"
        if not float_mode:
            session_name += f"{num_bits}_{weight_init_bits}_{low_precision_bits}_"
        session_name += "aligned_" if aligned_weights else "transposed_"
        session_name += "loss_" + loss_mode + "_"
        if data_params['duration'] is None:
            session_name += f"frames_{data_params['frames']}_{data_params['split_by']}_"
        else:
            session_name += f"duration_{data_params['duration']}_"
        # session_name += "dynamicNO_"

        if float_mode:
            session_name += "softmax_" if softmax else "simple_"
        session_name += str(seed)

        online_snn = BioGradNetworkWithSleep(in_shape, out_dim,
                                            snn_param, l2_loss, device, float_mode=float_mode,
                                            aligned_weights = aligned_weights, loss_mode=loss_mode,
                                            num_bits=num_bits, weight_init_bits=weight_init_bits, low_precision_bits=low_precision_bits)
        print("Loss mode: ", loss_mode)
        print("Float mode: ", float_mode)
        print("Aligned weights: ", aligned_weights)
        
        train_acc, val_acc, test_acc, fb_angle, fb_ratio, _ = biograd_snn_training(
            online_snn, device, session_name, data_params=data_params,
            batch_size=train_batch_size, sleep_batch_size=sleep_batch_size,
            test_batch_size=test_batch_size, epoch=epoch, save_epoch=save_epoch,
            sleep_oja_power=sleep_oja_power, sleep_lr=sleep_lr, loss_precision=loss_precision, 
            stats=(spike_stats, batch_stats), float_mode=float_mode, tf_folder=tf_folder, save_folder=save_folder)

        pickle.dump(train_acc, open(save_folder + session_name + "/train_accuracy_list.p", "wb+"))
        pickle.dump(val_acc, open(save_folder + session_name + "/val_accuracy_list.p", "wb+"))
        pickle.dump(test_acc, open(save_folder + session_name + "/test_accuracy_list.p", "wb+"))
        pickle.dump(fb_angle, open(save_folder + session_name + "/feedback_angle_list.p", "wb+"))
        pickle.dump(fb_ratio, open(save_folder + session_name + "/feedback_ratio_list.p", "wb+"))
