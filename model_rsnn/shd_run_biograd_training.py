import pickle
import torch
import numpy as np
from shd_exp.config import Config
from shd_exp.training_biograd import biograd_snn_training
from biograd_snn.network_w_biograd import BioGradNetworkWithSleep
from biograd_snn.online_error_functions import cross_entropy_loss_error_function, LossFunction
from lr_schedulers import ExponentialDecay, CosineAnnealingWarmRestarts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# soft_error_start = 5
# spike_ts = 20
# sleep_spike_ts = 50

# Define Training parameters
train_batch_size = 128
sleep_batch_size = 128
test_batch_size = 256
epoch = 2
sleep_oja_power = 2.0
sleep_lr = 1.0e-4 / 3.

num_bits = 16
weight_init_bits = 16
low_precision_bits = 8
activation_bits = 16

float_mode = False # False: Int, True: Float
bias = False
aligned_weights = False # True: Aligned feedback weights, False: Transposed feedforward weights
loss_mode = 'direct'
# 'feedback': traditional (accumulated loss over all time steps)
# 'final': loss only on final step
# 'direct': loss only on final step not processed in OutputLayer.feedbackstep
softmax = True # True: Use softmax, False: Use simplified loss function -> HAS TO BE CHANGED MANUALLY (only for float_mode)

spike_stats = False
batch_stats = False
plot_batch = False
writer_spike, writer_batch, writer_epoch = False, False, True

# Define network architecture
if float_mode:
    out_layer_dict = {'Vdecay': 0.5, 'Vth': 1., 'Grad_win': 1., 'Grad_amp': 1., 'Fb_th': 1., 'lr': 3e-3, 
                      'Weight_decay': 0}
    hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': 2., 'Grad_win': 4., 'Grad_amp': 1., 'lr': 1e-3, 
                         'Weight_decay': 0, 'lr_rec': 0., 'Vdecay_rec': 0.5, 'fw_factor': 1., 'rec_factor': 1.}
    # hidden_layer_dict2 = {'Vdecay': 0.5, 'Vth': 1., 'Grad_win': 2., 'Grad_amp': 1., 'lr': 1e-2, 
    #                      'Weight_decay': 0, 'lr_rec': 1e-2, 'Vdecay_rec': 0.005}
    
    loss_precision = 1

else:
    factor = 1.
    out_layer_dict = {'Vdecay': 0.5, 'Vth': factor*1600., 'Grad_win': factor*1600., 'Grad_amp': 1., 'Fb_th': 1., 'lr': 2**-1, 
                      'Weight_decay': 2**-12}
    hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': factor*2000., 'Grad_win': factor*4000., 'Grad_amp': 1., 'lr': 2**-12,
                         'Weight_decay': 2**-12, 'lr_rec': 0., 'Vdecay_rec': 0.5, 'fw_factor': 1., 'rec_factor': 1.}
    
    loss_precision = 128

# Define network architecture
gradient_clip = 256
hidden_dim_list = [256]
snn_param = {'out_layer': out_layer_dict, 'hidden_layer': [hidden_layer_dict1]}

data_params = {'data_type': 'frame', 'duration': None, 'n_bins': 4, 'frames': 10, 'split_by': 'time'}
in_shape = 700 // data_params['n_bins']
out_dim = 20

# Loss function
loss_fn, loss_parameters = 'normal', None
# loss_fn, loss_parameters = 'random', (-1, 2)
# loss_fn, loss_parameters = 'l2_loss_positive', None

loss_function = LossFunction(loss_fn, parameters=loss_parameters)

# Learning rate scheduler
# lr_scheduler = ExponentialDecay(initial_lr=1, decay_factor=1.1)
# lr_scheduler = CosineAnnealingWarmRestarts(initial_lr=1, T_0=10, T_mult=2)
lr_scheduler = 0

# Define save folder and tensorboard folder
experiment_group = "shd/activity_coefs/"
tf_folder = "./runs/" + experiment_group
save_folder = "./save_models/" + experiment_group
tf_name = None
# tf_name = "rec_spike"

# Define seeds and start training
seed_list = [65941348, 55674239, 3164977, 6334270]
seed_list = [100364571]
with torch.no_grad():
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        session_name = f"shd_biograd_rsnn_{str(len(snn_param['hidden_layer']))}_layers_"
        session_name += "float_" if float_mode else "int_"
        session_name += "bias_" if bias else ""
        if not float_mode:
            session_name += f"{num_bits}_{weight_init_bits}_{low_precision_bits}_"
        session_name += "aligned_" if aligned_weights else "transposed_"
        session_name += "loss_" + loss_mode + "_"
        if data_params['duration'] is None:
            session_name += f"frames_{data_params['frames']}_{data_params['split_by']}_"
        else:
            session_name += f"duration_{data_params['duration']}_"

        if float_mode:
            session_name += "softmax_" if softmax else "simple_"
        session_name += f"spike_"
        session_name += str(seed)

        online_snn = BioGradNetworkWithSleep(in_shape, out_dim, hidden_dim_list,
                                            snn_param, loss_function, device, float_mode=float_mode,
                                            aligned_weights = aligned_weights, loss_mode=loss_mode, bias=bias, gradient_clip=gradient_clip,
                                            num_bits=num_bits, weight_init_bits=weight_init_bits, low_precision_bits=low_precision_bits, activation_bits=activation_bits)
        print("Loss mode: ", loss_mode)
        print("Float mode: ", float_mode)
        print("Bias: ", bias)
        print("Aligned weights: ", aligned_weights)
        
        train_acc, val_acc, test_acc, fb_angle, fb_ratio, _ = biograd_snn_training(
            online_snn, device, session_name, data_params=data_params,
            batch_size=train_batch_size, sleep_batch_size=sleep_batch_size,
            test_batch_size=test_batch_size, epoch=epoch,
            sleep_oja_power=sleep_oja_power, sleep_lr=sleep_lr, loss_precision=loss_precision, 
            stats=(spike_stats, batch_stats, plot_batch), float_mode=float_mode, lr_scheduler=lr_scheduler, 
            writer=(writer_spike, writer_batch, writer_epoch), tf_folder=tf_folder, save_folder=save_folder, tf_name=tf_name)

        pickle.dump(train_acc, open(save_folder + session_name + "/train_accuracy_list.p", "wb+"))
        pickle.dump(val_acc, open(save_folder + session_name + "/val_accuracy_list.p", "wb+"))
        pickle.dump(test_acc, open(save_folder + session_name + "/test_accuracy_list.p", "wb+"))
        pickle.dump(fb_angle, open(save_folder + session_name + "/feedback_angle_list.p", "wb+"))
        pickle.dump(fb_ratio, open(save_folder + session_name + "/feedback_ratio_list.p", "wb+"))
