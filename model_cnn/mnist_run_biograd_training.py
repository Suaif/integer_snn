import pickle
import torch
import numpy as np
from mnist_exp.training_biograd import biograd_snn_training
from biograd_snn.network_w_biograd import BioGradNetworkWithSleep
from biograd_snn.online_error_functions import cross_entropy_loss_error_function, LossFunction
from lr_schedulers import ExponentialDecay, CosineAnnealingWarmRestarts

# Define SNN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_shape = (28, 28)
out_dim = 10

soft_error_start = 5
spike_ts = 20
sleep_spike_ts = 50

# Define Training parameters
val_size = 10000
train_batch_size = 128
sleep_batch_size = 128
test_batch_size = 256
epoch = 50
save_epoch = 1
# lr = 1
sleep_oja_power = 2.0
sleep_lr = 1.0e-4 / 3.

num_bits = 16
weight_init_bits = 16
low_precision_bits = 16
activation_bits = 32

float_mode = False # False: Int, True: Float
aligned_weights = False # True: Aligned feedback weights, False: Transposed feedforward weights
loss_mode = 'direct'
# 'feedback': traditional (accumulated loss over all time steps)
# 'final': loss only on final step
# 'direct': loss only on final step not processed in OutputLayer.feedbackstep	

spike_stats = False
batch_stats = False
plot_batch = True
writer_spike, writer_batch, writer_epoch = False, False, True

if float_mode:
    loss_precision = 1

    out_layer_dict = {'Vdecay': 0.5, 'Vth': 1.5, 'Grad_win': 1.5, 'Grad_amp': 1., 'Fb_th': 1., 
                      'lr': 1e-3, 'Weight_decay': 3e-5}
    hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': 0.5, 'Grad_win': 0.5, 'Grad_amp': 1., 'n_filters': 32, 'kernel_size': 5, 'reduce_dim': True,
                          'lr': 1e-3, 'Weight_decay': 3e-5}
    # hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': 2., 'Grad_win': 4., 'Grad_amp': 1., 'n_filters': 32, 'filter_size': 5,
    #                       'lr': 3e-3, 'Weight_decay': 0}

else:
    loss_precision = 32

    factor = 256.
    out_layer_dict = {'Vdecay': 0.5, 'Vth': factor*250, 'Grad_win': factor*500, 'Grad_amp': 1., 'Fb_th': 1., 
                      'lr': 2**-2, 'Weight_decay': 0}
    hidden_layer_dict1 = {'Vdecay': 0.5, 'Vth': factor*250., 'Grad_win': factor*500., 'Grad_amp': 1., 'n_filters': 32, 'kernel_size': 5, 'reduce_dim': True,
                          'lr': 2**-10, 'Weight_decay': 0}

# Build network architecture
gradient_clip = 2048
snn_param = {'out_layer': out_layer_dict, 'hidden_layer': [hidden_layer_dict1]}

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
experiment_group = "mnist/csnn/"
tf_folder = "./runs/" + experiment_group
save_folder = "./save_models/" + experiment_group
tf_name = None

# Define SNN and start training
seed_list = [100364571, 65941348, 55674239, 3164977, 6334270, 7026547]
seed_list = [100364571]
with torch.no_grad():
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)

        session_name = f"mnist_biograd_cnn_quant_{str(len(snn_param['hidden_layer']))}_layers_"
        session_name += "float_" if float_mode else "int_"
        if not float_mode:
            session_name += f"{num_bits}_{weight_init_bits}_{low_precision_bits}_"
        session_name += "aligned_" if aligned_weights else "transposed_"
        session_name += "loss_" + loss_mode + "_"
        session_name += str(seed)

        online_snn = BioGradNetworkWithSleep(in_shape, out_dim,
                                            snn_param, loss_function, device, float_mode=float_mode,
                                            aligned_weights = aligned_weights, loss_mode=loss_mode, gradient_clip=gradient_clip,
                                            num_bits=num_bits, weight_init_bits=weight_init_bits, low_precision_bits=low_precision_bits, activation_bits=activation_bits)
        print("Loss mode: ", loss_mode)
        print("Float mode: ", float_mode)
        print("Aligned weights: ", aligned_weights)
                
        train_acc, val_acc, test_acc, fb_angle, fb_ratio, _ = biograd_snn_training(
            online_snn, spike_ts, device, soft_error_start, session_name,
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
