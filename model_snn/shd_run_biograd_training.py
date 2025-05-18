import pickle
import torch
import numpy as np
from shd_exp.config import Config
from shd_exp.training_biograd import biograd_snn_training
from biograd_snn.network_w_biograd import BioGradNetworkWithSleep
from biograd_snn.online_error_functions import LossFunction
from lr_schedulers import ExponentialDecay, CosineAnnealingWarmRestarts

# Define SNN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dim = 20

# Training parameters
train_batch_size = 128
sleep_batch_size = 128
test_batch_size = 256
epoch = 50
save_epoch = 1
sleep_oja_power = 2.0
sleep_lr = 1.0e-4 / 3.

num_bits = 16
weight_init_bits = 16
low_precision_bits = 16
activation_bits = 32

float_mode = False  # False: Int, True: Float
aligned_weights = False  # True: Aligned feedback weights, False: Transposed feedforward weights
loss_mode = 'direct'  # 'feedback', 'final', 'direct'
softmax = True  # True: Use softmax, False: Use simplified loss function

spike_stats = False
batch_stats = False
plot_batch = True
writer_spike, writer_batch, writer_epoch = False, False, True

# Network architecture
if float_mode:
    loss_precision = 1
    out_layer_dict = {
        'Vdecay': 1., 'Vth': 0.2, 'Grad_win': 0.4, 'Grad_amp': 1., 'Fb_th': 1.,
        'lr': 1e-3, 'Weight_decay': 0
    }
    hidden_layer_dict1 = {
        'Vdecay': 1., 'Vth': 0.3, 'Grad_win': 0.3, 'Grad_amp': 1.,
        'lr': 1e-3, 'Weight_decay': 0
    }
else:
    loss_precision = 128
    factor = 256.
    out_layer_dict = {
        'Vdecay': 1., 'Vth': factor * 1600, 'Grad_win': factor * 1600, 'Grad_amp': 1., 'Fb_th': 1.,
        'lr': 2**-1, 'Weight_decay': 0.
    }
    hidden_layer_dict1 = {
        'Vdecay': 1., 'Vth': factor * 2000., 'Grad_win': factor * 4000., 'Grad_amp': 1.,
        'lr': 2**-15, 'Weight_decay': 0.
    }

gradient_clip = 128
hidden_dim_list = [256]
snn_param = {'out_layer': out_layer_dict, 'hidden_layer': [hidden_layer_dict1]}

data_params = {
    'data_type': 'frame', 'duration': None, 'n_bins': 4, 'frames': 10, 'split_by': 'time'
}
in_shape = 700 // data_params['n_bins']

# Save and tensorboard folders
experiment_group = "shd/final_results/"
tf_folder = f"./runs/{experiment_group}"
save_folder = f"./save_models/{experiment_group}"
tf_name = None

# Loss function
loss_fn, loss_parameters = 'normal', None
loss_function = LossFunction(loss_fn, parameters=loss_parameters)

# Learning rate scheduler
lr_scheduler = 0

# Training loop
seed_list = [100364571]
with torch.no_grad():
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        session_name = (
            f"shd_biograd_quant_{len(snn_param['hidden_layer'])}_layers_"
            f"{'float_' if float_mode else 'int_'}"
            f"{num_bits}_{weight_init_bits}_{low_precision_bits}_" if not float_mode else ""
            f"{'aligned_' if aligned_weights else 'transposed_'}"
            f"loss_{loss_mode}_"
            f"frames_{data_params['frames']}_{data_params['split_by']}_" if data_params['duration'] is None else
            f"duration_{data_params['duration']}_"
            f"{'softmax_' if float_mode and softmax else 'simple_'}"
            f"{seed}"
        )

        online_snn = BioGradNetworkWithSleep(
            in_shape, out_dim, hidden_dim_list, snn_param, loss_function, device,
            float_mode=float_mode, aligned_weights=aligned_weights, loss_mode=loss_mode,
            gradient_clip=gradient_clip, num_bits=num_bits, weight_init_bits=weight_init_bits,
            low_precision_bits=low_precision_bits, activation_bits=activation_bits
        )

        print("Loss mode:", loss_mode)
        print("Float mode:", float_mode)
        print("Aligned weights:", aligned_weights)

        train_acc, val_acc, test_acc, fb_angle, fb_ratio, _ = biograd_snn_training(
            online_snn, device, session_name, data_params=data_params,
            batch_size=train_batch_size, sleep_batch_size=sleep_batch_size,
            test_batch_size=test_batch_size, epoch=epoch,
            sleep_oja_power=sleep_oja_power, sleep_lr=sleep_lr, loss_precision=loss_precision,
            stats=(spike_stats, batch_stats, plot_batch), float_mode=float_mode,
            lr_scheduler=lr_scheduler, writer=(writer_spike, writer_batch, writer_epoch),
            tf_folder=tf_folder, save_folder=save_folder, tf_name=tf_name
        )

        for metric, name in zip(
            [train_acc, val_acc, test_acc, fb_angle, fb_ratio],
            ["train_accuracy_list", "val_accuracy_list", "test_accuracy_list", "feedback_angle_list", "feedback_ratio_list"]
        ):
            pickle.dump(metric, open(f"{save_folder}{session_name}/{name}.p", "wb+"))
