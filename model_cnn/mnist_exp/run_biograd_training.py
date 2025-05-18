import pickle
import torch
import numpy as np
from mnist_exp.training_biograd import biograd_snn_training
from biograd_snn.network_w_biograd import BioGradNetworkWithSleep
from biograd_snn.online_error_functions import cross_entropy_loss_error_function

# Define SNN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_shape = 1 * 28 * 28
out_dim = 10
snn_param = {'hidden_layer': [(0.6, 0.3, 0.3, 1.0),
                              (0.6, 0.3, 0.3, 1.0)],
             'out_layer': (0.6, 0.3, 0.3, 1.0)}
soft_error_start = 5
spike_ts = 20
sleep_spike_ts = 50

# Define Training parameters
val_size = 10000
train_batch_size = 128
sleep_batch_size = 128
test_batch_size = 256
epoch = 100
save_epoch = 1
lr = 1.0e-3
sleep_oja_power = 2.0
sleep_lr = 1.0e-4 / 3.

# Define SNN and start training
hidden_dim_list = [500, 100]
seed_list = [0, 5, 10, 15, 20]

for seed in seed_list:
    torch.manual_seed(seed)
    np.random.seed(seed)
    session_name = "biograd_snn_sleep_seed_" + str(seed)

    online_snn = BioGradNetworkWithSleep(in_shape, out_dim, hidden_dim_list,
                                         snn_param, cross_entropy_loss_error_function, device)
    train_acc, val_acc, test_acc, fb_angle, fb_ratio = biograd_snn_training(
        online_snn, spike_ts, sleep_spike_ts, device, soft_error_start, session_name,
        validation_size=val_size, batch_size=train_batch_size, sleep_batch_size=sleep_batch_size,
        test_batch_size=test_batch_size, epoch=epoch, save_epoch=save_epoch, lr=lr,
        sleep_oja_power=sleep_oja_power, sleep_lr=sleep_lr)

    pickle.dump(train_acc, open("./save_models/" + session_name + "/train_accuracy_list.p", "wb+"))
    pickle.dump(val_acc, open("./save_models/" + session_name + "/val_accuracy_list.p", "wb+"))
    pickle.dump(test_acc, open("./save_models/" + session_name + "/test_accuracy_list.p", "wb+"))
    pickle.dump(fb_angle, open("./save_models/" + session_name + "/feedback_angle_list.p", "wb+"))
    pickle.dump(fb_ratio, open("./save_models/" + session_name + "/feedback_ratio_list.p", "wb+"))
