import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
import time
import os
import pickle


def biograd_snn_training(network, train_dataset, test_dataset, sleep_spike_ts, device, soft_error_step, session_name,
                         validation_size=10000, batch_size=128, sleep_batch_size=128,
                         test_batch_size=128, epoch=100,
                         save_epoch=1, lr=1e-3,
                         sleep_oja_power=2, sleep_lr=1e-3):
    """
    BioGrad SNN training with sleep

    Args:
        network (SNN): Online learning SNN
        train_dataset (dataset): training dataset
        test_dataset (dataset): test dataset
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
        os.mkdir("./save_models")
        print("Directory save_models Created")
    except FileExistsError:
        print("Directory params already exists")

    session_file_dir = "./save_models/" + session_name
    try:
        os.mkdir(session_file_dir)
        print("Directory " + session_file_dir + " Created")
    except FileExistsError:
        print("Directory " + session_file_dir + " already exists")

    # Train, validation, and test dataloader
    train_idx = [idx for idx in range(len(train_dataset) - validation_size)]
    val_idx = [(idx + len(train_idx)) for idx in range(validation_size)]
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  shuffle=False, num_workers=4)
    val_dataloader = DataLoader(train_dataset, batch_size=test_batch_size, sampler=val_sampler,
                                shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 shuffle=False, num_workers=4)

    # Number of samples in train, validation, and test dataset
    train_num = len(train_idx)
    val_num = len(val_idx)
    test_num = len(test_dataset)

    # List for save accuracy
    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []

    # Define tensorboard
    tf_writer = SummaryWriter(comment='_' + session_name)

    # save init model
    pickle.dump(network, open(session_file_dir + "/model_init.p", "wb+"))

    # Compute init angle and ratio between feedback weight and forward weight
    feedback_angle, feedback_ratio = [], []
    angle_list, ratio_list = network.compute_feedback_angle_ratio()
    feedback_angle.append(angle_list)
    feedback_ratio.append(ratio_list)
    for hh in range(len(angle_list)):
        tf_writer.add_scalar('nmnist_exp/feedback_angle_hidden' + str(hh), angle_list[hh], len(feedback_angle))
        tf_writer.add_scalar('nmnist_exp/feedback_ratio_hidden' + str(hh), ratio_list[hh], len(feedback_ratio))

    # Start training
    with torch.no_grad():
        for ee in range(epoch):
            # Training
            train_correct = 0
            train_start = time.time()
            for data in train_dataloader:
                event_img, labels = data
                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()
                event_img, labels, labels_one_hot = event_img.to(device), labels.to(device), labels_one_hot.to(device)
                predict_label, hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step = network.train_online(
                    event_img, labels_one_hot, soft_error_step)
                network.train_update_parameter(hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step, lr)
                train_correct += ((predict_label == labels).sum().to("cpu")).item()

                # Put network to sleep for feedback training
                network.sleep_feedback_update(sleep_batch_size, sleep_spike_ts, sleep_oja_power, sleep_lr)

                # Compute angle and ratio between feedback weight and forward weight after each update
                angle_list, ratio_list = network.compute_feedback_angle_ratio()
                feedback_angle.append(angle_list)
                feedback_ratio.append(ratio_list)
                for hh in range(len(angle_list)):
                    tf_writer.add_scalar('nmnist_exp/feedback_angle_hidden' + str(hh), angle_list[hh],
                                         len(feedback_angle))
                    tf_writer.add_scalar('nmnist_exp/feedback_ratio_hidden' + str(hh), ratio_list[hh],
                                         len(feedback_ratio))

            train_end = time.time()
            train_accuracy_list.append(train_correct / train_num)
            tf_writer.add_scalar('nmnist_exp/train_accuracy', train_accuracy_list[-1], ee)
            log_message = "Epoch %d Training Accuracy %.4f" % (ee, train_accuracy_list[-1])
            print(log_message, end=" ")
            with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
                log_file.write(log_message + '\n')

            # Validation
            val_correct = 0
            val_start = time.time()
            for data in val_dataloader:
                event_img, labels = data
                event_img, labels = event_img.to(device), labels.to(device)
                predict_label = network.test(event_img)
                val_correct += ((predict_label == labels).sum().to("cpu")).item()
            val_end = time.time()
            val_accuracy_list.append(val_correct / val_num)
            tf_writer.add_scalar('nmnist_exp/val_accuracy', val_accuracy_list[-1], ee)
            log_message = "Validate Accuracy %.4f" % val_accuracy_list[-1]
            print(log_message, end=" ")
            with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
                log_file.write(log_message + '\n')

            # Testing
            test_correct = 0
            test_start = time.time()
            for data in test_dataloader:
                event_img, labels = data
                event_img, labels = event_img.to(device), labels.to(device)
                predict_label = network.test(event_img)
                test_correct += ((predict_label == labels).sum().to("cpu")).item()
            test_end = time.time()
            test_accuracy_list.append(test_correct / test_num)
            tf_writer.add_scalar('nmnist_exp/test_accuracy', test_accuracy_list[-1], ee)
            log_message = "Test Accuracy %.4f Training Time: %.1f Val Time: %.1f Test Time: %.1f" % (
            test_accuracy_list[-1], train_end - train_start, val_end - val_start, test_end - test_start)
            print(log_message)
            with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
                log_file.write(log_message + '\n')

            # Save model
            if (ee + 1) % save_epoch == 0:
                pickle.dump(network, open(session_file_dir + "/model_" + str(ee + 1) + ".p", "wb+"))

    print("End Training")
    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write("End Training\n")
    return train_accuracy_list, val_accuracy_list, test_accuracy_list, feedback_angle, feedback_ratio
