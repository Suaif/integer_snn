import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
import time
import os


def stbp_snn_training(network, optimizer, train_dataset, test_dataset, device, session_name,
                      validation_size=10000, batch_size=128, test_batch_size=256, epoch=100):
    """
    STBP SNN training

    Args:
        network (SNN): STBP learning SNN
        optimizer (Torch Function): Optimizer
        train_dataset (dataset): training dataset
        test_dataset (dataset): test dataset
        device (device): device
        session_name (str): name of the training session
        validation_size (int): size of validation set
        batch_size (int): batch size for training
        test_batch_size (int): batch size for testing
        epoch (int): number of epoches

    Returns:
        train_loss_list: list of training loss for each epoch
        val_accuracy_list: list of validation accuracy for each epoch
        test_accuracy_list: list of test accuracy for each epoch

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

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    # List for save accuracy
    train_loss_list, val_accuracy_list, test_accuracy_list = [], [], []
    val_num = len(val_idx)
    test_num = len(test_dataset)

    # Define tensorboard
    tf_writer = SummaryWriter(comment='_' + session_name)

    # Start training
    network.to(device)
    for ee in range(epoch):
        running_loss = 0.0
        running_batch_num = 0
        train_start = time.time()
        for data in train_dataloader:
            event_img, labels = data
            event_img, labels = event_img.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(event_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_batch_num += 1
        train_end = time.time()
        train_loss_list.append(running_loss / running_batch_num)
        tf_writer.add_scalar('nmnist_exp/train_loss', train_loss_list[-1], ee)
        log_message = "Epoch %d Training Loss %.4f" % (ee, train_loss_list[-1])
        print(log_message)
        with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
            log_file.write(log_message + '\n')

        val_correct_num = 0
        val_start = time.time()
        with torch.no_grad():
            for data in val_dataloader:
                event_img, labels = data
                event_img, labels = event_img.to(device), labels.to(device)
                outputs = network(event_img)
                _, predicted = torch.max(outputs, 1)
                val_correct_num += ((predicted == labels).sum().to("cpu")).item()
        val_end = time.time()
        val_accuracy_list.append(val_correct_num / val_num)
        tf_writer.add_scalar('nmnist_exp/val_accuracy', val_accuracy_list[-1], ee)
        log_message = "Validate Accuracy %.4f" % val_accuracy_list[-1]
        print(log_message)
        with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
            log_file.write(log_message + '\n')

        test_correct_num = 0
        test_start = time.time()
        with torch.no_grad():
            for data in test_dataloader:
                event_img, labels = data
                event_img, labels = event_img.to(device), labels.to(device)
                outputs = network(event_img)
                _, predicted = torch.max(outputs, 1)
                test_correct_num += ((predicted == labels).sum().to("cpu")).item()
        test_end = time.time()
        test_accuracy_list.append(test_correct_num / test_num)
        tf_writer.add_scalar('nmnist_exp/test_accuracy', test_accuracy_list[-1], ee)
        log_message = "Test Accuracy %.4f Training Time: %.1f Val Time: %.1f Test Time: %.1f" % (
            test_accuracy_list[-1], train_end - train_start, val_end - val_start, test_end - test_start)
        print(log_message)
        with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
            log_file.write(log_message + '\n')

    print("End Training")
    with open(os.path.join(session_file_dir, 'training.log'), 'a') as log_file:
        log_file.write("End Training\n")
    return train_loss_list, val_accuracy_list, test_accuracy_list
