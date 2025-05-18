import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class NMNISTDataset(Dataset):
    """ N-MNIST Dataset """

    def __init__(self, data_root_path, timestep, dt, num_sample, unlimited_mem=False):
        self.compact_event_label_list = gen_all_event_label_data(data_root_path, timestep, num_sample, unlimited_mem)
        self.overall_timestep = 60
        self.timestep = timestep
        self.dt = dt
        self.unlimited_mem = unlimited_mem

    def __len__(self):
        return len(self.compact_event_label_list)

    def __getitem__(self, item):
        if self.unlimited_mem:
            event_image = self.compact_event_label_list[item][0]
        else:
            event_image = np.zeros((2, 34, 34, self.overall_timestep))
            event_image[self.compact_event_label_list[item][0]] = 1
            if self.timestep < self.overall_timestep:
                event_image = event_image[:, :, :, :self.timestep]
            event_image = event_image.astype(np.float)
            event_image = torch.from_numpy(event_image).float()
        label = self.compact_event_label_list[item][1]
        sample = [event_image, label]
        return sample


def gen_all_event_label_data(data_root_path, timestep, num_sample, unlimited_mem):
    """
    Read all event data and label

    Args:
        data_root_path (str): root path to data
        timestep (int): spike timestep
        num_sample (int): number of samples
        unlimited_mem (bool): if true read every thing to memory

    Returns:
        event_label_list: list of event and label

    """
    event_label_list = []

    for num in range(num_sample):
        event_label = pickle.load(open(data_root_path + "/" + str(num) + ".p", "rb"))
        if unlimited_mem:
            event_image = np.zeros((2, 34, 34, 60))
            event_image[event_label[0]] = 1
            if timestep < 60:
                event_image = event_image[:, :, :, :timestep]
            event_image = event_image.astype(np.float)
            event_label_list.append([event_image, event_label[1]])
        else:
            event_label_list.append(event_label)

    return event_label_list
