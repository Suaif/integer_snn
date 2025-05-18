import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NMNISTDataset(Dataset):
    """ N-MNIST Dataset """

    def __init__(self, data_root_path, timestep, dt):
        self.compact_event_label_list = gen_all_event_label_data(data_root_path)
        self.timestep = timestep
        self.dt = dt

    def __len__(self):
        return len(self.compact_event_label_list)

    def __getitem__(self, item):
        event_image = transform_compact_event_to_array(np.zeros((2, 34, 34, self.timestep * self.dt)),
                                                       self.compact_event_label_list[item][0])
        event_image = event_image.astype(np.float)
        event_image = event_image.reshape(2, 34, 34, self.timestep, self.dt).max(axis=4).squeeze()
        event_image = torch.from_numpy(event_image).float()
        label = self.compact_event_label_list[item][1]
        sample = [event_image, label]
        return sample


def gen_all_event_label_data(root_path):
    """
    Generate all compact event and label data from given path (sort with file name index)

    :param root_path: root path to all data
    :type root_path: str
    :return: [*[compact_event, label]]
    :rtype: list
    """
    file_name_label_list = []
    for label in range(10):
        file_name_label_list += [[f, label] for f in os.listdir(root_path+'/'+str(label))
                                 if os.path.isfile(os.path.join(root_path+'/'+str(label), f))]
    file_name_label_list.sort(key=lambda seq: seq[0])
    compact_event_label_list = []
    for ii, name_label in enumerate(file_name_label_list, 0):
        event_file_name = root_path + '/' + str(name_label[1]) + '/' + name_label[0]
        compact_event = read_raw_nmnist_file(event_file_name)
        compact_event_label_list.append([compact_event, name_label[1]])
    print("Event data in ", root_path, ' all read, number of data ', len(compact_event_label_list))
    return compact_event_label_list


def read_raw_nmnist_file(filename):
    """
    Read raw N-MNIST file and return compact data

    :param filename: filename
    :type filename: str
    :return: [x, y, p, t]
    :rtype: [*numpy array]
    """
    with open(filename, 'rb') as input_file:
        input_byte_array = input_file.read()
    input_as_int = np.asarray([x for x in input_byte_array])
    x_event = input_as_int[0::5]
    y_event = input_as_int[1::5]
    p_event = input_as_int[2::5] >> 7
    t_event = ((input_as_int[2::5] << 16) | (input_as_int[3::5] << 8) | (input_as_int[4::5])) & 0x7FFFFF
    t_event = t_event / 1000  # convert spike times to ms
    # round event information
    x = np.round(x_event).astype(int)
    y = np.round(y_event).astype(int)
    p = np.round(p_event).astype(int)
    t = np.round(t_event).astype(int)
    return [x, y, p, t]


def transform_compact_event_to_array(empty_array, compact_event_data):
    """
    Transform compact event data to a numpy array

    :param empty_array: empty numpy array
    :type empty_array: numpy array
    :param compact_event_data: compact event data
    :type compact_event_data: list of numpy array
    :return: empty_array
    :rtype: numpy array
    """
    x_event, y_event, p_event, t_event = compact_event_data
    valid_ind = np.argwhere((x_event < empty_array.shape[2]) &
                            (y_event < empty_array.shape[1]) &
                            (p_event < empty_array.shape[0]) &
                            (t_event < empty_array.shape[3]) &
                            (x_event >= 0) &
                            (y_event >= 0) &
                            (p_event >= 0) &
                            (t_event >= 0))
    empty_array[p_event[valid_ind],
                y_event[valid_ind],
                x_event[valid_ind],
                t_event[valid_ind]] = 1
    return empty_array
