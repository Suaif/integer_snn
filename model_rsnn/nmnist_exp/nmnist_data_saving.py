import os
import numpy as np
import pickle
from nmnist_exp.nmnist_dataset import gen_all_event_label_data, transform_compact_event_to_array


def save_nmnist_data(data_root_path, save_root_path, timestep, dt):
    """
    Transform N-MNIST data and save them

    Args:
        data_root_path (str): root path for data
        save_root_path (str): root path for saving
        timestep (int): spike timestep
        dt (int): time per step

    """
    # Create folder for saving data
    try:
        os.mkdir(save_root_path)
        print("Directory " + save_root_path + " Created")
    except FileExistsError:
        print("Directory params already exists")

    compact_event_label_list = gen_all_event_label_data(data_root_path)
    event_num = len(compact_event_label_list)
    for num in range(event_num):
        event_image = transform_compact_event_to_array(np.zeros((2, 34, 34, timestep * dt)),
                                                       compact_event_label_list[num][0])
        event_image = event_image.reshape(2, 34, 34, timestep, dt).max(axis=4).squeeze()
        event_image_idx = np.where(event_image == 1)
        label = compact_event_label_list[num][1]
        sample = [event_image_idx, label]
        pickle.dump(sample, open(save_root_path + "/" + str(num) + ".p", "wb+"))
        if (num + 1) % 10000 == 0:
            print("Done Transform and Save ", num + 1)


if __name__ == '__main__':
    train_dir = './data/Train'
    save_dir = './data/nmnist_compact_train'
    save_nmnist_data(train_dir, save_dir, 60, 5)
    test_dir = './data/Test'
    save_dir = './data/nmnist_compact_test'
    save_nmnist_data(test_dir, save_dir, 60, 5)
