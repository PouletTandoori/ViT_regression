import os
import shutil
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import glob
import h5py as h5
import matplotlib.pyplot as plt

def setup_directories(name='dataset_name'):
    # Verify if the 'figures' directory exists create it if it does not
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Same for the output_dir
    vit_debugg_dir = os.path.join(figures_dir, name)

    # Remove the output_dir if it already exists
    if os.path.exists(vit_debugg_dir):
        shutil.rmtree(vit_debugg_dir)

    # Create the 'ViT_debugg' directory
    os.makedirs(vit_debugg_dir)
    print(f'All figures will be saved in {vit_debugg_dir} folder.')


def verify(folders):
    bad_files= []
    for folder in folders:
        count = 0
        for file in folder:
            with h5.File(file, 'r') as h5file:
                inputs = h5file['shotgather'][:]
                labels_data = h5file['vsdepth'][:]

                # verify if inputs contains Nan values
                if np.isnan(inputs).any():
                    count += 1
                    bad_files.append(file)
        print(f'folder contains {count}/{len(folder)} files with Nan values')
    return bad_files

def clean(bad_files, files_path):
    for file in bad_files:
        os.remove(file)
    print('Files with Nan values have been removed')
    #check the final amount of files
    train_folder = glob.glob(f'{files_path}/train/*')
    validate_folder = glob.glob(f'{files_path}/validate/*')
    test_folder = glob.glob(f'{files_path}/test/*')
    print(f'{len(train_folder)} files remaining in the training dataset')
    print(f'{len(validate_folder)} files remaining in the validation dataset')
    print(f'{len(test_folder)} files remaining in the test dataset')


class DeadTraces:
    """
    Applies dead traces to a shotgather.
    """

    def __init__(self, dead_trace_ratio=0.04):
        self.dead_trace_ratio = dead_trace_ratio
    def add_dead_traces(self,image, dead_trace_ratio):
        ''''
        Replace some random traces by dead traces to the data: a dead trace is a trace (=column) with all values set to 1

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels, H is the height and W is the width
        :param dead_trace_ratio: From 0 to 1, the ratio of dead traces to add

        :return: augmented_data: 3D tensor of shape (C, H, W) with some dead traces
        '''



        # Make a copy of the original image
        augmented_data = image.clone() if isinstance(image, torch.Tensor) else image.copy()

        # verify if image is (h,w) or (c,h,w):
        if len(image.shape) == 2:
            num_columns = augmented_data.shape[1]
            num_dead_traces = int(num_columns * dead_trace_ratio)
            dead_traces_indices = random.sample(range(num_columns), num_dead_traces)
            # set the values of the dead traces to 1
            augmented_data[:, dead_traces_indices] = 1
            return augmented_data

        # count colums in the data
        num_columns = augmented_data[0].shape[1]
        # print('nb traces=',num_columns)
        # choose missing_trace_ratio % of the traces to be dead traces randomly
        num_dead_traces = int(num_columns * dead_trace_ratio)
        # print('nb dead traces',num_dead_traces)
        # choose the indices of the dead traces
        dead_traces_indices = random.sample(range(num_columns), num_dead_traces)
        # print('indices of the dead traces:',dead_traces_indices)

        # set the values of the dead traces to 1
        augmented_data[:, :, dead_traces_indices] = 1
        return augmented_data

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.add_dead_traces(sample, self.dead_trace_ratio)

class MissingTraces:
    """
    Applies dead traces to a shotgather.
    """
    def __init__(self, missing_trace_ratio=0.04):
        self.missing_trace_ratio = missing_trace_ratio

    def Missing_traces(self,image, missing_trace_ratio):
        ''''
        Replace some random traces by missing traces to the data: a missing trace is a trace (=column) with all values set to the average value of the trace

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels, H is the height and W is the width
        :param missing_trace_ratio: From 0 to 1, the ratio of missing traces to add

        :return: augmented_data: 3D tensor of shape (C, H, W) with some missing traces
        '''

        img2D = 0
        # verify if image is (h,w) or (c,h,w):
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            img2D = 1

        # Make a copy of the original image
        augmented_data = image.clone() if isinstance(image, torch.Tensor) else image.copy()

        # count colums in the data
        num_columns = augmented_data[0].shape[1]
        # print('nb traces=',num_columns)
        # choose missing_trace_ratio % of the traces to be dead traces randomly
        num_missing_traces = int(num_columns * missing_trace_ratio)
        # print('nb missing traces',num_dead_traces)
        # choose the indices of the dead traces
        missing_traces_indices = random.sample(range(num_columns), num_missing_traces)
        # print('indices of the missing traces:',missing_traces_indices)

        average_value = augmented_data[:, :, missing_traces_indices].mean()

        # set the values of the dead traces to average
        augmented_data[:, :, missing_traces_indices] = average_value
        if img2D==1:
            augmented_data=augmented_data.squeeze(0)

        return augmented_data

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.Missing_traces(sample, self.missing_trace_ratio)

class GaussianNoise:
    """
    Adds Gaussian noise to a shotgather.
    """
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean

    def add_gaussian_noise(self, image, mean, std):
        '''
        Add Gaussian noise to the data

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels, H is the height and W is the width
        :param mean: mean of the Gaussian noise
        :param std: standard deviation of the Gaussian noise

        :return: augmented_data: 3D tensor of shape (C, H, W) with Gaussian noise
        '''
        # Make a copy of the original image
        augmented_data = image.clone() if isinstance(image, torch.Tensor) else image.copy()

        # Add Gaussian noise
        noise = torch.randn(augmented_data.shape) * std + mean
        augmented_data += noise

        return augmented_data

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.add_gaussian_noise(sample, self.mean, self.std)


def plot_a_sample(train_dataloader, i=0):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5 * 1))

    # Sélectionner le premier échantillon
    sample = train_dataloader.dataset[i]
    image = sample['data']
    label = sample['label']
    print('shape image=', image.shape)

    # Afficher l'image dans la première colonne
    axs[0].imshow(image[0], aspect='auto', cmap='gray')
    axs[0].set_title(f'Original Shot Gather {i + 1} reshaped 224x224')
    axs[0].set_xlabel('Distance (grid points, reshaped)')
    axs[0].set_ylabel('Time (dt, reshaped)')

    # Afficher le label dans la deuxième colonne
    axs[1].plot(label, range(len(label)))
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Vs (m/s)')
    axs[1].set_ylabel('Depth (grid points)')
    axs[1].set_title(f'Vs Depth ')

    plt.tight_layout()
    plt.show()

