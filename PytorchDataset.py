import os
import glob
import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_augmentation import MissingTraces, DeadTraces, GaussianNoise, TraceShift2

class CustomDataset(Dataset):
    def __init__(self, folder, transform=None, use_dispersion=False):
        self.use_dispersion = use_dispersion
        self.data, self.labels = self.load_data_from_folder(folder)
        self.transform = transform

    def load_data_from_folder(self, folder):
        data, labels = [], []

        for file_path in folder:
            with h5.File(file_path, 'r') as h5file:
                inputs = h5file['shotgather'][:]

                if self.use_dispersion:
                    half_ind = inputs.shape[1] // 2
                    inputs = inputs[:, half_ind:]
                    # transform to tensor:
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    #add one dimension
                    inputs = inputs.unsqueeze(0)

                else:
                    inputs = inputs[:, int(inputs.shape[1] / 2):]
                    #noramlize data, each trace individually
                    min_vals = np.min(inputs, axis=1, keepdims=True)
                    max_vals = np.max(inputs, axis=1, keepdims=True)
                    range_vals = max_vals - min_vals
                    inputs = (inputs - min_vals) / np.where(range_vals == 0, 1,
                                                            range_vals)  # Évite une division par zéro

                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    transform_resize = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor()
                    ])
                    inputs = transform_resize(inputs)

                    if inputs.shape[0] == 1:
                        inputs = inputs.repeat(3, 1, 1)

                    inputs = inputs.numpy()

                labels_data = h5file['vsdepth'][:]
                data.append(inputs)
                labels.append(labels_data)

        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.data[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            inputs = self.transform(inputs)

        return {'data': inputs, 'label': labels}

def create_datasets(data_path, dataset_name, use_dispersion=False, data_augmentation=None):
    train_folder = glob.glob(os.path.join(data_path, dataset_name, 'train', '*'))
    validate_folder = glob.glob(os.path.join(data_path, dataset_name, 'validate', '*'))
    test_folder = glob.glob(os.path.join(data_path, dataset_name, 'test', '*'))

    transform = None
    if data_augmentation:
        transform = transforms.Compose([
            DeadTraces(),
            MissingTraces(),
            GaussianNoise(mean=0, std=0.05),
            TraceShift2(shift_ratio=0.01, contiguous_ratio=0.2),
        ])

    train_dataset = CustomDataset(train_folder, transform=transform, use_dispersion=use_dispersion)
    validate_dataset = CustomDataset(validate_folder, use_dispersion=use_dispersion)
    test_dataset = CustomDataset(test_folder, use_dispersion=use_dispersion)

    return train_dataset, validate_dataset, test_dataset

def create_dataloaders(data_path, dataset_name, batch_size, use_dispersion=False, data_augmentation=None):
    train_dataset, validate_dataset, test_dataset = create_datasets(data_path, dataset_name, use_dispersion, data_augmentation)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
