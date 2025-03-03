import os
import glob
import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_augmentation import MissingTraces, DeadTraces, GaussianNoise, TraceShift2
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, folder, transform=None, use_dispersion=False):
        self.use_dispersion = use_dispersion
        self.data, self.labelsVS,self.labelsVP = self.load_data_from_folder(folder)
        self.transform = transform

    def load_data_from_folder(self, folder):
        data, labels_vs, labels_vp = [], [], []

        for file_path in folder:
            with h5.File(file_path, 'r') as h5file:
                inputs = h5file['shotgather'][:]

                if self.use_dispersion:
                    half_ind = inputs.shape[1] // 2
                    inputs = inputs[:, half_ind:]
                    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
                else:
                    inputs = inputs[:, int(inputs.shape[1] / 2):]

                    # display before nromalisation:
                    plt.imshow(inputs, aspect='auto', cmap='gray')
                    plt.title('tir sismique non normalisé')
                    plt.xlabel('Numéro de la trace')
                    plt.ylabel('Temps (s)')
                    plt.show()
                    plt.close()

                    # Normalisation, divide by the maximum value, each trace is normalised independently:
                    #first step, increase each trace by the minimum value if it is negative:
                    inputs = inputs - np.min(inputs, axis=1)[:, None]
                    inputs = inputs / np.max(np.abs(inputs), axis=1)[:, None]

                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    transform_resize = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor()
                    ])
                    inputs = transform_resize(inputs)

                    # display after nromalisation:
                    plt.imshow(inputs[0], aspect='auto', cmap='gray')
                    plt.title('tir sismique normalisé')
                    plt.xlabel('Numéro de la trace')
                    plt.ylabel('Temps (s)')
                    plt.show()
                    plt.close()

                    if inputs.shape[0] == 1:
                        inputs = inputs.repeat(3, 1, 1)

                    inputs = inputs.numpy()

                labels_data = h5file['vsdepth'][:]
                data.append(inputs)
                #data = [torch.tensor(d, dtype=torch.float32) for d in data]



                labels_vs.append(labels_data)

                if 'vpdepth' in h5file.keys():
                    labels_data_vp = h5file['vpdepth'][:]
                    labels_vp.append(labels_data_vp)

        return data, labels_vs, labels_vp


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.data[idx], dtype=torch.float32)
        labelsVS = torch.tensor(self.labelsVS[idx], dtype=torch.float32)
        labelsVP = torch.tensor(self.labelsVP[idx], dtype=torch.float32)

        if self.transform:
            inputs = self.transform(inputs)

        return {'data': inputs, 'label_VS': labelsVS, 'label_VP': labelsVP}

def create_datasets(data_path, dataset_name, use_dispersion=False, data_augmentation=None):
    train_folder = glob.glob(os.path.join(data_path, dataset_name, 'train', '*'))
    validate_folder = glob.glob(os.path.join(data_path, dataset_name, 'validate', '*'))
    test_folder = glob.glob(os.path.join(data_path, dataset_name, 'test', '*'))

    print(' creating subdatasets folders : DONE')

    transform = None
    if data_augmentation is not None:
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print('creating dataloaders: DONE')

    return train_dataloader, val_dataloader, test_dataloader


#test it on a new dataset: ('Halton_debug')
if __name__ == '__main__':
    data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_ViT/DatasetGeneration/Datasets/'
    dataset_name = 'Halton_debug'

    # Ces lignes ici donnent de bons shotgathers
    #train_folder = glob.glob(os.path.join(data_path, dataset_name, 'train', '*'))
    #train_dataset = CustomDataset(train_folder, use_dispersion=False)
    #train_dataloader=DataLoader(train_dataset, batch_size=1, shuffle=False)


    # tandis que ces lignes donnent des shotgathers qui ne sont pas bons
    train_dataloader,_,_=create_dataloaders(data_path=data_path, dataset_name=dataset_name, batch_size=1, use_dispersion=False, data_augmentation=None)

    shotgather=train_dataloader.dataset.data[0]
    print('Shotgather shape:', shotgather.shape)
    # Time vector and number of traces for the shot gathers
    time_vector = np.linspace(0, 1.5, shotgather.shape[1])
    print('Time vector:', time_vector.shape)
    nb_traces = shotgather.shape[2]
    print('Number of traces:', nb_traces)

    #verify if each colomn is normalised:
    print('min:',shotgather[0].min(),' max:', shotgather[0].max())

    # check one image from train_dataset:
    import matplotlib.pyplot as plt
    plt.imshow(shotgather[0], aspect='auto', cmap='gray',
                       extent=[0, nb_traces, time_vector[-1], time_vector[0]])
    plt.title('tir sismique normalisé trace par trace entre 0 et 1')
    plt.xlabel('Numéro de la trace')
    plt.ylabel('Temps (s)')

    plt.show()