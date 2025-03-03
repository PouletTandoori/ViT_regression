# packages importation
from random import randint
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from Utilities import *
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import h5py as h5
import torch.nn.functional as F

# define parser
parser = argparse.ArgumentParser(description='CNN MASW')

parser.add_argument('--dataset_name', '-data', type=str, default='Dataset1Dsmall', required=False,
                    help='Name of the dataset to use, choose between \n Dataset1Dbig \n Dataset1Dsmall \n TutorialDataset')
parser.add_argument('--nepochs', '-ne', type=int, default=1, required=False, help='number of epochs for training')
parser.add_argument('--lr', '-lr', type=float, default=0.0005, required=False, help='learning rate')
parser.add_argument('--batch_size', '-bs', type=int, default=1, required=False, help='batch size')
parser.add_argument('--output_dir', '-od', type=str, default='ResNet_debug', required=False, help='output directory for figures')
parser.add_argument('--data_augmentation', '-aug', type=bool, default=False, required=False, help='data augmentation')
parser.add_argument('--decay', '-dec', type=float, default=0, required=False, help='weight decay')
parser.add_argument('--loss','-lo', type=str, default='NMSERL1Loss', required=False, help='loss function to use, there is MSE, NMSELoss, NMSERL1Loss')

args = parser.parse_args()

# Paths
data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
dataset_name = args.dataset_name
files_path = os.path.join(data_path, dataset_name)

train_folder = glob.glob(f'{files_path}/train/*')
validate_folder = glob.glob(f'{files_path}/validate/*')
test_folder = glob.glob(f'{files_path}/test/*')

setup_directories(name=args.output_dir)

# Verify if the dataset contains NaN values
print('VERIFYING DATASET')
bad_files = verify([train_folder, validate_folder, test_folder])

class CustomDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.data, self.labels = self.load_data_from_folder(folder)
        self.transform = transform

    def load_data_from_folder(self, folder):
        data = []
        labels = []

        for file_path in folder:
            with h5.File(file_path, 'r') as h5file:
                inputs = h5file['shotgather'][:]
                len_inp = inputs.shape[1]
                half_ind = len_inp // 2
                inputs = h5file['shotgather'][:, half_ind:]
                labels_data = h5file['vsdepth'][:]

                inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))

                transform_tensor = transforms.Compose([
                    transforms.ToTensor()
                ])

                inputs = transform_tensor(inputs)

                data.append(inputs)
                labels.append(labels_data)

        data = np.array(data)

        labels = np.array(labels)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx]
        labels = self.labels[idx]

        # Convert inputs and labels to Tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            inputs = self.transform(inputs)

        sample = {'data': inputs, 'label': labels}

        return sample

def create_datasets(data_path, dataset_name):
    train_folder = glob.glob(os.path.join(data_path, dataset_name, 'train', '*'))
    validate_folder = glob.glob(os.path.join(data_path, dataset_name, 'validate', '*'))
    test_folder = glob.glob(os.path.join(data_path, dataset_name, 'test', '*'))

    if args.data_augmentation:
        view_transform = transforms.Compose(
            [
                DeadTraces(),
                MissingTraces(),
                GaussianNoise(mean=0, std=0.1),
            ]
        )
        train_dataset = CustomDataset(train_folder, transform=view_transform)
    else:
        train_dataset = CustomDataset(train_folder)
    validate_dataset = CustomDataset(validate_folder)
    test_dataset = CustomDataset(test_folder)

    return train_dataset, validate_dataset, test_dataset

train_dataset, validate_dataset, test_dataset = create_datasets(data_path, dataset_name)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

print('INFORMATION ABOUT THE DATASET:')
print('Image size: ', train_dataloader.dataset.data[0].shape)
channels, img_height, img_width = train_dataloader.dataset.data[0].shape
print('Label size: ', train_dataloader.dataset.labels[0].shape)
lab_size = train_dataloader.dataset.labels[0].shape[0]
out_dim = lab_size
print('Output dimension: ', out_dim)
print('\n')

# Verify if data is normalized
print('check a random image to verify if data is normalized:')
print('min:', train_dataloader.dataset.data[0].min(), ' max:', train_dataloader.dataset.data[0].max())

def plot_random_samples(train_dataloader, num_samples=5, od=args.output_dir):
    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        idx = randint(0, len(train_dataloader.dataset) - 1)
        sample = train_dataloader.dataset[idx]
        image = sample['data']
        label = sample['label']

        axs[i, 0].imshow(image[0], aspect='auto', cmap='gray')
        axs[i, 0].set_title(f'Shot Gather {i + 1}')
        axs[i, 0].set_xlabel('Distance (m)')
        axs[i, 0].set_ylabel('Time (sample)')

        axs[i, 1].plot(label, range(len(label)))
        axs[i, 1].invert_yaxis()
        axs[i, 1].set_xlabel('Vs (m/s)')
        axs[i, 1].set_ylabel('Depth (m)')
        axs[i, 1].set_title(f'Vs Depth {i + 1}')

    plt.tight_layout()
    plt.savefig(f'figures/{od}/random_samples.png', format='png')
    plt.close()

plot_random_samples(train_dataloader, num_samples=5)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, out_dim=out_dim):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(11776, out_dim)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)


# Initialize and test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
print(model)


class Model:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.lr = learning_rate
        if args.loss == 'NMSELoss':
            self.loss = NMSELoss()
        elif args.loss == 'NMSERL1Loss':
            self.loss = NMSERL1Loss(l1_weight=1e-6)
        elif args.loss == 'MSE':
            self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.val_loss = []
        self.device = device

    def train_step(self, dataloader):
        self.model.train()
        batch_loss = []
        for sample in dataloader:
            inputs = sample['data'].to(self.device)
            targets = sample['label'].to(self.device)
            self.opt.zero_grad()

            outputs = self.model(inputs)

            # check dimensions match
            if outputs.shape != targets.shape:
                targets = targets.view_as(outputs)
            if args.loss == 'NMSERL1Loss':
                loss = self.loss(outputs, targets, self.model)
            elif args.loss == 'NMSELoss':
                loss = self.loss(outputs, targets)
            elif args.loss == 'MSE':
                loss = self.loss(outputs, targets)
            loss.backward()
            self.opt.step()
            batch_loss.append(loss.item())

        self.train_loss.append(np.mean(batch_loss))

    def validation_step(self, dataloader):
        self.model.eval()
        batch_loss = []
        with torch.no_grad():
            for sample in dataloader:
                inputs = sample['data'].to(self.device)
                targets = sample['label'].to(self.device)

                outputs = self.model(inputs)

                # check dimensions match
                if outputs.shape != targets.shape:
                    targets = targets.view_as(outputs)

                if args.loss == 'NMSERL1Loss':
                    loss = self.loss(outputs, targets, self.model)
                elif args.loss == 'NMSELoss':
                    loss = self.loss(outputs, targets)
                elif args.loss == 'MSE':
                    loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())

        self.val_loss.append(np.mean(batch_loss))

    def visualize_predictions(self,inputs, outputs, targets, od=args.output_dir, num_samples=5,bs=args.batch_size):
        print('\nVisualizing predictions')
        # Select random images

        fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(15, 5 * num_samples))

        if num_samples == 1:
            print('You must display at least 2 samples')

        else:
            for i in range(num_samples):
                # Sélectionner un indice aléatoire
                idx1 = random.randint(0, int(len(inputs)) - 1)

                # Select a random value inside the batch
                idx= random.randint(0,bs-1)
                # Récupérer les données et les étiquettes à l'indice sélectionné
                data = inputs[idx1][idx][0]
                print('data shape:', data.shape)
                label = targets[idx1][idx].T
                # print('label shape:', label.shape)
                prediction = outputs[idx1][idx].T
                # print('prediction shape:', prediction.shape)

                # convert the image from grid points into real units
                dt = 0.00002 * 100  # dt * resampling
                time_vector = np.arange(data.shape[0]) * dt
                # print('time vector len:',len(time_vector))
                nb_traces = data.shape[1]
                # print('nb traces:',nb_traces)
                dz = 0.25
                # print('label shape:',label.shape)
                depth_vector = np.arange(label.shape[0]) * dz

                # Afficher l'image dans la première colonne
                axs[i, 0].imshow(data, aspect='auto', cmap='gray',
                                 extent=[0, nb_traces, time_vector[-1], time_vector[0]])
                axs[i, 0].set_title(f'Shot Gather {idx}')
                axs[i, 0].set_xlabel('Traces')
                axs[i, 0].set_ylabel('Time (s)')

                # Afficher le label dans la deuxième colonne
                axs[i, 1].plot(label, depth_vector)
                axs[i, 1].plot(prediction, depth_vector)
                axs[i, 1].invert_yaxis()
                axs[i, 1].set_xlabel('Vs (m/s)')
                axs[i, 1].set_ylabel('Depth (m)')
                axs[i, 1].set_title(f'Vs Depth {idx}')

        plt.tight_layout()
        plt.savefig(f'figures/{od}/predictionsVSlabels.png', format='png')
        plt.close()

    def test_step(self, dataloader):
        self.model.eval()
        batch_loss = []
        images = []
        predictions = []
        labels = []
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                inputs = sample['data'].to(self.device)
                targets = sample['label'].to(self.device)

                outputs = self.model(inputs)

                # check dimensions match
                if outputs.shape != targets.shape:
                    targets = targets.view_as(outputs)

                if args.loss == 'NMSERL1Loss':
                    loss = self.loss(outputs, targets, self.model)
                elif args.loss == 'NMSELoss':
                    loss = self.loss(outputs, targets)
                elif args.loss == 'MSE':
                    loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())

                # Store the results to visualize them later
                images.append(inputs.cpu().numpy())
                predictions.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())

        print("loss : ", np.mean(batch_loss))
        return images, predictions, labels


# Initialize EarlyStopping
early_stopping = EarlyStopping(patience=5, verbose=True, path='checkpoint.pt')

# Initialize and train the model
epochs = args.nepochs
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = args.lr
batch = args.batch_size
ConvNet = ResNet18().to(device)
model = Model(ConvNet, learning_rate, device)

for epoch in tqdm(range(epochs), desc='Epoch'):
    model.train_step(train_dataloader)
    model.validation_step(val_dataloader)

    # Check for early stopping
    early_stopping(model.val_loss[-1], model.model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model
model.model.load_state_dict(torch.load('checkpoint.pt'))
inputs, outputs, targets = model.test_step(test_dataloader)
print('inputs shape:', inputs[0].shape)
print('outputs shape:', outputs[0].shape)
print('targets shape:', targets[0].shape)
model.visualize_predictions(inputs, outputs, targets, od=args.output_dir, num_samples=5, bs=args.batch_size)

# Plot accuracy
plt.figure(dpi=150)
plt.grid()
plt.plot(model.train_loss, label='Train Loss')
plt.plot(model.val_loss, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'figures/{args.output_dir}/learning_curves.png', format='png')
plt.close()
