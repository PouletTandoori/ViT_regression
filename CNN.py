# packages importation
from random import randint
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from einops import repeat
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import time
from Utilities import *
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import h5py as h5
from data_augmentation import *

# define parser
parser = argparse.ArgumentParser(description='VGG19 MASW')

parser.add_argument('--dataset_name', '-data', type=str, default='Dataset1Dsimple', required=False,
                    help='Name of the dataset to use, choose between \n Dataset1Dbig \n Dataset1Dsmall \n TutorialDataset')
parser.add_argument('--nepochs', '-ne', type=int, default=1, required=False, help='number of epochs for training')
parser.add_argument('--lr', '-lr', type=float, default=0.0005, required=False, help='learning rate')
parser.add_argument('--batch_size', '-bs', type=int, default=1, required=False, help='batch size')
parser.add_argument('--output_dir', '-od', type=str, default='CNN_debug', required=False, help='output directory for figures')
parser.add_argument('--data_augmentation', '-aug', type=bool, default=False, required=False, help='data augmentation')
parser.add_argument('--decay', '-dec', type=float, default=0, required=False, help='weight decay')
parser.add_argument('--loss','-lo', type=str, default='JeffLoss', required=False, help='loss function to use, there is MSE, NMSELoss, NMSERL1Loss, MSERL1Loss')

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
                GaussianNoise(mean=0, std=0.05),
                TraceShift2(shift_ratio=0.01, contiguous_ratio=0.2)
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


class VGG19(nn.Module):
    def __init__(self, out_dim=out_dim):
        super(VGG19, self).__init__()

        # Feature extraction layers: Convolutional and pooling layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=3, padding=1
            ),  # 3 input channels, 64 output channels, 3x3 kernel, 1 padding
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # Max pooling with 2x2 kernel and stride 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for classification
        self.regressor = nn.Sequential(
            nn.Linear(
                141312, 4096
            ),  # 512 channels, 46x3 spatial dimensions after max pooling
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer with 0.5 dropout probability
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, out_dim),  # Output layer with 'ou_dim' output units
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Pass input through the feature extractor layers
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.regressor(x)  # Pass flattened output through the classifier layers
        return x




# Initialize and test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG19(out_dim).to(device)
print(model)
#print the number of learnable parameters
unfrozen_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of learnable parameters: {unfrozen_parameters}')




class NMSERL1Loss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super(NMSERL1Loss, self).__init__()
        self.l1_weight = l1_weight

    def forward(self, y_pred, y_true, model):
        mse = torch.mean((y_true - y_pred) ** 2)
        variance = torch.mean(y_true ** 2)
        nmse = mse / variance
        l1_loss = sum(param.abs().sum() for param in model.parameters())
        total_loss = nmse + self.l1_weight * l1_loss
        return total_loss


class Model:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.lr = learning_rate
        if args.loss == 'NMSELoss':
            self.loss = NMSELoss()
        elif args.loss == 'NMSERL1Loss':
            self.loss = NMSERL1Loss(l1_weight=5*1e-6)
        elif args.loss == 'MSE':
            self.loss = nn.MSELoss()
        elif args.loss == 'MSERL1Loss':
            self.loss = MSERL1Loss(l1_weight=5*1e-6)
        else:
            alpha = 0.02; beta = 0.1; l1param = 0.05; vmaxparam = 0.2
            self.loss= JeffLoss(alpha=alpha, beta=beta, l1=l1param, v_max=vmaxparam)
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
            elif args.loss == 'MSERL1Loss':
                loss = self.loss(outputs.T, targets, self.model)
            else:
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
                elif args.loss == 'MSERL1Loss':
                    loss = self.loss(outputs.T, targets, self.model)
                else:
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
                # print('data shape:', data.shape)
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
                elif args.loss == 'MSERL1Loss':
                    loss = self.loss(outputs.T, targets, self.model)
                else:
                    loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())

                # Store the results to visualize them later
                images.append(inputs.cpu().numpy())
                predictions.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())

        print("loss : ", np.mean(batch_loss))
        return images, predictions, labels,np.mean(batch_loss)


# Initialize and train the model
epochs = args.nepochs
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = args.lr
batch = args.batch_size
ConvNet = VGG19(out_dim).to(device)
model = Model(ConvNet, learning_rate, device)

t0= time.time()
for epoch in tqdm(range(epochs), desc='Epoch'):
    model.train_step(train_dataloader)
    model.validation_step(val_dataloader)
t1= time.time()
best_time= t1-t0
inputs, outputs, targets,test_loss = model.test_step(test_dataloader)
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

#save runs infos
#display on a signgle image all informations about the current model
main_path= os.path.abspath(__file__)
display_run_info(model=model,od=args.output_dir,args=args,metrics=test_loss,training_time=best_time,main_path=main_path,nb_param=unfrozen_parameters)


#save the model parameters mais model.state.dict n'existe pas
torch.save(model, f'{args.output_dir}/model.pth')

