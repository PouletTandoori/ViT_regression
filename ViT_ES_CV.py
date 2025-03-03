# packages importation
from random import randint
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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
from sklearn.model_selection import KFold
from einops import repeat
from einops.layers.torch import Rearrange
from torch import Tensor


# define parser
parser = argparse.ArgumentParser(description='ViT MASW')

parser.add_argument('--dataset_name', '-data', type=str, default='Dataset1Dsmall', required=False,
                    help='Name of the dataset to use, choose between \n Dataset1Dbig \n Dataset1Dsmall \n TutorialDataset')
parser.add_argument('--nepochs', '-ne', type=int, default=2, required=False, help='number of epochs for training')
parser.add_argument('--lr', '-lr', type=float, default=0.0005, required=False, help='learning rate')
parser.add_argument('--batch_size', '-bs', type=int, default=5, required=False, help='batch size')
parser.add_argument('--output_dir', '-od', type=str, default='ViT_ES_CV_debug', required=False,
                    help='output directory for figures')
parser.add_argument('--data_augmentation', '-aug', type=bool, default=False, required=False, help='data augmentation')
parser.add_argument('--decay', '-dec', type=float, default=0, required=False, help='weight decay')
parser.add_argument('--loss', '-lo', type=str, default='NMSERL1Loss', required=False,
                    help='loss function to use, there is MSE, NMSELoss, NMSERL1Loss')

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
    '''
    Custom dataset class: load data from a folder, transform into pytorch tensors
    inputs: folder: list of files
            transform: data augmentation
    outputs: data: list of shotgathers
             labels: list of Vs profiles
    '''
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
    '''
    Create train, validate and test datasets, apply data augmentation if needed
    :param data_path: path to the data
    :param dataset_name: name of the dataset folder
    :return: train_dataset, validate_dataset, test_dataset
    '''
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
        train_dataset = CustomDataset(train_folder,transform=view_transform)
    else:
        train_dataset = CustomDataset(train_folder)
    validate_dataset = CustomDataset(validate_folder)
    test_dataset = CustomDataset(test_folder)

    return train_dataset, validate_dataset, test_dataset


train_dataset, validate_dataset, test_dataset= create_datasets(data_path, dataset_name)

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

        if self.best_score == None:
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


class PatchEmbedding(nn.Module):
    '''
    Patch Embedding class
    '''
    def __init__(self, in_channels = 1, patch_height=img_height/5,patch_width=img_width/4, emb_size = 120):

        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_height * patch_width * in_channels, emb_size),
            nn.LayerNorm(emb_size) #augmentation des performances ?
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

#%%
class Attention(nn.Module):
    '''
    Attention class
    '''
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output

#%%
class PreNorm(nn.Module):
    '''
    PreNorm class
    '''
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Sequential):
    '''
    FeedForward class
    '''
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # activation function
            nn.Dropout(dropout),# apply dropout to avoid overfitting
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

#%%
class ResidualAdd(nn.Module):
    '''
    Residual connection class
    '''
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

#%%
class ViT(nn.Module):
    '''
    Vision Transformer class, composed of PatchEmbedding, PositionalEncoding, TransformerEncoder, and RegressionHead
    '''
    def __init__(self, ch=1, img_height=img_height,img_width=img_width, emb_dim=36,
                n_layers=12, out_dim=out_dim, dropout=0.1, heads=12):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.img_height= img_height
        self.img_width = img_width
        self.patch_height= img_height // 5
        self.patch_width =  img_width // 2
        #print(f'patch size: {self.patch_height}x{self.patch_width}')
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_height=self.patch_height,
                                              patch_width=self.patch_width,
                                              emb_size=emb_dim)
        # Learnable params
        num_patches = (self.img_height // self.patch_height) * (self.img_width // self.patch_width)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))
        self.dropout = nn.Dropout(dropout) # better performances ?

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

        # Regression head
        self.head = nn.Sequential(
        nn.LayerNorm(emb_dim),  # Normalization
        nn.Linear(emb_dim, out_dim)
        )
        # out_dim = output size



    def forward(self, img):
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on regression token
        return self.head(x[:,0,:])


class Model:
    def __init__(self, model, learning_rate, device, l1_weight=1e-6):
        self.model = model
        self.lr = learning_rate
        self.l1_weight = l1_weight
        if args.loss == 'NMSELoss':
            self.loss = NMSELoss()
        elif args.loss == 'NMSERL1Loss':
            self.loss = NMSERL1Loss(l1_weight=self.l1_weight)
        elif args.loss == 'MSE':
            self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.val_loss = []
        self.device = device

    def train_step(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for i, sample in enumerate(dataloader, 0):
            inputs, labels = sample['data'].to(self.device), sample['label'].to(self.device)
            self.opt.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss(outputs, labels, self.model)
            loss.backward()
            self.opt.step()

            running_loss += loss.item()

        self.train_loss.append(running_loss / len(dataloader))

    def validation_step(self, dataloader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for i, sample in enumerate(dataloader, 0):
                inputs, labels = sample['data'].to(self.device), sample['label'].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels, self.model)
                running_loss += loss.item()

        self.val_loss.append(running_loss / len(dataloader))

    def test_step(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        inputs_list = []
        outputs_list = []
        labels_list = []

        with torch.no_grad():
            for i, sample in enumerate(dataloader, 0):
                inputs, labels = sample['data'].to(self.device), sample['label'].to(self.device)
                outputs = self.model(inputs)

                inputs_list.append(inputs.cpu().numpy())
                outputs_list.append(outputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                loss = self.loss(outputs, labels, self.model)
                running_loss += loss.item()

        self.val_loss.append(running_loss / len(dataloader))

        inputs = np.concatenate(inputs_list, axis=0)
        outputs = np.concatenate(outputs_list, axis=0)
        targets = np.concatenate(labels_list, axis=0)

        return inputs, outputs, targets

    def visualize_predictions(self, inputs, outputs, targets, od=args.output_dir, num_samples=5, bs=args.batch_size):
        print('\nVisualizing predictions')
        # Select random images

        fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(15, 5 * num_samples))

        if num_samples == 1:
            print('You must display at least 2 samples')

        else:
            for i in range(num_samples):
                # Sélectionner un indice aléatoire
                idx1 = random.randint(0, int(len(inputs)) - 1)
                print('nb samples:', int(len(inputs)) - 1)
                # Récupérer les données et les étiquettes à l'indice sélectionné
                print('inputs shape:', inputs.shape)
                data = inputs[idx1,0,:,:]
                print('data shape:', data.shape)
                label = targets[idx1]
                print('label shape:', label.shape)
                print('target shape:', targets.shape)
                prediction = outputs[idx1]
                print('prediction shape:', prediction.shape)

                # convert the image from grid points into real units
                dt = 0.00002 * 100  # dt * resampling
                time_vector = np.arange(data.shape[0]) * dt
                print('time vector len:',len(time_vector))
                nb_traces = data.shape[1]
                print('nb traces:',nb_traces)
                dz = 0.25
                # print('label shape:',label.shape)
                depth_vector = np.arange(label.shape[0]) * dz

                # Afficher l'image dans la première colonne
                axs[i, 0].imshow(data, aspect='auto', cmap='gray',
                                 extent=[0, nb_traces, time_vector[-1], time_vector[0]])
                axs[i, 0].set_title(f'Shot Gather {idx1}')
                axs[i, 0].set_xlabel('Traces')
                axs[i, 0].set_ylabel('Time (s)')

                # Afficher le label dans la deuxième colonne
                axs[i, 1].plot(label, depth_vector)
                axs[i, 1].plot(prediction, depth_vector)
                axs[i, 1].invert_yaxis()
                axs[i, 1].set_xlabel('Vs (m/s)')
                axs[i, 1].set_ylabel('Depth (m)')
                axs[i, 1].set_title(f'Vs Depth {idx1}')

        plt.tight_layout()
        plt.savefig(f'figures/{od}/predictionsVSlabels.png', format='png')
        plt.close()


def cross_validate(data_path, dataset_name, l1_weight, k=5):
    dataset = CustomDataset(glob.glob(os.path.join(data_path, dataset_name, '*', '*')))
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{k}')

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ConvNet = ViT().to(device)
        model = Model(ConvNet, args.lr, device, l1_weight=l1_weight)
        early_stopping = EarlyStopping(patience=5, verbose=True, path=f'checkpoint_fold{fold}.pt')

        for epoch in tqdm(range(args.nepochs), desc=f'Epoch {fold + 1}'):
            model.train_step(train_dataloader)
            model.validation_step(val_dataloader)

            early_stopping(model.val_loss[-1], model.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.model.load_state_dict(torch.load(f'checkpoint_fold{fold}.pt'))
        fold_results.append(min(model.val_loss))

    return np.mean(fold_results)


# Tester différentes valeurs de régularisation
l1_weights = [1e-8,1e-7,1e-6, 1e-5]
#l1_weights= [1e-6]
best_l1_weight = None
best_loss = float('inf')

for l1_weight in l1_weights:
    avg_loss = cross_validate(data_path, dataset_name, l1_weight, k=5)
    print(f'l1_weight: {l1_weight}, Avg Loss: {avg_loss}')
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_l1_weight = l1_weight

    print(f'Best l1_weight: {best_l1_weight} with Avg Loss: {best_loss}')

# Entraîner le modèle final avec la meilleure valeur de régularisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ConvNet = ViT().to(device)
final_model = Model(ConvNet, args.lr, device, l1_weight=best_l1_weight)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True)
early_stopping = EarlyStopping(patience=5, verbose=True, path='checkpoint_final.pt')

for epoch in tqdm(range(args.nepochs), desc='Epoch'):
    final_model.train_step(train_dataloader)
    final_model.validation_step(val_dataloader)

    early_stopping(final_model.val_loss[-1], final_model.model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

final_model.model.load_state_dict(torch.load('checkpoint_final.pt'))
inputs, outputs, targets = final_model.test_step(test_dataloader)
print(f'inputs shape: {inputs.shape}')
final_model.visualize_predictions(inputs, outputs, targets, od=args.output_dir, num_samples=5, bs=args.batch_size)

# Plot des courbes d'apprentissage
plt.figure(dpi=150)
plt.grid()
plt.plot(final_model.train_loss, label='Train Loss')
plt.plot(final_model.val_loss, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'figures/{args.output_dir}/learning_curves.png', format='png')
plt.close()
