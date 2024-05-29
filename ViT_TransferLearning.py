import os
import glob
import h5py as h5
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from random import random
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import ViTModel
import torch.optim as optim
import argparse
import time
from Utilities import *

#define parser
parser = argparse.ArgumentParser(description='ViT MASW')

parser.add_argument('--dataset_name','-data', type=str,default='Debug_simple_Dataset', required=False,
                    help='Name of the dataset to use, choose between \n Debug_simple_Dataset \n SimpleDataset \n IntermediateDataset')
parser.add_argument('--nepochs','-ne', type=int, default=101, required=False,help='number of epochs for training')
parser.add_argument('--lr','-lr', type=float, default=0.0005, required=False,help='learning rate')
parser.add_argument('--batch_size','-bs', type=int, default=1, required=False,help='batch size')
parser.add_argument('--output_dir','-od', type=str, default='ViT_TransferLearning_debug', required=False,help='output directory for figures')



args = parser.parse_args()

# Paths
data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
dataset_name = args.dataset_name
files_path = os.path.join(data_path, dataset_name)

train_folder = glob.glob(f'{files_path}/train/*')
validate_folder = glob.glob(f'{files_path}/validate/*')
test_folder = glob.glob(f'{files_path}/test/*')

setup_directories(name=args.output_dir)


class CustomDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.data, self.labels = self.load_data_from_folder(folder)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def load_data_from_folder(self, folder):
        data = []
        labels = []

        for file_path in folder:
            with h5.File(file_path, 'r') as h5file:
                inputs = h5file['shotgather'][:]
                labels_data = h5file['vsdepth'][:]

                # print('data shape:',inputs.shape)
                # print('min data=',np.min(inputs))
                inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))

                # reshape data
                inputs = torch.tensor(inputs, dtype=torch.float32)
                transform_resize = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                inputs = transform_resize(inputs)

                if inputs.shape[0] == 1:  # Si l'image est en grayscale
                    inputs = inputs.repeat(3, 1, 1)  # Convertir en RGB
                inputs = inputs.numpy()

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

        sample = {'data': inputs, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

def create_datasets(data_path, dataset_name):
    train_folder = glob.glob(os.path.join(data_path, dataset_name, 'train', '*'))
    validate_folder = glob.glob(os.path.join(data_path, dataset_name, 'validate', '*'))
    test_folder = glob.glob(os.path.join(data_path, dataset_name, 'test', '*'))

    train_dataset = CustomDataset(train_folder)
    validate_dataset = CustomDataset(validate_folder)
    test_dataset = CustomDataset(test_folder)

    return train_dataset, validate_dataset, test_dataset


train_dataset, validate_dataset, test_dataset = create_datasets(data_path, dataset_name)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None)
val_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None)

#%%
#Data processing
# Verify if data is normalized
print('check a random image to verify if data is normalized:')
print('min:',train_dataloader.dataset.data[0].min(),' max:', train_dataloader.dataset.data[0].max())

def plot_random_samples(train_dataloader, num_samples=5,od=args.output_dir):
    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Sélectionner un échantillon aléatoire
        idx = random.randint(0, len(train_dataloader.dataset) - 1)
        image = train_dataloader.dataset.data[idx]

        label= train_dataloader.dataset.labels[idx]

        # Afficher l'image dans la première colonne
        axs[i, 0].imshow(image[0], aspect='auto', cmap='gray')
        axs[i, 0].set_title(f'Shot Gather {i + 1}')
        axs[i, 0].set_xlabel('Distance (m)')
        axs[i, 0].set_ylabel('Time (sample)')

        # Afficher le label dans la deuxième colonne
        axs[i, 1].plot(label, range(len(label)))
        axs[i, 1].invert_yaxis()
        axs[i, 1].set_xlabel('Vs (m/s)')
        axs[i, 1].set_ylabel('Depth (m)')
        axs[i, 1].set_title(f'Vs Depth {i + 1}')

    plt.tight_layout()
    plt.savefig(f'figures/{od}/random_samples.png',format='png')
    plt.close()


# Afficher des échantillons aléatoires
plot_random_samples(train_dataloader, num_samples=5,)


out_dim=len(train_dataloader.dataset.labels[0])
print('Output dimension: ',out_dim)
class PretrainedViT(nn.Module):
    def __init__(self, out_dim=out_dim, pretrained_model_name='google/vit-base-patch16-224-in21k'):
        super(PretrainedViT, self).__init__()

        # Charger le modèle ViT pré-entraîné de Hugging Face
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        emb_dim = self.vit.config.hidden_size  # Taille des embeddings du modèle pré-entraîné

        # Geler les poids du modèle pré-entraîné
        for param in self.vit.parameters():
            param.requires_grad = False

        # Tête de régression
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),  # Normalisation
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, img):
        # Passer les images dans le modèle pré-entraîné pour obtenir les embeddings
        outputs = self.vit(pixel_values=img)

        # Extraire l'embedding du token [CLS]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Passer l'embedding du token [CLS] dans la tête de régression
        output = self.head(cls_embedding)

        return output

# Display the model architecture
print('MODEL ARCHITECTURE:')
model = PretrainedViT()
print(model)


def training(device=None,lr=0.0005,nepochs=101):
    print('\nTRAINING LOOP:')
    # GPU or CPU
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        device = ("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        device = device
    print('Device: ', device)
    model = PretrainedViT().to(device)

    # Initializations
    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # Optimizer for training, and learning rate
    print('Optimizer: AdamW')
    criterion = nn.MSELoss()  # Loss function for regression
    print('Loss function: MSE')
    train_losses = []
    val_losses = []
    epochs_count_train = []
    epochs_count_val = []
    print(f'Number of epochs: {nepochs}')
    # Training loop
    for epoch in range(nepochs): # Change this to train longer/shorter
        #print('epoch:',epoch)
        epoch_losses = []
        model.train()
        for step in range(len(train_dataloader)):
            inputs= torch.tensor(train_dataloader.dataset.data[step], dtype=torch.float32)
            inputs= inputs.unsqueeze(0)
            labels = torch.tensor(train_dataloader.dataset.labels[step], dtype=torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print('shape inputs',inputs.shape)
            #print('shape outputs:',outputs.shape,'shape labels:' ,labels.shape)
            # print('outputs:',outputs)
            loss = criterion(outputs.float(), labels.permute(1,0).float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        # Print epoch loss, every 5 epochs
        if epoch % 5 == 0:
            print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
            train_losses.append(np.mean(epoch_losses));
            epochs_count_train.append(epoch)
            epoch_losses = []
            # Something was strange when using this?
            # model.eval()
            # Validation loop, every 5 epochs
            for step in range(len(val_dataloader)):
                inputs = torch.tensor(val_dataloader.dataset.data[step], dtype=torch.float32)
                labels= torch.tensor(val_dataloader.dataset.labels[step], dtype=torch.float32)
                inputs= inputs.unsqueeze(0)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                #print('shape outputs:',outputs.shape,'shape labels:' ,labels.shape)
                loss = criterion(outputs, labels.permute(1,0))
                epoch_losses.append(loss.item())
            val_losses.append(np.mean(epoch_losses));
            epochs_count_val.append(epoch)
            print(f">>> Epoch {epoch} validation loss: ", np.mean(epoch_losses))

    return train_losses, val_losses, epochs_count_train, epochs_count_val, device,model

# Training
time0= time.time()
train_losses, val_losses, epochs_count_train, epochs_count_val,device,model=training(nepochs=args.nepochs,lr=args.lr)
training_time=time.time()-time0
print('\nTraining time:',training_time)

def learning_curves(epochs_count_train, train_losses, epochs_count_val, val_losses,od=args.output_dir):
    # Plot training and validation losses
    plt.plot(epochs_count_train, train_losses, label='Training Loss')
    plt.plot(epochs_count_val, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss MSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig(f'figures/{od}/learning_curves.png',format='png')
    plt.close()

# Display learning curves
learning_curves(epochs_count_train, train_losses, epochs_count_val, val_losses)

#%%
def evaluate(model=model, test_dataloader=test_dataloader, device=device):
    print('\nEVLUATION:')
    # Evaluate the model
    model.eval()  # Set the model to evaluation mode

    # Initializations
    all_images = []
    all_predictions = []
    all_labels = []

    # Loop through the test set
    for step in range(len(test_dataloader)):
        inputs= torch.tensor(test_dataloader.dataset.data[step], dtype=torch.float32)
        labels= torch.tensor(test_dataloader.dataset.labels[step], dtype=torch.float32)
        inputs= inputs.unsqueeze(0)
        inputs, labels = inputs.to(device), labels.to(device)  # Move to device
        all_images.append(inputs.cpu().numpy()) # Transfer images to CPU
        with torch.no_grad():  # No need to calculate gradients
            outputs = model(inputs)  # Make predictions
        all_predictions.append(outputs.detach().cpu().numpy())  # Transfer predictions to CPU
        all_labels.append(labels.cpu().numpy().T)  # Transfer labels to CPU

    # Concatenate all images, predictions, and labels
    all_images = np.concatenate(all_images, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate the mean squared error
    mse = np.mean((all_predictions - all_labels) ** 2)

    # Display the mean squared error
    print("Mean Squared Error on Test Set:", mse)
    return all_images, all_predictions, all_labels


# Evaluate the model
all_images, all_predictions, all_labels = evaluate(model, test_dataloader, device)


#%%
# Display some predictions
def visualize_predictions(all_predictions=all_predictions,test_dataloader=test_dataloader,num_samples=4,od=args.output_dir):
    # Select random images
    if num_samples > len(all_predictions):
        num_samples = len(all_predictions)
    indices = random.sample(range(len(all_predictions)), num_samples)

    fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        print('You must display at least 2 samples')

    else:
        for i in range(num_samples):
            # Sélectionner un indice aléatoire
            idx = random.randint(0, len(test_dataloader.dataset) - 1)

            # Récupérer les données et les étiquettes à l'indice sélectionné
            data = test_dataloader.dataset.data[idx]

            label = test_dataloader.dataset.labels[idx]
            prediction = all_predictions[idx]

            # Afficher l'image dans la première colonne
            axs[i, 0].imshow(data[0], aspect='auto', cmap='gray')
            axs[i, 0].set_title(f'Shot Gather {idx}')
            axs[i, 0].set_xlabel('Distance (m)')
            axs[i, 0].set_ylabel('Time (sample)')

            # Afficher le label dans la deuxième colonne
            axs[i, 1].plot(label, range(len(label)))
            axs[i, 1].plot(prediction.reshape(-1), range(len(prediction)))
            axs[i, 1].invert_yaxis()
            axs[i, 1].set_xlabel('Vs (m/s)')
            axs[i, 1].set_ylabel('Depth (m)')
            axs[i, 1].set_title(f'Vs Depth {idx}')

    plt.tight_layout()
    plt.savefig(f'figures/{od}/predictionsVSlabels.png',format='png')
    plt.close()


# Afficher quelques images avec leurs étiquettes et prédictions
visualize_predictions(num_samples=5)