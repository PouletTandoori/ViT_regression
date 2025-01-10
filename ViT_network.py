#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from einops.layers.torch import Rearrange
from torch import Tensor
from einops import repeat
from torch.utils.data import DataLoader, Dataset
from random import random
import torch.optim as optim
import argparse
import time
from Utilities import *
from data_augmentation import *
import numpy as np



#%%
#define parser
parser = argparse.ArgumentParser(description='ViT MASW')

parser.add_argument('--dataset_name','-data', type=str,default='Dataset1Dsimple', required=False,
                    help='Name of the dataset to use, choose between \n Dataset1Dbig \n Dataset1Dsmall \n TutorialDataset')
parser.add_argument('--nepochs','-ne', type=int, default=1, required=False,help='number of epochs for training')
parser.add_argument('--lr','-lr', type=float, default=0.0005, required=False,help='learning rate')
parser.add_argument('--batch_size','-bs', type=int, default=1, required=False,help='batch size')
parser.add_argument('--output_dir','-od', type=str, default='ViT_debug', required=False,help='output directory for figures')
parser.add_argument('--data_augmentation','-aug', type=bool, default=False, required=False,help='data augmentation')
parser.add_argument('--decay','-dec', type=float, default=0, required=False,help='weight decay')
parser.add_argument('--loss','-lo', type=str, default='Jeff_Loss', required=False, help='loss function to use, there is MSE, NMSELoss, NMSERL1Loss')
parser.add_argument('--num_heads','-nh', type=int, default=12, required=False, help='number of heads in the transformer')
parser.add_argument('--num_layers','-nl', type=int, default=12, required=False, help='number of layers in the transformer')

args = parser.parse_args()

#Paths
data_path='/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
dataset_name = args.dataset_name
files_path=os.path.join(data_path,dataset_name)

train_folder=glob.glob(f'{files_path}/train/*')
validate_folder=glob.glob(f'{files_path}/validate/*')
test_folder=glob.glob(f'{files_path}/test/*')

setup_directories(name=args.output_dir)

# Verify the dataset
print('VERIFYING DATASET')
bad_files = verify([train_folder,validate_folder,test_folder])
#%%
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
                GaussianNoise(mean=0, std=0.05),
                TraceShift2(shift_ratio=0.01, contiguous_ratio=0.2),
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


#%%
print('INFORMATION ABOUT THE DATASET:')
print('Image size: ',train_dataloader.dataset.data[0].shape)
channels,img_height, img_width = train_dataloader.dataset.data[0].shape
print('Label size: ',train_dataloader.dataset.labels[0].shape)
lab_size= train_dataloader.dataset.labels[0].shape[0]
out_dim=lab_size
print('Output dimension: ',out_dim)
print('\n')

#%%
#Data processing
# Verify if data is normalized
print('check a random image to verify if data is normalized:')
print('min:',train_dataloader.dataset.data[0].min(),' max:', train_dataloader.dataset.data[0].max())


#%%
def plot_random_samples(train_dataloader, num_samples=5,od=args.output_dir):
    '''
    Display random samples, shot gathers and Vs profiles
    :param train_dataloader: dataloader
    :param num_samples: number of samples to display
    :param od: output directory
    :return: None, save the figure in the output directory
    '''
    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Sélectionner un échantillon aléatoire
        idx = random.randint(0, len(train_dataloader.dataset) - 1)
        sample = train_dataloader.dataset[idx]
        image = sample['data']
        label = sample['label']
        #print('shape image=', image.shape)

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

# Display random samples
plot_random_samples(train_dataloader, num_samples=5)


#%%
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
    #emb_dim was 36 for nh=12
    def __init__(self, ch=1, img_height=img_height,img_width=img_width, emb_dim=int(args.num_heads * 3),
                n_layers=args.num_layers, out_dim=out_dim, dropout=0.1, heads=args.num_heads):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.img_height= img_height
        self.img_width = img_width
        self.patch_height= img_height // 5
        self.patch_width =  img_width // 2
        print(f'patch size: {self.patch_height}x{self.patch_width}')
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

# Display the model architecture
print('MODEL ARCHITECTURE:')
model = ViT()
print(model)

#print the amount of unfrozen parameters:
unfrozen_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of unfrozen parameters:',unfrozen_parameters)

def training(device=None, lr=args.lr, nepochs=args.nepochs, early_stopping_patience=3,alpha=0.02,beta=0.1,l1param=0.05,vmaxparam=0.2):
    print('\nTRAINING LOOP:')
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Device: ', device)
    model = ViT().to(device)

    # Initializations
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer for training
    print(f'Optimizer:{optimizer} Learning rate: {lr}')

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    epochs_count_train = []
    epochs_count_val = []

    print(f'Number of epochs: {nepochs}')
    for epoch in range(nepochs):
        epoch_losses = []
        model.train()
        for step in range(len(train_dataloader)):
            sample = train_dataloader.dataset[step]
            inputs = sample['data']
            labels = sample['label']
            inputs = inputs.unsqueeze(0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            criterion = JeffLoss(alpha=alpha, beta=beta, l1=l1param, v_max=vmaxparam)
            loss = criterion(outputs.float(), labels.permute(1, 0).float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % 5 == 0:
            mean_train_loss = np.mean(epoch_losses)
            print(f">>> Epoch {epoch} train loss: ", mean_train_loss)
            train_losses.append(mean_train_loss)
            epochs_count_train.append(epoch)

            # Validation loop
            model.eval()
            val_losses_epoch = []
            with torch.no_grad():
                for step in range(len(val_dataloader)):
                    sample = val_dataloader.dataset[step]
                    inputs = sample['data']
                    labels = sample['label']
                    inputs = inputs.unsqueeze(0)
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.permute(1, 0).float())
                    val_losses_epoch.append(loss.item())

            mean_val_loss = np.mean(val_losses_epoch)
            val_losses.append(mean_val_loss)
            epochs_count_val.append(epoch)
            print(f">>> Epoch {epoch} validation loss: ", mean_val_loss)

            # Early stopping check
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_without_improvement = 0
                # Save the best model here if needed
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    return train_losses, val_losses, epochs_count_train, epochs_count_val, device, model

# Training
time0= time.time()
train_losses, val_losses, epochs_count_train, epochs_count_val,device,model=training(nepochs=args.nepochs,lr=args.lr)
training_time=time.time()-time0
print('\nTraining time:',training_time)
#%%
# Display learning curves
def learning_curves(epochs_count_train, train_losses, epochs_count_val, val_losses,od=args.output_dir):
    # Plot training and validation losses
    plt.plot(epochs_count_train, train_losses, label='Training Loss')
    plt.plot(epochs_count_val, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss MSE')
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig(f'figures/{od}/learning_curves.png',format='png')
    plt.close()

# Display learning curves
learning_curves(epochs_count_train, train_losses, epochs_count_val, val_losses)

#%%
def evaluate(model=model, test_dataloader=test_dataloader, device=device,alpha=0.02,beta=0.1,l1param=0.05,vmaxparam=0.2):
    print('\nEVALUATION:')
    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    #loss
    criterion = JeffLoss(alpha=alpha, beta=beta, l1=l1param, v_max=vmaxparam)

    # Initializations
    all_images = []
    all_predictions = []
    all_labels = []

    # Loop through the test set
    for step in range(len(test_dataloader)):
        sample = test_dataloader.dataset[step]
        inputs = sample['data']
        labels = sample['label']
        #inputs= torch.tensor(test_dataloader.dataset.data[step], dtype=torch.float32)
        #labels= torch.tensor(test_dataloader.dataset.labels[step], dtype=torch.float32)
        inputs= inputs.unsqueeze(0)
        inputs, labels = inputs.to(device), labels.to(device)  # Move to device
        all_images.append(inputs.cpu().numpy()) # Transfer images to CPU
        with torch.no_grad():  # No need to calculate gradients
            outputs = model(inputs)  # Make predictions
            #calculate error:
            loss = criterion(outputs.float(), labels.permute(1,0).float())
        all_predictions.append(outputs.detach().cpu().numpy())  # Transfer predictions to CPU
        all_labels.append(labels.cpu().numpy().T)  # Transfer labels to CPU

    # Concatenate all images, predictions, and labels
    all_images = np.concatenate(all_images, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    #mean loss:
    loss_np = loss.cpu().detach().numpy()
    mean_loss = np.mean(loss_np)

    # Display the mean squared error
    print("Loss on Test Set:", mean_loss)
    return all_images, all_predictions, all_labels,mean_loss

# Evaluate the model
all_images, all_predictions, all_labels,mean_loss = evaluate()

#%%
# Display some predictions
def visualize_predictions(all_predictions=all_predictions,test_dataloader=test_dataloader,num_samples=4,od=args.output_dir,bs=args.batch_size):
    # Select random images
    if num_samples > len(all_predictions):
        num_samples = len(all_predictions)

    fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        print('You must display at least 2 samples')

    else:
        for i in range(num_samples):
            # Sélectionner un indice aléatoire
            idx = random.randint(0, int(len(test_dataloader.dataset)/bs) - 1)

            # Récupérer les données et les étiquettes à l'indice sélectionné
            sample = test_dataloader.dataset[idx]
            data = sample['data']
            label = sample['label']
            prediction = all_predictions[idx]

            #convert the image from grid points into real units
            dt = 0.00002*100 #dt * resampling
            time_vector = np.arange(data.shape[1]) * dt
            #print('time vector len:',len(time_vector))
            nb_traces = data.shape[2]
            #print('nb traces:',nb_traces)
            dz = 0.25
            #print('label shape:',label.shape)
            depth_vector = np.arange(label.shape[0]) * dz


            # Afficher l'image dans la première colonne
            axs[i, 0].imshow(data[0], aspect='auto', cmap='gray',extent=[0,nb_traces,time_vector[-1],time_vector[0]])
            axs[i, 0].set_title(f'Shot Gather {idx}')
            axs[i, 0].set_xlabel('Traces')
            axs[i, 0].set_ylabel('Time (s)')

            # Afficher le label dans la deuxième colonne
            axs[i, 1].plot(label, depth_vector)
            axs[i, 1].plot(prediction.reshape(-1), depth_vector)
            axs[i, 1].invert_yaxis()
            axs[i, 1].set_xlabel('Vs (m/s)')
            axs[i, 1].set_ylabel('Depth (m)')
            axs[i, 1].set_title(f'Vs Depth {idx}')

    plt.tight_layout()
    plt.savefig(f'figures/{od}/predictionsVSlabels.png',format='png')
    plt.close()



# Afficher quelques images avec leurs étiquettes et prédictions
visualize_predictions(num_samples=5)

#save runs infos
#display on a signgle image all informations about the current model
main_path= os.path.abspath(__file__)
display_run_info(model=model,od=args.output_dir,args=args,metrics=mean_loss,training_time=training_time,main_path=main_path,best_params=None,nb_param=unfrozen_parameters)


#save the model parameters
torch.save(model.state_dict(), f'figures/{args.output_dir}/model.pth')