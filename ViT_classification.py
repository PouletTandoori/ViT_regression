import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from random import random
from transformers import ViTModel
import torch.optim as optim
import argparse
import time
from Utilities import *
import h5py as h5
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PhaseShiftMethod import *
from data_augmentation import *

#define parser
parser = argparse.ArgumentParser(description='ViT with Transfer Learning for MASW')

parser.add_argument('--dataset_name','-data', type=str,default='Dataset1Dsimple', required=False,
                    help='Name of the dataset to use, choose between \n Dataset1Dsmall \n Dataset1Dbig \n TutorialDataset')
parser.add_argument('--nepochs','-ne', type=int, default=1, required=False,help='number of epochs for training')
parser.add_argument('--lr','-lr', type=float, default=0.0001, required=False,help='learning rate')
parser.add_argument('--batch_size','-bs', type=int, default=2, required=False,help='batch size')
parser.add_argument('--output_dir','-od', type=str, default='ViT_classification_debug', required=False,help='output directory for figures')
parser.add_argument('--data_augmentation','-aug', type=bool, default=True, required=False,help='data augmentation')
parser.add_argument('--decay','-dec', type=float, default=0, required=False,help='weight decay')
parser.add_argument('--loss','-lo', type=str, default='CategoricalCrossEntropy', required=False, help='loss function to use, there is CategoricalCrossEntropy')

parser.add_argument('--dispersion_image','-disp', type=bool, default=False, required=False, help='Use the dispersion image instead of shotgather')
parser.add_argument('--k','-k', type=float, default=1.2, required=False, help='factor for the one hot encoding of the labels')
parser.add_argument('--lambda_entropy','-lam', type=float, default=0.1, required=False, help='lambda for the entropy in the loss function')


args = parser.parse_args()

# Paths
data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
dataset_name = args.dataset_name
files_path = os.path.join(data_path, dataset_name)

train_folder = glob.glob(f'{files_path}/train/*')
validate_folder = glob.glob(f'{files_path}/validate/*')
test_folder = glob.glob(f'{files_path}/test/*')

setup_directories(name=args.output_dir)

# Verify the dataset
print('VERIFYING DATASET')
bad_files = verify([train_folder,validate_folder,test_folder])



if args.dispersion_image:
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

        train_dataset = CustomDataset(train_folder)
        validate_dataset = CustomDataset(validate_folder)
        test_dataset = CustomDataset(test_folder)

        return train_dataset, validate_dataset, test_dataset

else:
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
                    # Prendre seulement la deuxième moitié = composante Z
                    inputs = inputs[:, int(inputs.shape[1] / 2):]
                    labels_data = h5file['vsdepth'][:]

                    for i in range(labels_data.shape[0]):
                        labels_data[i][0] = int(labels_data[i][0])

                    # Normaliser les données
                    inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))

                    # Reshape data
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
                    labels.append(labels_data[:])

            data = np.array(data)

            # Aconvert labels to integers
            labels = np.array(labels).astype(int)
            return data, labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            inputs = self.data[idx]
            labels = self.labels[idx]

            # Convertir les inputs et labels en Tensors
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)  # Utiliser dtype=torch.long pour des entiers

            sample = {'data': inputs, 'label': labels}

            if self.transform:
                sample['data'] = self.transform(sample['data'])

            return sample

    def create_datasets(data_path, dataset_name):
        train_folder = glob.glob(os.path.join(data_path, dataset_name, 'train', '*'))
        validate_folder = glob.glob(os.path.join(data_path, dataset_name, 'validate', '*'))
        test_folder = glob.glob(os.path.join(data_path, dataset_name, 'test', '*'))
        if args.data_augmentation == True:
            # define augmentations
            view_transform = transforms.Compose(
                [
                    TraceShift(),
                    GaussianNoise(mean=0, std=0.1),
                    DeadTraces(),
                    MissingTraces()

                ]
            )
            train_dataset = CustomDataset(train_folder, transform=view_transform)
        else:

            train_dataset = CustomDataset(train_folder)
        validate_dataset = CustomDataset(validate_folder)
        test_dataset = CustomDataset(test_folder)

        return train_dataset, validate_dataset, test_dataset


train_dataset, validate_dataset, test_dataset = create_datasets(data_path, dataset_name)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None)
val_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=None)

#%%
# Verify if data is normalized
print('check a random image to verify if data is normalized:')
print('min:',train_dataloader.dataset.data[0].min(),' max:', train_dataloader.dataset.data[0].max())

#calculate the number of classes and the output length
#1 for one label, do the OneHotEncoding
example_label=test_dataloader.dataset.labels[0]
print('shape of example label:', np.shape(example_label))
#2 apply the OneHotEncoding to obtain the newlabel matrix
_,_,example_label_OHE=OneHotEncode_IrregularBins2(example_label, k=1.1, plot=True,figure_path=f'figures/{args.output_dir}/example_k11')
_,_,example_label_OHE=OneHotEncode_IrregularBins2(example_label, k=1.3, plot=True,figure_path=f'figures/{args.output_dir}/example_k13')
_,_,example_label_OHE=OneHotEncode_IrregularBins2(example_label, k=args.k, plot=True,figure_path=f'figures/{args.output_dir}/example_k{int(args.k*10)}')
#3 obtain the dimensions of example_label_OHE
out_dim, nb_classes = np.shape(example_label_OHE)
print('Number of classes:', nb_classes)
print('Output dimensions:', out_dim)

def plot_random_samples(train_dataloader, num_samples=5,od=args.output_dir):

    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Select a random index
        idx = random.randint(0, len(train_dataloader.dataset) - 1)

        sample = train_dataloader.dataset[idx]
        image = sample['data']
        label = sample['label']

        # Display the image in the first column
        axs[i, 0].imshow(image[0], aspect='auto', cmap='gray')
        axs[i, 0].set_title(f'Shot Gather {i + 1}')
        axs[i, 0].set_xlabel('Distance (m)')
        axs[i, 0].set_ylabel('Time (sample)')

        # Display the label in the second column
        axs[i, 1].plot(label, range(len(label)))
        axs[i, 1].invert_yaxis()
        axs[i, 1].set_xlabel('Vs (m/s)')
        axs[i, 1].set_ylabel('Depth (m)')
        axs[i, 1].set_title(f'Vs Depth {i + 1}')

    plt.tight_layout()
    plt.savefig(f'figures/{od}/random_samples.png',format='png')
    plt.close()


# Display random samples
if args.dispersion_image == False:
    plot_random_samples(train_dataloader, num_samples=5)



# prepare labels, create classes:
train_dataloader, val_dataloader, test_dataloader, max_label = PrepareClasses2(train_dataloader, val_dataloader, test_dataloader)

print('Output dimensions are matrices of size (label length * number of classes):')
print('Output dimensions:', out_dim, '*', nb_classes)

class PretrainedViT(nn.Module):
    '''
    For pretrained model, you can choose between: 'google/vit-base-patch16-224-in21k', 'google/vit-base-patch32-224-in21k',

    '''
    def __init__(self, out_dim=out_dim,nb_classes=nb_classes, pretrained_model_name='google/vit-base-patch16-224-in21k'):
        super(PretrainedViT, self).__init__()

        # Load the pretrained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        emb_dim = self.vit.config.hidden_size  # Taille des embeddings du modèle pré-entraîné

        # Freeze the weights of the ViT model
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze the weights of the last layer of the ViT model
        for param in self.vit.encoder.layer[-1].parameters():
            param.requires_grad = True

        # Define the new head
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),  # Normalization
            nn.Linear(emb_dim, 256),  # Intermediate linear layer
            nn.ReLU(),  # Activation non-linéaire
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, int(out_dim) * int(nb_classes))  # Output layer for nb_classes classes
        )

        # Reshape the output to have shape (batch_size, 200, nb_classes)
        self.output_shape = (int(out_dim), int(nb_classes))

    def forward(self, img):
        # Obtain embeddings
        outputs = self.vit(pixel_values=img)

        # Extract the embedding of the [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Use the embedding to make predictions
        output = self.head(cls_embedding)

        # Reshape the output to (batch_size, out_dim, nb_classes)
        output = output.view(-1, *self.output_shape)


        return output


# Display the model architecture
print('MODEL ARCHITECTURE:')
model = PretrainedViT()
print(model)

#print the amount of unfrozen parameters:
unfrozen_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of unfrozen parameters:',unfrozen_parameters)


def training(device=None, lr=0.0005, nepochs=101, patience=2, min_delta=0):
    print('\nTRAINING LOOP:')
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    model = PretrainedViT().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.decay)
    if args.loss=='CategoricalCrossEntropy':
        #criterion = CategoricalCrossEntropy2(lambda_entropy=args.lambda_entropy,num_classes=nb_classes*out_dim)
        criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    train_losses = []
    val_losses = []
    epochs_count_train = []
    epochs_count_val = []

    if args.dispersion_image:
        fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name)

    for epoch in range(nepochs):
        model.train()
        epoch_losses = []
        for step in range(len(train_dataloader)):
            sample = train_dataloader.dataset[step]
            inputs = sample['data']
            if args.dispersion_image:
                disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
                disp,_,_ = prepare_disp_for_NN(disp)
                disp = disp.unsqueeze(0).to(device)
            labels = sample['label']
            #print('shape of labels during training:',np.shape(labels))
            _, _, labels = OneHotEncode_IrregularBins2(labels.numpy(), k=args.k, plot=False)
            inputs = inputs.unsqueeze(0).to(device)
            #transform labels into torch tensor
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            if args.dispersion_image:
                outputs = model(disp)
            else:
                outputs = model(inputs)

            outputs = outputs.to(device)
            if args.loss=='WeithgtedCrossEntropy':
                loss = criterion(outputs.permute(0,2,1).float(), labels.permute(0,2,1).float())
            elif args.loss=='CategoricalCrossEntropy':
                #print('shape of outputs:',outputs.shape)
                #print('shape of labels:',labels.shape)
                #loss = 1/(out_dim)*criterion(outputs.squeeze(0).float(), labels.float())
                loss=normalized_crossentropy_loss(outputs.squeeze(0).float(), labels.float(),nb_classes)
            loss.backward()
            # apply softmax
            if args.loss == 'CategoricalCrossEntropy':
                outputs = torch.nn.functional.softmax(outputs, dim=2)
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % 5 == 0:
            print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
            train_losses.append(np.mean(epoch_losses))
            epochs_count_train.append(epoch)
            epoch_losses = []

            model.eval()
            val_epoch_losses = []
            with torch.no_grad():
                for step in range(len(val_dataloader)):
                    sample = val_dataloader.dataset[step]
                    inputs = sample['data']
                    if args.dispersion_image:
                        disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
                        disp,_,_ = prepare_disp_for_NN(disp)
                        disp = disp.unsqueeze(0).to(device)
                    labels = sample['label']
                    _,_,labels = OneHotEncode_IrregularBins2(labels.numpy(),k=args.k, plot=False)
                    inputs = inputs.unsqueeze(0).to(device)
                    labels = torch.tensor(labels).to(device)
                    if args.dispersion_image:
                        outputs = model(disp)
                    else:
                        outputs = model(inputs)
                    #loss = 1/(out_dim)*criterion(outputs.squeeze(0).float(), labels.float())
                    loss = normalized_crossentropy_loss(outputs.squeeze(0).float(), labels.float(), nb_classes)
                    val_epoch_losses.append(loss.item())
            val_loss = np.mean(val_epoch_losses)
            val_losses.append(val_loss)
            epochs_count_val.append(epoch)
            print(f">>> Epoch {epoch} validation loss: ", val_loss)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    return train_losses, val_losses, epochs_count_train, epochs_count_val, device, model

# Training
time0= time.time() #initial time
train_losses, val_losses, epochs_count_train, epochs_count_val,device,model=training(nepochs=args.nepochs,lr=args.lr)
training_time=time.time()-time0 #training time
print('\nTraining time:',training_time)

def learning_curves(epochs_count_train, train_losses, epochs_count_val, val_losses, od=args.output_dir):
    # Créer la figure avec une taille adaptée pour LaTeX
    fig, ax = plt.subplots(figsize=(7.5, 5))  # Largeur de 7.5 pouces pour s'adapter à une colonne

    # Tracer les courbes de perte
    ax.plot(epochs_count_train, train_losses, label='Training Loss', linewidth=1.5)
    ax.plot(epochs_count_val, val_losses, label='Validation Loss', linewidth=1.5)

    # Ajuster les étiquettes et le titre
    ax.set_xlabel('Epoch', fontsize=11)  # Taille de la police ajustée
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_yscale('log')  # Echelle logarithmique pour l'axe y
    ax.set_title('Learning Curves', fontsize=13)  # Titre avec une taille de police adaptée

    # Ajouter une légende
    ax.legend(fontsize=12)

    # Optimiser l'agencement
    plt.tight_layout()

    # Sauvegarder l'image en haute résolution
    plt.savefig(f'figures/{od}/learning_curves.pdf', format='pdf', dpi=300)  # Format PDF pour une qualité optimale
    plt.close()

# Display learning curves
learning_curves(epochs_count_train, train_losses, epochs_count_val, val_losses)

#%%
def compute_metrics(predictions, labels, top_k=(1, 3, 5)):
    # Flatten predictions and labels
    labels = labels.view(-1, labels.shape[-1])
    true_classes = torch.argmax(labels, dim=-1)

    # Get top-k predictions
    top_k_preds = torch.topk(predictions, max(top_k), dim=-1).indices
    top_k_preds = top_k_preds.view(-1, max(top_k))  # Flatten

    # Initialize a dictionary for accuracy at top-1, top-3, and top-5
    accuracy_dict = {}

    for k in top_k:
        # Check if the true class is in the top-k predicted classes
        correct_top_k = torch.any(top_k_preds[:, :k] == true_classes.unsqueeze(1), dim=1)
        accuracy_top_k = correct_top_k.sum().item() / len(true_classes)
        accuracy_dict[f'top-{k} accuracy'] = accuracy_top_k

    return accuracy_dict


def evaluate(model=model, test_dataloader=test_dataloader, device=device):
    print('\nEVALUATION:')
    # Evaluate the model
    model.eval()  # Set the model to evaluation mode

    # Initializations
    all_images = []
    all_disp = []
    all_predictions = []
    all_labels = []
    global_loss=0

    if args.dispersion_image:
        fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name)

    # Loop through the test set
    for step in range(len(test_dataloader)):
        sample = test_dataloader.dataset[step]
        inputs = sample['data']
        if args.dispersion_image:
            disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
            disp,shape1,shape2 = prepare_disp_for_NN(disp)
            disp = disp.unsqueeze(0).to(device)
        labels = sample['label']
        # one hot encoding
        _, _, labels = OneHotEncode_IrregularBins2(labels.numpy(), k=args.k, plot=False)
        inputs= inputs.unsqueeze(0)
        inputs, labels = inputs.to(device), torch.tensor(labels).to(device) # Move to device
        all_images.append(inputs.cpu().numpy()) # Transfer images to CPU
        with torch.no_grad():  # No need to calculate gradients
            if args.dispersion_image:
                outputs = model(disp)
            else:
                outputs = model(inputs)  # Make predictions

            if args.loss == 'CategoricalCrossEntropy':
                outputs_prob = torch.nn.functional.softmax(outputs, dim=2)
            if not torch.allclose(outputs_prob.sum(dim=2), torch.ones(outputs.shape[0], outputs.shape[1]).to(device)):
                print('Test: Sum of probabilities is not 1:', outputs.sum(dim=2))
            # apply softmax
        all_predictions.append(outputs_prob.detach().cpu().numpy())  # Transfer predictions to CPU
        all_labels.append(labels.cpu().numpy())  # Transfer labels to CPU
        if args.dispersion_image:
            all_disp.append(disp.cpu().numpy())

        #calculate loss
        if args.loss == 'CategoricalCrossEntropy':
            #criterion = CategoricalCrossEntropy2(lambda_entropy=args.lambda_entropy,num_classes=nb_classes*out_dim)
            criterion = nn.CrossEntropyLoss()
        #loss = 1/(out_dim)* criterion(outputs.squeeze(0).float(), labels.float())
        loss = normalized_crossentropy_loss(outputs.squeeze(0).float(), labels.float(), nb_classes)
        global_loss+=loss.item()

    # Concatenate all images, predictions, and labels
    all_images = np.concatenate(all_images, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    for key, value in metrics.items():
        print(f'{key}: {100 * value:.2f}%\n')
    print("CrossEntropyLoss on Test Set:", global_loss/len(test_dataloader))

    if args.dispersion_image:
        return all_images, all_predictions, all_labels, all_disp, metrics,shape1,shape2
    else:
        return all_images, all_predictions, all_labels, metrics

# Evaluate the model
if args.dispersion_image:
    all_images, all_predictions, all_labels, all_disp, metrics,shape1,shape2 = evaluate(model, test_dataloader, device)
else:
    all_images, all_predictions, all_labels, metrics = evaluate(model, test_dataloader, device)

#%%
# Display some predictions
def visualize_predictions(all_predictions=all_predictions, all_dispersion='None', shape1='None', shape2='None', test_dataloader=test_dataloader, num_samples=5, od=args.output_dir):
    # Vérifier le nombre d'échantillons
    if num_samples > len(all_predictions):
        num_samples = len(all_predictions)

    if args.dispersion_image:
        all_dispersion = all_disp

        # Paramètres d'acquisition
        fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name)

        fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(7.5, 3.75 * num_samples))  # Taille ajustée

        if num_samples == 1:
            print('You must display at least 2 samples')
        else:
            for i in range(num_samples):
                sample = test_dataloader.dataset[i]
                data = sample['data']
                original_label = sample['label']

                # Utiliser OneHotEncode_IrregularBins pour encoder le label
                fig_path = f'figures/{od}/'
                x, y, label = OneHotEncode_IrregularBins2(original_label.numpy(), k=args.k, plot=False, figure_path=f'{fig_path}{i}_')
                y = y / y[-1] * 50
                prediction = all_predictions[i]
                disp = all_dispersion[i]

                # Conversion en unités réelles
                time_vector = np.linspace(0, 1.5, data.shape[0])
                nb_traces = data.shape[1]
                dz = 0.25
                max_speed = 5000
                depth_vector = np.logspace(0, np.log10(max_speed), int(round(label.shape[0])))

                # Afficher l'image de dispersion dans la première colonne
                axs[i, 0].imshow(disp[0][0][:shape1, :shape2], aspect='auto')
                axs[i, 0].set_xticks(np.linspace(0, disp.shape[1] - 1, 5).astype(int))
                axs[i, 0].set_xticklabels(np.round(np.linspace(np.min(c), np.max(c), 5)).astype(int))
                axs[i, 0].set_title(f'Dispersion Image {i}', fontsize=11)
                axs[i, 0].set_xlabel('Phase Velocity (m/s)', fontsize=10)
                axs[i, 0].set_ylabel('Frequency (Hz)', fontsize=10)

                # Afficher le label et la prédiction dans la deuxième colonne
                axs[i, 1].pcolormesh(x, y, prediction, shading='auto')
                f = interp1d(np.arange(len(label)), label, kind='linear', axis=0)
                label = f(np.linspace(0, len(label) - 1, 51))
                axs[i, 1].plot(label, np.arange(len(label)), label='Label', linewidth=2, color='red')
                axs[i, 1].set_xlim(0, max_speed)
                axs[i, 1].set_ylim(0, y[-1])
                axs[i, 1].invert_yaxis()
                axs[i, 1].set_xlabel('Vs (m/s)', fontsize=10)
                axs[i, 1].set_ylabel('Depth (m)', fontsize=10)
                axs[i, 1].set_title(f'Vs classification {i}', fontsize=11)

                # Ajouter une barre de couleurs
                cbar = fig.colorbar(axs[i, 1].collections[0], ax=axs[i, 1])
                cbar.set_label('Probability', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'figures/{od}/predictionsVSlabels.pdf', format='pdf', dpi=300)  # Sauvegarde en PDF avec 300 dpi
        plt.close()

    else:
        fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(7.5, 3.75 * num_samples))  # Taille ajustée

        if num_samples == 1:
            print('You must display at least 2 samples')
        else:
            for i in range(num_samples):
                sample = test_dataloader.dataset[i]
                data = sample['data']
                label = sample['label']

                # Encodage OneHot
                fig_path = f'figures/{od}/'
                x, y, label_OHE = OneHotEncode_IrregularBins2(label.numpy(), k=args.k, plot=False, figure_path=f'{fig_path}{i}_')
                y = y / y[-1] * 50
                prediction = all_predictions[i]

                # Conversion en unités réelles
                time_vector = np.linspace(0, 1.5, data.shape[0])
                nb_traces = 96 if args.dataset_name in ['Dataset1Dsimple', 'Dataset1Dhuge_96tr'] else 48
                max_speed = 5000
                depth_vector = np.logspace(0, np.log10(max_speed), int(round(label.shape[0])))

                # Afficher l'image du "Shot Gather" dans la première colonne
                axs[i, 0].imshow(data[0], aspect='auto', cmap='gray', extent=[0, nb_traces, time_vector[-1], time_vector[0]])
                axs[i, 0].set_title(f'Shot Gather {i}', fontsize=11)
                axs[i, 0].set_xlabel('Distance (m)', fontsize=10)
                axs[i, 0].set_ylabel('Time (sample)', fontsize=10)

                # Afficher le label et la prédiction dans la deuxième colonne
                axs[i, 1].pcolormesh(x, y, prediction, shading='auto')
                #plot the label to compare
                f = interp1d(np.arange(len(label)), label, kind='linear', axis=0)
                label = f(np.linspace(0, len(label) - 1, 51))
                axs[i, 1].plot(label, np.arange(len(label)), label='Label', linewidth=2, color='red')
                axs[i, 1].set_xlim(0, max_speed)
                axs[i, 1].set_ylim(0, y[-1])
                axs[i, 1].invert_yaxis()
                axs[i, 1].set_xlabel('Vs (m/s)', fontsize=10)
                axs[i, 1].set_ylabel('Depth (m)', fontsize=10)
                axs[i, 1].set_title(f'Vs classification {i}', fontsize=11)

                # Ajouter une barre de couleurs
                cbar = fig.colorbar(axs[i, 1].collections[0], ax=axs[i, 1])
                cbar.set_label('Probability', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'figures/{od}/predictionsVSlabels.pdf', format='pdf', dpi=300)  # Sauvegarde en PDF avec 300 dpi
        plt.close()


# Display some predictions
if args.dispersion_image:
    visualize_predictions(all_predictions,all_disp,shape1,shape2,test_dataloader,num_samples=5,od=args.output_dir)
else:
    visualize_predictions(num_samples=5)

#use fonction to compute histogram of probabilities:
print('Computing histogram of probabilities')
print('shape of all_predictions:',np.shape(all_predictions))
probability_histogram_top_k_classes(all_predictions, num_bins=10,top_k=10, od=args.output_dir)

#display on a single image all informations about the current model
main_path= os.path.abspath(__file__)
display_run_info(model=model,od=args.output_dir,args=args,metrics=metrics,training_time=training_time,main_path=main_path,best_params=None,nb_param=unfrozen_parameters,nb_classes=nb_classes)


#save the model parameters
torch.save(model.state_dict(), f'figures/{args.output_dir}/model.pth')