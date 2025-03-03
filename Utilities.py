import os
import shutil
import random
import numpy as np
import glob
import h5py as h5
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
import torchvision.transforms as transforms
import subprocess
import tensorflow as tf
from PhaseShiftMethod import *
import timm
from torchinfo import summary
from transformers import ViTModel
import torchvision.models as models
from NN_architectures import *
from scipy.sparse.linalg import lsqr

def PrepareClasses(train_dataloader=None, val_dataloader=None, test_dataloader=None,multiplier=20):
    """
    Convert labels to integers and round to the nearest multiple of 10.
    :param train_dataloader: training dataloader
    :param val_dataloader: validation dataloader
    :param test_dataloader: test dataloader
    :return: train_dataloader, val_dataloader, test_dataloader, num_classes
    """
    def round_to_nearest_MULTIPLIER(value, multiplier=multiplier):
        # Round to the nearest multiple of MULTIPLIER, and convert to index
        return int(round(value / multiplier))

    def process_labels(dataloader):
        max= 0
        for i in range(len(dataloader.dataset.labels)):
            for j in range(len(dataloader.dataset.labels[i])):
                if not isinstance(dataloader.dataset.labels[i][j][0], int):
                    # Convert to integer if not already
                    dataloader.dataset.labels[i][j][0] = int(dataloader.dataset.labels[i][j][0])
                # Round to the nearest multiple of MULTIPLIER
                rounded_label = round_to_nearest_MULTIPLIER(dataloader.dataset.labels[i][j][0],multiplier)
                dataloader.dataset.labels[i][j][0] = rounded_label
                if rounded_label > max:
                    max = rounded_label
        return dataloader, max

    # Set to store unique labels
    max = 0

    # Process train, validation, and test dataloaders
    if train_dataloader is not None:
        train_dataloader,train_max = process_labels(train_dataloader)
        if max < train_max:
            max = train_max
    if val_dataloader is not None:
        val_dataloader,val_max = process_labels(val_dataloader)
        if max < val_max:
            max = val_max
    if test_dataloader is not None:
        test_dataloader,test_max = process_labels(test_dataloader)
        if max < test_max:
            max = test_max

    # Number of unique classes = max()/10
    num_classes = max + 1

    # create a vector of size nb_classes with the total occurence of each label:
    #class by class:
    #train
    occurences_train = torch.zeros(num_classes)
    for i in range(len(train_dataloader.dataset.labels)):
        for j in range(len(train_dataloader.dataset.labels[i])):
            occurences_train[train_dataloader.dataset.labels[i][j][0]] += 1
    #validate
    occurences_val = torch.zeros(num_classes)
    for i in range(len(val_dataloader.dataset.labels)):
        for j in range(len(val_dataloader.dataset.labels[i])):
            occurences_val[val_dataloader.dataset.labels[i][j][0]] += 1
    #test
    occurences_test = torch.zeros(num_classes)
    for i in range(len(test_dataloader.dataset.labels)):
        for j in range(len(test_dataloader.dataset.labels[i])):
            occurences_test[test_dataloader.dataset.labels[i][j][0]] += 1
    #total
    occurences_total = occurences_train + occurences_val + occurences_test
    print('occurences_total:',occurences_total)



    return train_dataloader, val_dataloader, test_dataloader, num_classes, occurences_total

def PrepareClasses2(train_dataloader=None, val_dataloader=None, test_dataloader=None):
    """
    Create logarithmically spaced bins and assign labels to the corresponding bin index.
    :param train_dataloader: training dataloader
    :param val_dataloader: validation dataloader
    :param test_dataloader: test dataloader
    :return: train_dataloader, val_dataloader, test_dataloader, num_classes, bin_edges
    """

    def get_all_labels(dataloader):
        labels = []
        for i in range(len(dataloader.dataset.labels)):
            for j in range(len(dataloader.dataset.labels[i])):
                labels.append(dataloader.dataset.labels[i][j][0])
        return labels

    # Combine all labels from train, val, and test dataloaders
    all_labels = []
    if train_dataloader is not None:
        all_labels += get_all_labels(train_dataloader)
    if val_dataloader is not None:
        all_labels += get_all_labels(val_dataloader)
    if test_dataloader is not None:
        all_labels += get_all_labels(test_dataloader)

    max_label = max(all_labels)

    return train_dataloader, val_dataloader, test_dataloader, max_label

def OneHotEncoding(labels,nb_classes,dz=1):
    # Interpoler les labels : de taille 200 à taille 100
    #print('labels shape before interpolation:',labels.shape)
    #print('labels:', labels)
    new_positions = np.logspace(0, np.log10(199 - 1e-10), int(round(200 / dz)))
    labels = interp1d(np.arange(200), labels, kind='linear',axis=1)(new_positions)
    #convert to torch
    labels = torch.tensor(labels)
    #print('labels shape after interpolation:',labels.shape)
    #print('labels:',labels)
    labels = (torch.round(labels)).long()
    #print('max labels:', torch.max(labels))
    #print('nb_classes:',nb_classes)
    labels = torch.nn.functional.one_hot(labels, num_classes=int(nb_classes))
    #print('labels after One Hot:',labels)
    print('labels shape after One Hot:',labels.shape)
    return labels, new_positions

def OneHotEncode_IrregularBins(label,max_label=500,n1=50,n2=60,plot=False,figure_path='figures/'):
    """
    One hot encode the labels with irregular bins<
    :param label: the label to encode

    """

    log_power = 10

    # interpolate the label to make it correspond to the grid
    new_positions = np.logspace(0, np.emath.logn(log_power, 199 - 1e-10), round(n1), base=log_power)
    #print('original shape label:', np.shape(label))
    label = interp1d(np.arange(200), label, kind='linear', axis=0)(new_positions)

    # Generate non-linear grid
    x = np.zeros(n2)
    x_max = max_label
    y = np.zeros(n1)
    y_max = len(label)

    # try with logarithmic x and y:
    x = np.logspace(0, np.emath.logn(log_power, x_max), n2 + 1,
                    base=log_power)  # n2 + 1 points to include the max value
    y = np.logspace(0, np.emath.logn(log_power, y_max), n1 + 1,
                    base=log_power)  # n1 + 1 points to include the max value

    # Generate z variable
    z = np.zeros((n1, n2))
    # Generate z matrix, full of 0 and 1.
    # 1 if the value of the label is between the two values of the grid
    for i in range(n1):
        # foar each line of z
        for j in range(n2):
            if label[i][0] >= x[j] and label[i][0] < x[j + 1]:
                z[i, j] = 1

    if plot:
        # Plot a big figure, with 3 subplots.
        # 1rst one is the label alone.
        # 2nd one is the grid alone, with a random value of z.
        # 3rd one is the label OneHotEncoded on the grid.
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # Plot the label
        axs[0].plot(label, np.arange(n1),color='red', linewidth=2)
        axs[0].set_title('Label')
        axs[0].set_xlim(0, x_max)
        axs[0].set_ylim(0, y_max)
        axs[0].invert_yaxis()
        # Plot the grid
        x1, x2 = np.meshgrid(x, y)
        cbar1 = axs[1].pcolormesh(x1, x2, np.random.rand(n1, n2), shading='auto')
        axs[1].set_xlim(0, x_max)
        axs[1].set_ylim(0, y_max)
        axs[1].invert_yaxis()
        axs[1].set_title('Irregular Grid, filled with random values')

        # Plot the label OneHotEncoded on the grid
        cbar2 = axs[2].pcolormesh(x1, x2, z, shading='auto')
        axs[2].set_xlim(0, x_max)
        axs[2].set_ylim(0, y_max)
        axs[2].invert_yaxis()
        axs[2].set_title('Label OneHotEncoded')
        plt.colorbar(cbar1)
        plt.colorbar(cbar2)
        plt.savefig(figure_path + 'label_OneHotEncoding.png')
        plt.close()

    return x,y,z

def generate_geometric_vector(start=1, k=1.2, max_value=200):
    """
    Crée un vecteur avec des valeurs qui augmentent par un coefficient k,
    en commençant à 'start' et s'arrêtant à 'max_value'.

    :param start: Valeur de départ (généralement 1).
    :param k: Coefficient multiplicatif.
    :param max_value: Valeur maximale à atteindre (ou dépasser légèrement).
    :return: Vecteur numpy avec les valeurs générées.
    """
    values = [start]  # Initialisation du vecteur avec la première valeur

    # Générer les valeurs jusqu'à ce qu'on atteigne ou dépasse max_value
    while True:
        next_value = values[-1] * k
        if next_value > max_value:
            break
        values.append(next_value)

    # S'assurer que la dernière valeur est exactement max_value si elle n'y est pas encore
    if values[-1] < max_value:
        values.append(max_value)
    #print('shape values HERE:',len(values),'with max value:',max_value)
    return np.array(values)


def OneHotEncode_IrregularBins2(label, k=1.2, plot=False, figure_path='figures/'):
    """
    One hot encode the labels with irregular bins
    :param label: the label to encode

    """
    dz = 4
    # use the class HarmonicMeanInterpolation to interpolate the label to make it correspond to the grid
    #print('début HMI')
    HMI = HarmonicMeanInterpolation(label, k=k)

    new_label = HMI.harmonic_mean_interpolation(label)

    # Generate non-linear grid
    x_max = 5000  # 5000m/s
    y_max = 199  # 50m or 200 pts

    # try with logarithmic x and y:
    # x = np.logspace(np.log10(1e-9), np.log10(x_max), n2, base=k)
    #print('début geometric vector')
    x = generate_geometric_vector(1, k, x_max)
    n2 = len(x)
    # y = np.logspace(np.log10(1), np.log10(y_max), n1, base=log_power)
    y = generate_geometric_vector(1, k, y_max)
    n1 = len(y)

    # Generate z variable
    z = np.zeros((n1, n2))
    # Generate z matrix, full of 0 and 1.
    # 1 if the value of the label is between the two values of the grid
    for i in range(n1):
        # print('i:',i)
        # foar each line of z
        for j in range(n2):
            # print('j:',j)
            if new_label[i][0] >= x[j] and new_label[i][0] < x[j + 1]:
                z[i, j] = 1


    if plot:
        print('plotting')
        # Diviser y pour l'échelle à 50
        y = y / y[-1] * 50

        # Créer une figure avec 3 sous-graphes.
        # 1er: le label seul.
        # 2ème: la grille seule avec des valeurs aléatoires pour z.
        # 3ème: le label encodé OneHot sur la grille.
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)  # Taille ajustée, résolution 300 dpi pour rapport

        # Interpoler le label pour l'adapter à une longueur de 50
        f = interp1d(np.arange(len(label)), label, kind='linear', axis=0)
        label_int = f(np.linspace(0, len(label) - 1, 51))

        # Graphique 1: Label
        axs[0].plot(label_int, np.arange(len(label_int)), color='red', linewidth=1.5)  # Épaisseur de ligne ajustée
        axs[0].set_title('Label', fontsize=12)  # Taille du titre ajustée
        axs[0].set_xlabel('Velocity (m/s)', fontsize=10)  # Taille des étiquettes ajustée
        axs[0].set_ylabel('Depth (m)', fontsize=10)
        axs[0].set_xlim(0, x_max)
        axs[0].set_ylim(0, len(label_int) - 1)
        axs[0].invert_yaxis()

        # Graphique 2: Grille avec valeurs aléatoires
        x1, x2 = np.meshgrid(x, y)
        cbar1 = axs[1].pcolormesh(x1, x2, np.random.rand(n1, n2), shading='auto')
        axs[1].set_xlim(0, x_max)
        axs[1].set_ylim(1, y[-1])
        axs[1].invert_yaxis()
        axs[1].set_title(f'Irregular Grid (Random Values)\nDimensions: {n1}x{n2}', fontsize=12)
        axs[1].set_xlabel('Velocity (m/s)', fontsize=10)
        axs[1].set_ylabel('Depth (m)', fontsize=10)

        # Graphique 3: Label OneHotEncoded sur la grille
        cbar2 = axs[2].pcolormesh(x1, x2, z, shading='auto')
        axs[2].plot(label_int, np.arange(len(label_int)), color='red', linewidth=1.5)
        axs[2].set_xlim(0, x_max)
        axs[2].set_ylim(1, y[-1])
        axs[2].invert_yaxis()
        axs[2].set_title(f'OneHotEncoded Label\nDimensions: {n1}x{n2}', fontsize=12)
        axs[2].set_xlabel('Velocity (m/s)', fontsize=10)
        axs[2].set_ylabel('Depth (m)', fontsize=10)

        # Ajouter les barres de couleurs
        fig.colorbar(cbar1, ax=axs[1], orientation='vertical', label='Random Values')  # Légende de la barre de couleur
        fig.colorbar(cbar2, ax=axs[2], orientation='vertical', label='Probability')  # Légende de la barre de couleur

        # Ajuster l'agencement et enregistrer la figure
        plt.tight_layout()
        plt.savefig(figure_path + 'label_OneHotEncoding.pdf', format='pdf', dpi=300)  # Format PNG avec 300 dpi
        #plt.show()
        plt.close()

    return x, y, z

#def a function to calculate the number of classes
def NbClasses(train_dataloader=None, val_dataloader=None, test_dataloader=None):
    max=0
    # check if all labels are integers, in purpose to make classification
    for i in range(len(train_dataloader.dataset.labels)):
        for j in range(len(train_dataloader.dataset.labels[i])):
            # store maximum value
            if train_dataloader.dataset.labels[i][j][0]>max:
                max=train_dataloader.dataset.labels[i][j][0]
    # apply the same to validate
    for i in range(len(val_dataloader.dataset.labels)):
        for j in range(len(val_dataloader.dataset.labels[i])):
            # store maximum value
            if val_dataloader.dataset.labels[i][j][0]>max:
                max=val_dataloader.dataset.labels[i][j][0]

    # apply the same to test
    for i in range(len(test_dataloader.dataset.labels)):
        for j in range(len(test_dataloader.dataset.labels[i])):
            # store maximum value
            if test_dataloader.dataset.labels[i][j][0]>max:
                max=test_dataloader.dataset.labels[i][j][0]


    return max+1

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
    bad_files = []
    invalid_files = []  # Liste pour les fichiers non valides
    for folder in folders:
        count_nan = 0
        count_invalid = 0
        for file in folder:
            try:
                with h5.File(file, 'r') as h5file:
                    inputs = h5file['shotgather'][:]
                    labels_data = h5file['vsdepth'][:]

                    # Vérifier si 'inputs' contient des valeurs NaN
                    if np.isnan(inputs).any():
                        count_nan += 1
                        bad_files.append(file)
            except OSError as e:
                # Si le fichier ne peut pas être ouvert, le compter comme non valide
                count_invalid += 1
                invalid_files.append(file)

        print(f'{count_nan}/{len(folder)} files contain NaN values.')
        print(f'{count_invalid}/{len(folder)} files are invalid or corrupt.')

    print(f"Total invalid files: {len(invalid_files)}")
    return bad_files, invalid_files


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

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, y_pred, y_true, model):
        mse = torch.mean((y_true - y_pred) ** 2)
        variance = torch.mean(y_true ** 2)
        nmse = mse / variance
        return nmse


class SparseNMSE(nn.Module):
    def __init__(self, l1_weight=1):
        super(SparseNMSE, self).__init__()
        self.l1_weight = l1_weight

    def forward(self, y_pred, y_true, model):
        # Mean Squared Error
        mse = torch.mean((y_true - y_pred) ** 2)

        # Variance of y_true
        variance = torch.var(y_true)

        # Normalized MSE
        nmse = mse / variance

        # Calculate L1 regularization
        l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())

        # Normalize L1 regularization by the number of parameters
        num_params = sum(param.numel() for param in model.parameters())
        normalized_l1_reg = l1_reg / num_params

        # Combine normalized NMSE and normalized L1 norm
        total_loss = nmse + self.l1_weight * normalized_l1_reg
        return total_loss

class NMSERL1Loss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super(NMSERL1Loss, self).__init__()
        self.l1_weight = l1_weight

    def forward(self, y_pred, y_true, model):
        #print('y_pred:',y_pred.shape)
        #print('y_true:',y_true.shape)
        mse = torch.mean((y_true - y_pred.T) ** 2)
        variance = torch.mean(y_true ** 2)
        nmse = mse / variance
        l1_loss = sum(param.abs().sum() for param in model.parameters())
        total_loss = nmse + self.l1_weight * l1_loss
        return total_loss

class MSERL1Loss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super(MSERL1Loss, self).__init__()
        self.l1_weight = l1_weight

    def forward(self, y_pred, y_true, model):
        #print('y_pred:',y_pred.shape)
        #print('y_true:',y_true.shape)
        mse = torch.mean((y_true - y_pred.T) ** 2)
        l1_loss = sum(param.abs().sum() for param in model.parameters())
        total_loss = mse + self.l1_weight * l1_loss
        return total_loss

class CategoricalCrossEntropy(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()

    def forward(self, y_pred, y_true):
        #ref to J.Simon et al, 2024
        #for line in y_pred:
        loss= 0
        for i in range(len(y_pred)):
            #print('y_pred:',y_pred[i],'y_pred shape:',y_pred[i].shape)
            log_preds = torch.log(y_pred[i,:] + 1e-9)  # Adding epsilon to avoid log(0)
            loss += -torch.sum(y_true * log_preds)

        return loss/y_pred.size(0)


class CategoricalCrossEntropy2(nn.Module):
    def __init__(self, lambda_entropy=0.01, num_classes=10):
        super(CategoricalCrossEntropy2, self).__init__()
        self.lambda_entropy = lambda_entropy
        self.num_classes = num_classes  # nombre de classes pour la normalisation

    def forward(self, y_pred, y_true):
        # Assurer que y_pred sont des probabilités
        #y_pred = torch.softmax(y_pred, dim=1)

        # Calcul de la perte cross-entropy classique et normalisation
        loss = 0
        for i in range(len(y_pred)):
            log_preds = torch.log(y_pred[i, :] + 1e-9)  # Ajout d'un epsilon pour éviter log(0)
            # Normalisation par la perte maximale théorique pour chaque élément
            crossentropy_per_sample = torch.sum(y_true * log_preds)
            max_crossentropy = -(torch.tensor(1.0 / self.num_classes)) #-1/N
            normalized_loss = crossentropy_per_sample / max_crossentropy
            loss += normalized_loss

        # Moyenne de la perte normalisée sur le batch
        normalized_loss_mean = loss / y_pred.size(0)

        # Calcul de la pénalité d'entropie
        probs = y_pred  # y_pred est supposé être déjà normalisé (probabilités)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  # Entropie de chaque prédiction
        entropy_penalty = entropy.mean()

        # Normalisation de l'entropie par la borne maximale
        max_entropy = torch.log(torch.tensor(float(self.num_classes)))
        normalized_entropy_penalty = entropy_penalty / max_entropy

        # Ajout de la pénalité d'entropie à la perte totale
        total_loss = (1-self.lambda_entropy)*normalized_loss_mean + self.lambda_entropy * normalized_entropy_penalty

        print('main term (normalized):', normalized_loss_mean)
        print('entropy penalty (normalized):', normalized_entropy_penalty)

        return total_loss

def normalized_crossentropy_loss(y_pred, y_true, num_classes):
    # Utiliser CrossEntropyLoss sans reduction pour obtenir la perte par échantillon
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_per_sample = criterion(y_pred, y_true)

    # Normaliser chaque échantillon par la perte maximale
    max_loss = -torch.log(torch.tensor(1.0 / num_classes))
    normalized_loss_per_sample = loss_per_sample / max_loss

    # Moyenne de la perte normalisée
    normalized_loss = normalized_loss_per_sample.mean()

    return normalized_loss


class WeightedCrossEntropy(nn.Module):
    def __init__(self,weights=None):
        super(WeightedCrossEntropy, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        if self.weights is None:
            weights = torch.ones(len(y_pred))
        else:
            weights = self.weights
        #ref to J.Simon et al, 2024
        #print('y_pred shape:',y_pred.shape,'and labels shape:',y_true.shape)
        loss= 0
        print('LEN Y_PRED:',y_pred.shape[0])
        for i in range(y_pred.shape[0]):
            #print('y_pred:',y_pred[i],'y_pred shape:',y_pred[i].shape)
            log_preds = torch.log(y_pred[i,:] + 1e-9)  # Adding epsilon to avoid log(0)
            loss += (-torch.sum(y_true * log_preds))*weights[i]
            print('loss(i):',loss,'and weights(i):',weights[i])

        return loss/y_pred.size(0)


def test_loss(y_true, y_pred, l1=0.05):
    """ Account for MSE, z derivative, gradient and l1 """
    alpha = 0.0                    # 0.02   MSE z derivative
    beta  = 0.0                    # 0.1    Blocky inversion (continuity y_pred[i+1]-y_pred[i])
    l1 = l1                        # 0.05   l1 norm
    v_max = 0.0                    # 0.2    max value
    fact1 = 1 - alpha - beta - l1 - v_max   # MSE
    losses_f = []

    # Vérifier les dimensions
    print('shape y_pred:', y_pred.shape)
    print('shape y_true:', y_true.shape)

    # Mean Square Error (MSE)
    mse_loss = torch.sum((y_true - y_pred) ** 2, dim=1) / torch.sum(y_true ** 2, dim=1)
    mse_loss = torch.mean(mse_loss)
    losses_f.append(fact1 * mse_loss)

    # Minimize gradient (blocky inversion)
    if beta != 0:
        num = torch.sum(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]), dim=1)
        den = torch.norm(y_pred, p=1, dim=1) / 0.02
        grad_loss = torch.mean(num / den)
        losses_f.append(beta * grad_loss)

    # l1 norm
    if l1 != 0:
        l1_loss = torch.sum(torch.abs(y_true - y_pred), dim=1) / torch.sum(torch.abs(y_true), dim=1)
        l1_loss = torch.mean(l1_loss)
        losses_f.append(l1 * l1_loss)

    # Retourne la somme des différentes composantes de la perte
    return torch.sum(torch.stack(losses_f))

class JeffLoss(nn.Module):
    def __init__(self, alpha=0.0, beta=0., l1=0.05, v_max=0.):
        super(JeffLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.v_max = v_max
        self.fact1 = 1 - self.alpha - self.beta - self.l1 - self.v_max

    def forward(self, y_pred, y_true):
        #init loss
        losses_f = []
        # Calcul de l'erreur quadratique normalisée (NMSE)
        if y_true.shape == y_pred.shape:
            # Calcul de la NMSE en utilisant dim=1 (car [1, 200] est 2D avec une seule ligne)
            mse_loss = torch.sum((y_true - y_pred) ** 2, dim=1) / torch.sum(y_true ** 2, dim=1)
            mse_loss = torch.mean(mse_loss)
            losses_f.append(self.fact1 * mse_loss)
            #print('LOSS / main term:', self.fact1 * mse_loss)  # Moyenne des pertes sur la dimension des 200 éléments
        else:
            raise ValueError("Shapes of y_true and y_pred must match.")

        # Calculate Mean Squared Error of the z derivative
        if self.alpha != 0:
            dlabel = y_true[:, 1:, :] - y_true[:, :-1, :]
            dout = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            num = torch.sum((dlabel - dout) ** 2, dim=[1, 2])
            den = torch.sum(dlabel ** 2, dim=[1, 2]) + 1e-6  # Small constant for stability
            z_deriv_loss = torch.mean(num / den)
            losses_f.append(self.alpha * z_deriv_loss)
            #print('LOSS / z deriv:',self.alpha * z_deriv_loss)

        # Minimize gradient (blocky inversion)
        if self.beta != 0:
            #num = torch.sum(torch.abs(y_pred[1:, :] - y_pred[ :-1, :]), dim=[0])
            #den = torch.norm(y_pred, p=1, dim=[1])   # Normalization constant
            #grad_loss = torch.mean(num / den)

            # Décalage du vecteur
            y_pred_shifted = y_pred[:, 1:]
            # Suppression du premier élément pour aligner les deux vecteurs
            y_pred_original = y_pred[:, :-1]
            # Calcul des différences absolues
            diffs = torch.abs(y_pred_shifted - y_pred_original)
            # Somme des différences
            num = torch.sum(diffs)
            #print('num:', num)
            den = torch.norm(y_pred, p=1, dim=[1]) / 0.1  # Normalization constant
            #print('den:', den)
            grad_loss = torch.mean(num / den)
            losses_f.append(self.beta * grad_loss)
            #print('LOSS / blocky:',self.beta * grad_loss)

        # L1 Norm
        if self.l1 != 0:
            l1_loss = torch.sum(torch.abs(y_true - y_pred), dim=1) / torch.sum(torch.abs(y_true), dim=1)
            l1_loss = torch.mean(l1_loss)
            losses_f.append(self.l1 * l1_loss)
            #print('LOSS / L1:',self.l1 * l1_loss)

        # Max difference
        if self.v_max != 0:
            max_diff_loss = torch.sum(torch.abs(torch.max(y_true, dim=1)[0] - torch.max(y_pred, dim=1)[0]), dim=-1)
            max_val= torch.max(max_diff_loss,dim=0)[0]
            max_diff_loss = torch.mean(max_diff_loss)/max_val #normalize
            losses_f.append(self.v_max * max_diff_loss)
            #print('LOSS / max diff:',self.v_max * max_diff_loss)

        # Vérifie que `losses_f` ne contient pas d'éléments vides avant de les empiler
        if len(losses_f) > 0:
            total_loss = sum(losses_f)
        else:
            total_loss = torch.tensor(0.0, device=y_pred.device)
        #print('TOTAL LOSS:',total_loss)

        return total_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def label_interpolation(label,k):
    '''
    Interpolate the label to reduce its size
    :param label: vector
    :param k: factor of reduction, should be <1
    :return: interpolated vector: newlabel
    '''
    x_old = np.linspace(0, label.shape[0] - 1, label.shape[0])
    x_new = np.linspace(0, label.shape[0] - 1,
                        int(round(label.shape[0] * k)))  # Nouveaux index interpolés pour 100 points

    # Utilisation de np.interp avec le label aplati
    label = label.flatten()
    newlabel = np.interp(x_new, x_old, label)
    newlabel = newlabel.reshape(-1, 1)
    return newlabel

class HarmonicMeanInterpolation:
    def __init__(self, label, window_size=5, k=1.1):
        self.window_size = window_size
        self.label = label
        self.len_init = len(label)
        self.k = k

    def harmonic_mean(self, values):
        """Calculer la moyenne harmonique des valeurs"""
        #print('values:', values)
        return len(values) / np.sum(1.0 / values)

    def harmonic_interpolation(self, indices, vector):
        """
        Créer un nouveau vecteur avec la moyenne harmonique des valeurs autour des indices du vecteur original
        """
        indices = indices.astype(int)  # Convertir en entier pour indexer le tableau
        harmonic_means = []
        n = len(vector)

        for idx in indices:
            # Déterminer les indices des valeurs proches
            start = max(0, idx - self.window_size // 2)
            end = min(n, idx + self.window_size // 2 + 1)
            close_values = vector[start:end]

            # Calculer la moyenne harmonique des valeurs proches
            harmonic_means.append(self.harmonic_mean(close_values))

        # Convertir la liste en tableau numpy et redimensionner pour obtenir une forme (m, 1)
        final_vector = np.array(harmonic_means).reshape(len(harmonic_means), 1)
        #print('final vector shape harmonic mean:', np.shape(final_vector))
        return final_vector

    def logarithmic_indices(self, init_vector, num=None, base=20):
        """
        Générer des indices logarithmiques à partir d'un vecteur, avec un nombre donné de points et une base donnée
        """
        len_init = len(init_vector)
        if num is None:
            num = self.len_final  # Utiliser len_final si num n'est pas fourni

        log_indices = np.logspace(0, np.emath.logn(base, len_init - 1 - 1e-10), round(num), base=base)  # de 0 à num
        log_indices = np.clip(log_indices, 0, len_init - 1)  # Assurer que les indices sont dans les limites
        log_indices = np.round(log_indices).astype(int)  # Arrondir et convertir en entiers
        return log_indices

    def increasing_spacing_vector(self, start=1, k=1.1, max_value=200):
        """
        Crée un vecteur avec des valeurs qui augmentent par un coefficient k,
        en commençant à 'start' et s'arrêtant à 'max_value'.

        :param start: Valeur de départ (généralement 1).
        :param k: Coefficient multiplicatif.
        :param max_value: Valeur maximale à atteindre (ou dépasser légèrement).
        :return: Vecteur numpy avec les valeurs générées.
        """
        values = [start]  # Initialisation du vecteur avec la première valeur

        # Générer les valeurs jusqu'à ce qu'on atteigne ou dépasse max_value
        while True:
            next_value = values[-1] * k
            if next_value > max_value:
                break
            values.append(next_value)

        # S'assurer que la dernière valeur est exactement max_value si elle n'y est pas encore
        if values[-1] < max_value:
            values.append(max_value)
        #print('shape values increasing spacing vector:', np.shape(values),'with max value:',max_value)
        return np.array(values)

    def harmonic_mean_interpolation(self, label):
        """
        Interpoler le label en utilisant la moyenne harmonique
        :param label: le label à interpoler (tableau)
        :return: le label interpolé (tableau numpy de forme (m, 1))
        """
        #print('label shape:', label.shape)
        # Calculer la moyenne harmonique pour chaque valeur du vecteur
        harmonic_means = self.harmonic_interpolation(np.linspace(0, self.len_init - 1, self.len_init), label)
        #print('shape harmonic means:', np.shape(harmonic_means)) # (200,1)

        indices = self.increasing_spacing_vector(start=1, k=self.k, max_value=self.len_init - 1) # 57 indices for k=1.1
        # transform into integers
        indices = indices.astype(int)
        #print('log indices shape:', indices.shape) # if velocity max=6000 so it takes 93 points for k=1.1


        # Créer le vecteur final, avec des moyennes harmoniques correspondant aux indices logarithmiques (len_final)
        final_vector = harmonic_means[indices]
        #print('final_vector shape:', final_vector.shape)

        return final_vector

    def __call__(self):
        return self.harmonic_mean_interpolation(self.label)

def display_run_info(model=None,od=None,args=None,training_time=None,metrics=None,main_path=None,best_params=None,nb_param=None,nb_classes=None):

    if model and od and args and training_time and main_path:

        Fonction_script_path = os.path.abspath(__file__)

        with open(f'figures/{od}/run_infos.txt', 'w') as f:
            #copy this whole code in the file
            f.write('CODE FUNCTIONS:\n')
            with open(Fonction_script_path, 'r') as main_script_file:
                main_script_code = main_script_file.read()
                f.write(main_script_code)
            f.write('\n----------------------------------------------------')
            f.write('\n')
            #main code
            f.write('MAIN CODE:\n')
            with open(main_path, 'r') as main_script_file:
                main_script_code = main_script_file.read()
                f.write(main_script_code)
            #model summary
            f.write('MODEL SUMMARY:\n')
            f.write(str(model))
            if nb_param:
                f.write(f'Number of learnable parameters: {nb_param}')
            f.write('\n----------------------------------------------------')
            f.write('\n')
            #arguments
            f.write('ARGUMENTS:\n')
            f.write(str(args))
            f.write('\n----------------------------------------------------')
            f.write('\n')
            #best hyperparameters
            try :
                f.write('BEST HYPERPARAMETERS:\n')
                f.write(f'Best hyperparameters: {best_params}')
                f.write('\n----------------------------------------------------')
                f.write('\n')
            except:
                pass
            #training time
            f.write('EFFICIENCY:\n')
            f.write(f'Training time: {training_time}\n')
            #if nb_classes diff None:
            if nb_classes:
                f.write(f'Number of classes: {nb_classes}\n')

            # if metrics = dict
            if isinstance(metrics, dict):
                # Ecrire toutes les accuracies dans le fichier
                for key, value in metrics.items():
                    f.write(f'{key}: {100 * value:.2f}%\n')  # Affiche les accuracies avec 2 décimales
                f.write('----------------------------------------------------\n')

            else:
                #metrics
                # metrics = loss
                f.write(f'Loss: {metrics}\n')
                f.write('----------------------------------------------------')


                pass


def reshaping(inputs):
    '''
    Objective is to reshape an image in shape (X,Y) to (224,224).
    '''
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

    return inputs

def get_gpu_with_most_free_memory():
    # Exécute la commande nvidia-smi pour récupérer les informations sur la mémoire des GPUs
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    # Convertit la sortie en une liste de tuples (total, free) pour chaque GPU
    memory_info = result.stdout.decode('utf-8').strip().split('\n')
    memory_info = [tuple(map(int, x.split(','))) for x in memory_info]

    # Affiche les informations de mémoire pour chaque GPU
    for idx, (total, free) in enumerate(memory_info):
        print(f"GPU {idx}: Total = {total} MiB, Libre = {free} MiB")

    # Calcule la mémoire libre pour chaque GPU
    free_memory = [x[1] for x in memory_info]

    # Trouver l'index du GPU avec le plus de mémoire libre
    return free_memory.index(max(free_memory))

def compare_models(train_dataloader,val_dataloader,test_dataloader,folder1,folder2,disper=False):
    # Load the models
    # Load the models
    path = '/home/rbertille/data/pycharm/ViT_project/pycharm_ViT/figures'

    # Recréer une instance du modèle
    model1 = PretrainedPVTv2()
    model2 = PretrainedPVTv2()

    # Charger les poids des modèles
    model1.load_state_dict(torch.load(f'figures/{folder1}/model.pth'))
    model2.load_state_dict(torch.load(f'figures/{folder2}/model.pth'))

    od = f'{path}/{folder1}'



    # Device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Envoyer les modèles sur le bon device
    model1.to(device)
    model2.to(device)

    def calculate_maxlabels(train_dataloader,val_dataloader,test_dataloader):
        max_label=0
        for i in range(len(train_dataloader.dataset.labels)):
            if train_dataloader.dataset.labels[i].max() > max_label:
                max_label = train_dataloader.dataset.labels[i].max()
        for i in range(len(val_dataloader.dataset.labels)):
            if val_dataloader.dataset.labels[i].max() > max_label:
                max_label = val_dataloader.dataset.labels[i].max()
        for i in range(len(test_dataloader.dataset.labels)):
            if test_dataloader.dataset.labels[i].max() > max_label:
                max_label = test_dataloader.dataset.labels[i].max()
        return max_label


    # Apply evaluation on both models
    def evaluate(model=model1, test_dataloader=test_dataloader, device=device, disper=disper):
        print('\nEVALUATION:')

        #dataset name:
        dataset_name='Dataset1Dhuge_96tr'

        # Evaluate the model
        model.eval()  # Set the model to evaluation mode
        # loss
        criterion = JeffLoss(l1=0,beta=0)

        # Initializations
        all_images = []
        all_predictions = []
        all_labels = []
        all_disp = []

        if disper==True:
            max_label=calculate_maxlabels(train_dataloader, val_dataloader, test_dataloader)
            fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name, max_c=max_label)

        # Loop through the test set
        for step in range(len(test_dataloader)):
            sample = test_dataloader.dataset[step]
            inputs = sample['data']
            if disper==True:
                disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=fmax).numpy().T
                disp, shape1, shape2 = prepare_disp_for_NN(disp)
                disp = disp.unsqueeze(0).to(device)
            labels = sample['label']
            # print('labels:',labels.shape)
            # inputs= torch.tensor(test_dataloader.dataset.data[step], dtype=torch.float32)
            # labels= torch.tensor(test_dataloader.dataset.labels[step], dtype=torch.float32)
            inputs = inputs.unsqueeze(0)
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            all_images.append(inputs.cpu().numpy())  # Transfer images to CPU
            with torch.no_grad():  # No need to calculate gradients
                if disper==True:
                    outputs = model(disp)
                else:
                    outputs = model(inputs)  # Make predictions
                # calculate error:
                loss = criterion(outputs.float(), labels.permute(1, 0).float())
                # print('test loss:',loss.item())
            all_predictions.append(outputs.detach().cpu().numpy())  # Transfer predictions to CPU
            all_labels.append(labels.cpu().numpy().T)  # Transfer labels to CPU
            if disper==True:
                all_disp.append(disp.cpu().numpy())

        # Concatenate all images, predictions, and labels
        all_images = np.concatenate(all_images, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # mean loss:
        loss_np = loss.cpu().detach().numpy()
        mean_loss = np.mean(loss_np)

        # Display the mean squared error
        print("NMSE on Test Set:", mean_loss)


        if disper==True:
            return all_images, all_predictions, all_labels, all_disp, mean_loss, shape1, shape2
        else:
            return all_images, all_predictions, all_labels, mean_loss

    # Evaluate the model1
    print(f'Model 1: {folder1}')
    if disper==True:
        all_images, all_predictions, all_labels, all_disp, mean_loss, shape1, shape2 = evaluate(model1, test_dataloader,
                                                                                                device)
    else:
        all_images, all_predictions, all_labels, mean_loss = evaluate(model1, test_dataloader, device)

    # Evaluate the model2
    print(f'Model 2: {folder2}')
    if disper==True:
        all_images2, all_predictions2, all_labels2, all_disp2, mean_loss2, shape1, shape2 = evaluate(model2, test_dataloader,
                                                                                                    device)
    else:
        all_images2, all_predictions2, all_labels2, mean_loss2 = evaluate(model2, test_dataloader, device)

    # Visualize some predictions on test dalaloader:


    def calculate_VSmoy(train_dataloader,val_dataloader,test_dataloader):
        # objective is to calculate the average Vs vector for all the dataset

        #put all labels together:
        all_labels = []
        for i in range(len(train_dataloader.dataset.labels)):
            all_labels.append(train_dataloader.dataset.labels[i])
        for i in range(len(val_dataloader.dataset.labels)):
            all_labels.append(val_dataloader.dataset.labels[i])
        for i in range(len(test_dataloader.dataset.labels)):
            all_labels.append(test_dataloader.dataset.labels[i])
        print('shape all labels:',np.shape(all_labels))

        #calculate the average Vs vector
        Vs_moy = np.mean(all_labels,axis=0)
        print('shape Vs moy:',np.shape(Vs_moy))

        return Vs_moy

    #calculate Vs_moy
    Vs_moy = calculate_VSmoy(train_dataloader,val_dataloader,test_dataloader)
    def visualize_predictions(all_predictions1=all_predictions,all_predictions2=all_predictions2, all_disperion='None', shape1='None', shape2='None',
                              test_dataloader=test_dataloader,val_dataloader=val_dataloader,train_dataloader=train_dataloader, num_samples=5,disper=disper,od=od,VS_moy=Vs_moy):

        max_label=calculate_maxlabels(train_dataloader,val_dataloader,test_dataloader)

        # Calculer Vs_min et Vs_max
        Vs_min = min([np.min(all_labels), np.min(all_predictions) * max_label])
        Vs_max = max([np.max(all_labels), np.max(all_predictions) * max_label])

        if disper == True:
            all_dispersion = all_disp
            # Paramètres de dispersion
            fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters('Dataset1Dhuge_96tr', max_c=max_label)

            if num_samples > len(all_predictions):
                num_samples = len(all_predictions)

            fig, axs = plt.subplots(nrows=num_samples, ncols=2,
                                    figsize=(7.5, 3.75 * num_samples))  # Taille ajustée pour rapport

            if num_samples == 1:
                print('You must display at least 2 samples')
            else:
                for i in range(num_samples):
                    # Données et étiquettes pour l'échantillon i
                    sample = test_dataloader.dataset[i]
                    data = sample['data']
                    label = sample['label']

                    prediction = all_predictions[i]
                    prediction2 = all_predictions2[i]
                    disp = all_dispersion[i]

                    # Conversion du label en unités réelles
                    dz = 0.25
                    depth_vector = np.arange(label.shape[0]) * dz

                    # Afficher l'image dans la première colonne
                    axs[i, 0].imshow(disp[0][0][:shape1, :shape2], aspect='auto')
                    xticks_positions = np.linspace(0, disp.shape[3] - 1, 5).astype(int)
                    xticks_labels = np.round(np.linspace(np.min(c), np.max(c), 5)).astype(int)
                    axs[i, 0].set_xticks(xticks_positions)
                    axs[i, 0].set_xticklabels(xticks_labels)
                    axs[i, 0].set_title(f'Dispersion image {i}', fontsize=10)  # Taille du titre
                    axs[i, 0].set_xlabel('phase velocity (m/s)', fontsize=9)  # Taille de l'étiquette
                    axs[i, 0].set_ylabel('frequency (Hz)', fontsize=9)

                    # Afficher le label dans la deuxième colonne
                    axs[i, 1].plot(label, depth_vector, label='Label', linewidth=1)
                    axs[i, 1].plot(prediction.reshape(-1) * max_label, depth_vector, label='alpha = 0.', linewidth=1)
                    axs[i, 1].plot(prediction2.reshape(-1) * max_label, depth_vector, label='alpha = 0.1', linewidth=1)
                    axs[i,1].plot(Vs_moy,depth_vector,label='Vs_moy',linewidth=1,linestyle='--')
                    axs[i, 1].invert_yaxis()
                    axs[i, 1].set_xlim(Vs_min, Vs_max)
                    axs[i, 1].set_xlabel('Vs (m/s)', fontsize=10)
                    axs[i, 1].set_ylabel('Depth (m)', fontsize=10)
                    axs[i, 1].set_title(f'Predictions vs label {i}', fontsize=11)
                    #legende en haut a droite
                    axs[i, 1].legend(fontsize=9, loc='upper right')

            plt.tight_layout()
            plt.savefig(f'{od}/2models_predictionsVSlabels.pdf', format='pdf', dpi=300)  # Format PDF et 300 dpi
            plt.close()

        else:
            if num_samples > len(all_predictions):
                num_samples = len(all_predictions)

            fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(7.5, 3.75 * num_samples))  # Taille ajustée

            if num_samples == 1:
                print('You must display at least 2 samples')
            else:
                for i in range(num_samples):
                    # Données et étiquettes pour l'échantillon i
                    sample = test_dataloader.dataset[i]
                    data = sample['data']
                    label = sample['label']

                    dataset_name = 'Dataset1Dhuge_96tr'

                    prediction = all_predictions[i]
                    prediction2 = all_predictions2[i]

                    # Conversion en unités réelles
                    time_vector = np.linspace(0, 1.5, data.shape[0])
                    nb_traces = 96
                    print('nb traces:', nb_traces)
                    dz = 0.25
                    depth_vector = np.arange(label.shape[0]) * dz

                    # Afficher l'image dans la première colonne
                    axs[i, 0].imshow(data[0], aspect='auto', cmap='gray',
                                     extent=[0, nb_traces, time_vector[-1], time_vector[0]])
                    axs[i, 0].set_title(f'Shot Gather {i}', fontsize=10)
                    axs[i, 0].set_xlabel('Distance (m)', fontsize=10)
                    axs[i, 0].set_ylabel('Time (sample)', fontsize=10)

                    # Afficher le label dans la deuxième colonne
                    axs[i, 1].plot(label, depth_vector, label='Label', linewidth=1)
                    axs[i, 1].plot(prediction.reshape(-1) * max_label, depth_vector, label='alpha = 0', linewidth=1)
                    axs[i, 1].plot(prediction2.reshape(-1) * max_label, depth_vector, label='alpha = 0.6', linewidth=1)
                    axs[i,1].plot(Vs_moy,depth_vector,label='Vs_moy',linewidth=1,linestyle='--')
                    axs[i, 1].invert_yaxis()
                    axs[i, 1].set_xlim(Vs_min, Vs_max)
                    axs[i, 1].set_xlabel('Vs (m/s)', fontsize=10)
                    axs[i, 1].set_ylabel('Depth (m)', fontsize=10)
                    axs[i, 1].set_title(f'Vs Depth {i}', fontsize=11)
                    axs[i, 1].legend(fontsize=8, loc='upper right')

            plt.tight_layout()
            plt.savefig(f'{od}/2models_predictionsVSlabels.pdf', format='pdf', dpi=300)  # Format PDF et 300 dpi
            plt.close()

    # Display some predictions
    if disper == True:
        visualize_predictions(all_predictions1=all_predictions,all_predictions2=all_predictions2,
                              all_disperion=all_disp, shape1=shape1, shape2=shape2,
                              train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                              test_dataloader=test_dataloader, num_samples=5,
                              od=od)
    else:
        visualize_predictions(all_predictions1=all_predictions,all_predictions2=all_predictions2,
                              train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                              test_dataloader=test_dataloader, num_samples=5,
                              od=od)

def compare_models_oral(train_dataloader,val_dataloader,test_dataloader,folder1,folder2,disper=False,alpha=0.1):
    # Load the models
    # Load the models
    path = '/home/rbertille/data/pycharm/ViT_project/pycharm_ViT/figures'

    # Recréer une instance du modèle
    model1 = PretrainedPVTv2()
    model2 = PretrainedPVTv2()

    # Charger les poids des modèles
    model1.load_state_dict(torch.load(f'figures/{folder1}/model.pth'))
    model2.load_state_dict(torch.load(f'figures/{folder2}/model.pth'))

    od = f'{path}/{folder1}'



    # Device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Envoyer les modèles sur le bon device
    model1.to(device)
    model2.to(device)

    def calculate_maxlabels(train_dataloader,val_dataloader,test_dataloader):
        max_label=0
        for i in range(len(train_dataloader.dataset.labels)):
            if train_dataloader.dataset.labels[i].max() > max_label:
                max_label = train_dataloader.dataset.labels[i].max()
        for i in range(len(val_dataloader.dataset.labels)):
            if val_dataloader.dataset.labels[i].max() > max_label:
                max_label = val_dataloader.dataset.labels[i].max()
        for i in range(len(test_dataloader.dataset.labels)):
            if test_dataloader.dataset.labels[i].max() > max_label:
                max_label = test_dataloader.dataset.labels[i].max()
        return max_label


    # Apply evaluation on both models
    def evaluate(model=model1, test_dataloader=test_dataloader, device=device, disper=disper):
        print('\nEVALUATION:')

        #dataset name:
        dataset_name='Dataset1Dhuge_96tr'

        # Evaluate the model
        model.eval()  # Set the model to evaluation mode
        # loss
        criterion = JeffLoss(l1=0,beta=0)

        # Initializations
        all_images = []
        all_predictions = []
        all_labels = []
        all_disp = []

        if disper==True:
            max_label=calculate_maxlabels(train_dataloader, val_dataloader, test_dataloader)
            print('max label:',max_label)
            fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name, max_c=max_label)

        # Loop through the test set
        for step in range(len(test_dataloader)):
            sample = test_dataloader.dataset[step]
            inputs = sample['data']
            if disper==True:
                disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=fmax).numpy().T
                disp, shape1, shape2 = prepare_disp_for_NN(disp)
                disp = disp.unsqueeze(0).to(device)
            labels = sample['label']
            # print('labels:',labels.shape)
            # inputs= torch.tensor(test_dataloader.dataset.data[step], dtype=torch.float32)
            # labels= torch.tensor(test_dataloader.dataset.labels[step], dtype=torch.float32)
            inputs = inputs.unsqueeze(0)
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            all_images.append(inputs.cpu().numpy())  # Transfer images to CPU
            with torch.no_grad():  # No need to calculate gradients
                if disper==True:
                    outputs = model(disp)
                else:
                    outputs = model(inputs)  # Make predictions
                # calculate error:
                loss = criterion(outputs.float(), labels.permute(1, 0).float())
                # print('test loss:',loss.item())
            all_predictions.append(outputs.detach().cpu().numpy())  # Transfer predictions to CPU
            all_labels.append(labels.cpu().numpy().T)  # Transfer labels to CPU
            if disper==True:
                all_disp.append(disp.cpu().numpy())

        # Concatenate all images, predictions, and labels
        all_images = np.concatenate(all_images, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # mean loss:
        loss_np = loss.cpu().detach().numpy()
        mean_loss = np.mean(loss_np)

        # Display the mean squared error
        print("NMSE on Test Set:", mean_loss)


        if disper==True:
            return all_images, all_predictions, all_labels, all_disp, mean_loss, shape1, shape2
        else:
            return all_images, all_predictions, all_labels, mean_loss

    # Evaluate the model1
    print(f'Model 1: {folder1}')
    if disper==True:
        all_images, all_predictions, all_labels, all_disp, mean_loss, shape1, shape2 = evaluate(model1, test_dataloader,
                                                                                                device)
    else:
        all_images, all_predictions, all_labels, mean_loss = evaluate(model1, test_dataloader, device)

    # Evaluate the model2
    print(f'Model 2: {folder2}')
    if disper==True:
        all_images2, all_predictions2, all_labels2, all_disp2, mean_loss2, shape1, shape2 = evaluate(model2, test_dataloader,
                                                                                                    device)
    else:
        all_images2, all_predictions2, all_labels2, mean_loss2 = evaluate(model2, test_dataloader, device)

    # Visualize some predictions on test dalaloader:


    def calculate_VSmoy(train_dataloader,val_dataloader,test_dataloader):
        # objective is to calculate the average Vs vector for all the dataset

        #put all labels together:
        all_labels = []
        for i in range(len(train_dataloader.dataset.labels)):
            all_labels.append(train_dataloader.dataset.labels[i])
        for i in range(len(val_dataloader.dataset.labels)):
            all_labels.append(val_dataloader.dataset.labels[i])
        for i in range(len(test_dataloader.dataset.labels)):
            all_labels.append(test_dataloader.dataset.labels[i])
        print('shape all labels:',np.shape(all_labels))

        #calculate the average Vs vector
        Vs_moy = np.mean(all_labels,axis=0)
        print('shape Vs moy:',np.shape(Vs_moy))

        return Vs_moy

    #calculate Vs_moy
    Vs_moy = calculate_VSmoy(train_dataloader,val_dataloader,test_dataloader)
    def visualize_predictions(all_predictions1=all_predictions,all_predictions2=all_predictions2, all_disperion='None', shape1='None', shape2='None',
                              test_dataloader=test_dataloader,val_dataloader=val_dataloader,train_dataloader=train_dataloader, num_samples=3,disper=disper,od=od,VS_moy=Vs_moy,alpha=0.1):

        max_label=calculate_maxlabels(train_dataloader,val_dataloader,test_dataloader)

        # Calculer Vs_min et Vs_max
        Vs_min = min([np.min(all_labels), np.min(all_predictions) * max_label])
        Vs_max = max([np.max(all_labels), np.max(all_predictions) * max_label])

        if disper == True:
            all_dispersion = all_disp
            # Paramètres de dispersion
            fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters('Dataset1Dhuge_96tr', max_c=max_label)

            if num_samples > len(all_predictions):
                num_samples = len(all_predictions)

            fig, axs = plt.subplots(nrows=num_samples, ncols=2,
                                    figsize=(7.5, 3.75 * num_samples))  # Taille ajustée pour rapport

            if num_samples == 1:
                print('You must display at least 2 samples')
            else:
                for i in range(num_samples):
                    # Données et étiquettes pour l'échantillon i
                    sample = test_dataloader.dataset[i]
                    data = sample['data']
                    label = sample['label']

                    prediction = all_predictions[i]
                    prediction2 = all_predictions2[i]
                    disp = all_dispersion[i]

                    # Conversion du label en unités réelles
                    dz = 0.25
                    depth_vector = np.arange(label.shape[0]) * dz

                    # Afficher l'image dans la première colonne
                    axs[i, 0].imshow(disp[0][0][:shape1, :shape2], aspect='auto')
                    xticks_positions = np.linspace(0, disp.shape[3] - 1, 5).astype(int)
                    xticks_labels = np.round(np.linspace(np.min(c), np.max(c), 5)).astype(int)
                    axs[i, 0].set_xticks(xticks_positions)
                    axs[i, 0].set_xticklabels(xticks_labels)
                    #afficher titre seulement si c'est la ligne du haut:
                    if i == 0:
                        axs[i, 0].set_title(f'Dispersion image', fontsize=16)
                    # afficher etiquette axe des x seulement pour la ligne du bas:
                    if i == num_samples - 1:
                        axs[i, 0].set_xlabel('phase velocity (m/s)', fontsize=16)  # Taille de l'étiquette
                    axs[i, 0].set_ylabel('frequency (Hz)', fontsize=16)

                    # Afficher le label dans la deuxième colonne
                    axs[i, 1].plot(label, depth_vector, label='Label', linewidth=1)
                    axs[i, 1].plot(prediction.reshape(-1) * max_label, depth_vector, label='alpha = 0.', linewidth=1)
                    axs[i, 1].plot(prediction2.reshape(-1) * max_label, depth_vector, label=f'alpha ={alpha}', linewidth=1)
                    axs[i,1].plot(Vs_moy,depth_vector,label='Vs_moy',linewidth=1,linestyle='--')
                    axs[i, 1].invert_yaxis()
                    axs[i, 1].set_xlim(Vs_min, Vs_max)
                    #afficher etiquette axe des x seulement pour la ligne du bas:
                    if i == num_samples - 1:
                        axs[i, 1].set_xlabel('Vs (m/s)', fontsize=16)
                    axs[i, 1].set_ylabel('Depth (m)', fontsize=16)
                    #afficher titre seulement si c'est la ligne du haut:
                    if i == 0:
                        axs[i, 1].set_title(f'Predictions vs label', fontsize=18)
                    #legende en haut a droite, seulement si ligne du haut:
                    if i == 0:
                        axs[i, 1].legend(fontsize=9, loc='upper right')

            plt.tight_layout()
            plt.savefig(f'{od}/2models_predictionsVSlabels_oral_DISP.png', format='png', dpi=300)  # Format PDF et 300 dpi
            plt.close()

        else:
            if num_samples > len(all_predictions):
                num_samples = len(all_predictions)

            fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(7.5, 3.75 * num_samples))  # Taille ajustée

            if num_samples == 1:
                print('You must display at least 2 samples')
            else:
                for i in range(num_samples):
                    # Données et étiquettes pour l'échantillon i
                    sample = test_dataloader.dataset[i]
                    data = sample['data']
                    label = sample['label']

                    prediction = all_predictions[i]
                    prediction2 = all_predictions2[i]

                    # Conversion en unités réelles
                    time_vector = np.linspace(0, 1.5, data.shape[0])
                    nb_traces = 96
                    print('nb traces:', nb_traces)
                    dz = 0.25
                    depth_vector = np.arange(label.shape[0]) * dz

                    # Afficher l'image dans la première colonne
                    axs[i, 0].imshow(data[0], aspect='auto', cmap='gray',
                                     extent=[0, nb_traces, time_vector[-1], time_vector[0]])
                    #afficher titre seulement si c'est la ligne du haut:
                    if i == 0:
                        axs[i, 0].set_title(f'Shot Gather', fontsize=18)
                    # afficher etiquette axe des x seulement pour la ligne du bas:
                    if i == num_samples - 1:
                        axs[i, 0].set_xlabel('Distance (m)', fontsize=16)
                    axs[i, 0].set_ylabel('Time (s)', fontsize=16)

                    # Afficher le label dans la deuxième colonne
                    axs[i, 1].plot(label, depth_vector, label='Label', linewidth=1)
                    axs[i, 1].plot(prediction.reshape(-1) * max_label, depth_vector, label='alpha = 0', linewidth=1)
                    axs[i, 1].plot(prediction2.reshape(-1) * max_label, depth_vector, label=f'alpha ={alpha}', linewidth=1)
                    axs[i,1].plot(Vs_moy,depth_vector,label='Vs_moy',linewidth=1,linestyle='--')
                    axs[i, 1].invert_yaxis()
                    axs[i, 1].set_xlim(Vs_min, Vs_max)
                    #afficher etiquette axe des x seulement pour la ligne du bas:
                    if i == num_samples - 1:
                        axs[i, 1].set_xlabel('Vs (m/s)', fontsize=16)
                    axs[i, 1].set_ylabel('Depth (m)', fontsize=16)
                    #afficher titre seulement si c'est la ligne du haut:
                    if i == 0:
                        axs[i, 1].set_title(f'Predictions vs label', fontsize=18)
                    #afficher la légende seulement pour la ligne du haut:
                    if i == 0:
                        axs[i, 1].legend(fontsize=9, loc='upper right')

            plt.tight_layout()
            plt.savefig(f'{od}/2models_predictionsVSlabels_oral.png', format='png', dpi=300)  # Format PDF et 300 dpi
            plt.close()

    # Display some predictions
    if disper == True:
        visualize_predictions(all_predictions1=all_predictions,all_predictions2=all_predictions2,
                              all_disperion=all_disp, shape1=shape1, shape2=shape2,
                              train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                              test_dataloader=test_dataloader, num_samples=3,
                              od=od,alpha=alpha)
    else:
        visualize_predictions(all_predictions1=all_predictions,all_predictions2=all_predictions2,
                              train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                              test_dataloader=test_dataloader, num_samples=3,
                              od=od,alpha=alpha)

def probability_histogram_top_k_classes(prob_tensor, top_k=5, num_bins=10, od="output"):
    """
    Calcule un histogramme des probabilités, mais ne conserve que les occurrences des top k classes
    les plus représentées et retourne les occurrences en pourcentage par rapport à ce sous-ensemble.

    Args:
    - prob_tensor (torch.Tensor): Tenseur contenant des probabilités (valeurs entre 0 et 1).
    - top_k (int): Nombre de classes les plus représentées à conserver (par défaut 10).
    - num_bins (int): Nombre de bins pour l'histogramme (par défaut 10).
    - od (str): Dossier de sortie pour sauvegarder l'histogramme.

    Returns:
    - hist_percent (numpy array): Histogramme des pourcentages d'occurrences des top k classes.
    """
    # Aplatir le tenseur en un vecteur 1D
    prob_matrix = prob_tensor.flatten().cpu().numpy()

    # 1. Calculer l'histogramme complet sur toutes les classes/bins
    bins = np.linspace(0, 1, num_bins + 1)
    hist, bin_edges = np.histogram(prob_matrix, bins=bins)

    # 2. Identifier les indices des top k classes les plus représentées
    top_k_indices = np.argsort(hist)[-top_k:]

    # 3. Filtrer les occurrences pour ne garder que celles des top k classes
    top_k_hist = hist[top_k_indices]

    # 4. Calculer le pourcentage par rapport au total du top k
    total_top_k_occurrences = np.sum(top_k_hist)
    hist_percent = (top_k_hist / total_top_k_occurrences) * 100 if total_top_k_occurrences > 0 else np.zeros_like(top_k_hist)

    # 5. Afficher l'histogramme
    plt.figure(figsize=(8, 6))
    plt.bar(top_k_indices, hist_percent, width=0.8, edgecolor='black', align='center')

    # Ajouter les labels
    plt.xlabel('Class Index (Top k)', fontsize=10)
    plt.ylabel('Percentage (%)', fontsize=10)
    plt.title(f'Top {top_k} Classes Probability Distribution Histogram', fontsize=12)

    # Sauvegarder la figure
    plt.tight_layout()
    plt.savefig(f'figures/{od}/probability_histogram_top_{top_k}_classes.pdf', format='pdf', dpi=300)
    plt.close()

    return hist_percent


def model_info(model, input_size=(1, 3, 224, 224)):

    # Nombre total de paramètres
    total_params = sum(p.numel() for p in model.parameters())

    # Nombre de paramètres entraînables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calcul des FLOPs pour une passe d'inférence
    flops_before_freeze = summary(model, input_size=input_size, verbose=0).total_mult_adds

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops_inference': flops_before_freeze
    }

def calculate_VSZ(label, i):
    """
    Calcule VsZ jusqu'à la profondeur i * 0.25 m.

    Parameters:
    -----------
    label : np.array
        Vecteur des vitesses Vs (200 éléments, 0.25m par point).
    i : int
        Indice jusqu'auquel on veut calculer VsZ.

    Returns:
    --------
    float
        Valeur de VsZ.
    """

    # Initialisation de la somme (Hi / Vi)
    Hi_Vi_sum = 0.0

    # Boucle pour calculer Hi / Vi pour chaque couche
    for j in range(i + 1):  # Inclure l'indice i
        Vi = label[j]
        if Vi == 0:  # Vérifier si la vitesse est nulle
            raise ValueError(f"Vitesse nulle détectée à l'indice {j}. Impossible de diviser par zéro.")

        Hi = 0.25  # Chaque échantillon correspond à 0.25m
        Hi_Vi_sum += Hi / Vi  # Somme pondérée par la vitesse

    # Calcul de VsZ : VsZ = Z / sum(Hi / Vi)
    Z = (i + 1) * 0.25  # Profondeur totale jusqu'à l'indice i
    VSZ = Z / Hi_Vi_sum  # Assurez-vous que ceci est un scalaire float

    # Impression sécurisée
    print(f"VSZ pour i={i}: {float(VSZ):.2f}")
    return VSZ


def calculate_vsz_vector(label):
    """
    Calcule un vecteur [Vs(z=1), Vs(z=5), ..., Vs(z=50)] pour un label donné.

    Parameters:
    -----------
    label : np.array
        Vecteur des vitesses Vs de taille (200, 1), avec une valeur tous les 0.25 m.

    Returns:
    --------
    np.array
        Vecteur de 11 valeurs représentant VsZ à 1m, 5m, 10m, ..., 50m.
    """
    # Définir les profondeurs cibles en mètres et convertir en indices
    depths = np.arange(1, 51, 1)  # Profondeurs de 1 à 50m par pas de 1m
    indices = [int(z / 0.25) for z in depths]

    vsz_values = []  # Liste pour stocker les valeurs VsZ

    for idx in indices:
        # Extraire les sous-labels jusqu'à l'indice donné
        label_cut = label[:idx]

        # Calculer les épaisseurs des couches (0.25m par changement de vitesse)
        thickness = []
        layer_thickness = 0.25

        for i in range(1, len(label_cut)):
            if label_cut[i] == label_cut[i - 1]:
                layer_thickness += 0.25
            else:
                thickness.append(layer_thickness)
                layer_thickness = 0.25

        # Ajouter l'épaisseur de la dernière couche
        thickness.append(layer_thickness)

        # Extraire les vitesses des couches
        vs_layers = []
        for i in range(1, len(label_cut)):
            if label_cut[i] != label_cut[i - 1]:
                vs_layers.append(label_cut[i - 1][0])

        # Ajouter la vitesse de la dernière couche
        vs_layers.append(label_cut[-1][0])

        # Calculer VsZ avec la formule : VsZ = Z / sum(hi / Vi)
        hi = np.array(thickness)
        vi = np.array(vs_layers)

        if len(hi) != len(vi):
            raise ValueError("Mismatch entre épaisseurs et vitesses !")

        Z = idx * 0.25  # Profondeur actuelle
        vsz = Z / np.sum(hi / vi)
        vsz_values.append(vsz)

    return np.array(vsz_values)


def normalize_labels(train_dataloader, val_dataloader, test_dataloader):
    """
    Normalize the labels in the dataloaders by dividing them by the maximum label value.

    Args:
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, max_label)
            - The dataloaders with normalized labels.
            - The maximum label value found across all datasets.
    """
    # Calculate the maximum label value across all datasets
    max_labelVP = 0
    for dataset in [train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset]:
        for i in range(len(dataset.labelsVP)):
            max_labelVP = max(max_labelVP, dataset.labelsVP[i].max())

    print('Max label VP:', max_labelVP)

    # Normalize the labels by dividing them by max_label
    for dataset in [train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset]:
        for i in range(len(dataset.labelsVP)):
            dataset.labelsVP[i] = dataset.labelsVP[i] / max_labelVP

    #do the same for VS:
    max_labelVS = 0
    for dataset in [train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset]:
        for i in range(len(dataset.labelsVS)):
            max_labelVS = max(max_labelVS, dataset.labelsVS[i].max())

    print('Max label VS:', max_labelVS)

    # Normalize the labels by dividing them by max_label
    for dataset in [train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset]:
        for i in range(len(dataset.labelsVS)):
            dataset.labelsVS[i] = dataset.labelsVS[i] / max_labelVS

    return train_dataloader, val_dataloader, test_dataloader, max_labelVP, max_labelVS


def plot_random_samples(train_dataloader=None, num_samples=5,od='None',max_labelVS=None,id='None'):
    print('Display some random samples from the dataloaders')
    #verify od is not None:
    if od=='None':
        print('no Output direction provided, figure will not be saved !')
    #verify train_dataloader is not None:
    if train_dataloader==None:
        raise ValueError('No dataloader provided, cannot plot samples')
    # metion max_label as well:
    if max_labelVS==None:
        print('No maximal value provided for the label, using 6000 m/s as default value')
        max_labelVS=2000

    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))

    #if i == 1 make only one plot
    if num_samples == 1:
        if id != 'None':
            idx = id
        else:
            # Sélectionner un échantillon aléatoire
            idx = random.randint(0, len(train_dataloader.dataset) - 1)

        sample = train_dataloader.dataset[idx]
        image = sample['data']
        print('[random samples] shape image:',np.shape(image))
        labelVS = sample['label_VS']


        # image = train_dataloader.dataset.data[idx]
        # label= train_dataloader.dataset.labels[idx]

        # Afficher l'image dans la première colonne
        axs[0].imshow(image[0], aspect='auto', cmap='gray')
        axs[0].set_title(f'Shot Gather')
        axs[0].set_xlabel('Distance (m)')
        axs[0].set_ylabel('Time (sample)')

        # Afficher le label dans la deuxième colonne
        axs[1].plot(labelVS * max_labelVS, range(len(labelVS)))
        axs[1].invert_yaxis()
        axs[1].set_xlabel('Vs (m/s)')
        # set x lim as max_labelVS:
        axs[1].set_xlim(0,max_labelVS)
        axs[1].set_ylabel('Depth (m)')
        axs[1].set_title(f'Vs Depth')

    else:
        for i in range(num_samples):
            # Sélectionner un échantillon aléatoire
            idx = random.randint(0, len(train_dataloader.dataset) - 1)

            sample = train_dataloader.dataset[idx]
            image = sample['data']
            print('[random samples] shape image:', np.shape(image))
            labelVS = sample['label_VS']

            # image = train_dataloader.dataset.data[idx]
            # label= train_dataloader.dataset.labels[idx]

            # Afficher l'image dans la première colonne
            axs[i, 0].imshow(image[0], aspect='auto', cmap='gray')
            axs[i, 0].set_title(f'Shot Gather {i + 1}')
            axs[i, 0].set_xlabel('Distance (m)')
            axs[i, 0].set_ylabel('Time (sample)')

            # Afficher le label dans la deuxième colonne
            axs[i, 1].plot(labelVS * max_labelVS, range(len(labelVS)))
            axs[i, 1].invert_yaxis()
            axs[i, 1].set_xlabel('Vs (m/s)')
            # set x lim as max_labelVS:
            axs[i,1].set_xlim(0, max_labelVS)
            axs[i, 1].set_ylabel('Depth (m)')
            axs[i, 1].set_title(f'Vs Depth {i + 1}')

    plt.tight_layout()
    if od == None:
        plt.show()
        plt.close()
    else:
        plt.savefig(f'figures/{od}/random_samples.pdf', format='pdf',dpi=300)
    plt.close()


def print_matrix(array, pretext=''):
    from IPython.display import display, Math
    if array.dtype == 'int64':
        fstring = ' %d '
    else:
        fstring = ' %.3f '

    data = ''
    for line in array:
        if not hasattr(line, '__len__'):
            data += fstring%line + r' \\'
            continue
        if len(line) == 1:
            data += fstring%line + r'& \\\n'
            continue
        for element in line:
            data += fstring%element + r'&'
        data = data[:-1] + r'\\' + '\n'
    display(Math(pretext+'\\begin{bmatrix} \n%s\end{bmatrix}'%data))


import numpy as np
from scipy.sparse.linalg import lsqr

def lm(fun, p0, tol, maxiter, maxinner=None):
    """
    Minimize a nonlinear least-squares problem using the Levenberg-Marquardt algorithm.

    Parameters
    ----------
    fun : callable
        Function returning the residual vector F and the Jacobian matrix J.
        Signature: F, J = fun(p), where:
            - F is the residual vector (n,).
            - J is the Jacobian matrix (n, m) or a LinearOperator.
    p0 : array-like
        Initial guess for the parameters (m,).
    tol : float
        Stopping tolerance. The algorithm stops if the gradient norm, function change,
        or parameter change is below this threshold.
    maxiter : int
        Maximum number of iterations allowed.
    maxinner : int, optional
        Maximum number of iterations for the inner loop (lsqr). Default: None.

    Returns
    -------
    pstar : array-like
        Best solution found (m,).
    k : int
        Number of iterations performed.
    """

    # Validate inputs
    p0 = np.asarray(p0)
    if p0.ndim != 1:
        raise ValueError("p0 must be a 1D array.")
    if not callable(fun):
        raise ValueError("fun must be a callable function.")

    # Initialize p and oldp
    p = p0.reshape((-1, 1))
    n = p0.shape[0]
    Fp, J = fun(p)
    fp = np.linalg.norm(Fp) ** 2
    oldp = p0 * 2
    oldfp = fp * 2

    # Initialize lambda
    lam = 0.0001

    # Main loop
    k = 0
    while k <= maxiter:
        # Compute rhs = -J' * Fp
        rhs = -J.T @ Fp

        # Check termination criteria
        if ((np.linalg.norm(rhs) < np.sqrt(tol) * (1 + np.abs(fp))) and
                (np.abs(oldfp - fp) < tol * (1 + np.abs(fp))) and
                (np.linalg.norm(oldp - p) < np.sqrt(tol) * (1 + np.linalg.norm(p)))):
            return p.flatten(), k

        # Solve the least-squares problem
        s = lsqr(J, -Fp, damp=np.sqrt(lam), iter_lim=maxinner)[0].reshape((-1, 1))

        # Compute the new residual and Jacobian
        Fpnew, Jnew = fun(p + s)
        fpnew = np.linalg.norm(Fpnew) ** 2

        # Update parameters if the solution improves
        if fpnew < fp:
            oldp = p
            oldfp = fp
            p = p + s
            fp = fpnew
            Fp = Fpnew
            J = Jnew
            lam /= 2
            lam = max(lam, 1.0e-12)  # Prevent lambda from becoming too small
        else:
            # Increase lambda and try again
            lam *= 2.5
            lam = min(lam, 1.0e16)  # Prevent lambda from becoming too large

        # Update iteration count
        k += 1

    # Return if maxiter is exceeded
    return p.flatten(), k
