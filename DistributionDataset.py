'''
Purpose of this code is to analyse the distribution of a given pytorch dataset.
Samples can be divided into data matrices (shot gathers) and labels vecors (Vs profiles).
The data shape is (b, c, h, w) where b is the batch size, c is the number of channels, h is the height and w is the width.
The lines and columns of the images correspond to time samples and traces respectively.
The label shape is (l,1) where l is the length of the label vector.
Each value of the label vector correspond to Vs at a given depth.

The idea behind this code is to plot the distribution of the labels value in the dataset.
It allows to see if the dataset is balanced or biased towards some values.
If biased, the NN will have a hard time and converge to a suboptimal solution.

We also want to plot maximum 25 shot gathers to see if there are some patterns in the data.
We will create several figures, with maximum 6 shot gathers per figure.
'''
# Import libraries -----------------------------------------------------------
import argparse
#use GPU 3:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np

from Utilities import *
from torch.utils.data import Dataset
from PhaseShiftMethod import *
from PytorchDataset import CustomDataset

# Define the arguments -------------------------------------------------------
parser = argparse.ArgumentParser(description='Dataset distribution analysis')
parser.add_argument('--dataset_name','-data', type=str,default='Halton_debug', required=False,
                    help='Name of the dataset to use, choose between \n Dataset1Dsmall \n Dataset1Dbig \n TutorialDataset')
args = parser.parse_args()


# Set correct input and output paths ------------------------------------------
dataset_name = args.dataset_name
if dataset_name == 'Halton_debug' or dataset_name == 'Halton_Dataset':
    data_path ='/home/rbertille/data/pycharm/ViT_project/pycharm_ViT/DatasetGeneration/Datasets/'
else:
    data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
files_path = os.path.join(data_path, dataset_name)

train_folder = glob.glob(f'{files_path}/train/*')
validate_folder = glob.glob(f'{files_path}/validate/*')
test_folder = glob.glob(f'{files_path}/test/*')

setup_directories(name=dataset_name)

# Create the dataset ----------------------------------------------------------
print('\nLOAD THE DATA')

def create_one_dataset(data_path, dataset_name):
    # Get the folders of the dataset
    train_folder = glob.glob(os.path.join(data_path, dataset_name, 'train', '*'))
    validate_folder = glob.glob(os.path.join(data_path, dataset_name, 'validate', '*'))
    test_folder = glob.glob(os.path.join(data_path, dataset_name, 'test', '*'))

    # Create the sub datasets
    train_dataset = CustomDataset(train_folder,use_dispersion=True)
    validate_dataset = CustomDataset(validate_folder,use_dispersion=True)
    test_dataset = CustomDataset(test_folder,use_dispersion=True)

    # Concatenate them
    Dataset = torch.utils.data.ConcatDataset([train_dataset, validate_dataset, test_dataset])

    return Dataset

# Create the dataset using above class and function:
Dataset = create_one_dataset(data_path, dataset_name)

# Get all the labels from the dataset and all the shot gathers ----------------
labels = []
labelsVP = []
shotgathers = []
for i in range(len(Dataset)):
    #if labels_VP is present in the dataset, print smthing:
    if 'label_VP' in Dataset[i]:
        print('label_VP is present in the dataset')
        #print('len of label_VP:', len(Dataset[i]['label_VP']))
    elif 'label_VS' in Dataset[i]:
        print('label_VS is present in the dataset')
        #print('len of label_VS:', len(Dataset[i]['label_VS']))
    else:
        print('no label in the dataset')
    labels.append(Dataset[i]['label_VS'].numpy())
    labelsVP.append(Dataset[i]['label_VP'].numpy())
    shotgathers.append(Dataset[i]['data'].numpy())
# count the total number of labels
# Plot the distribution of the labels -----------------------------------------
def plot_distribution(labels, dataset_name):
    '''
    Cette fonction crée une matrice de distribution des labels dans le dataset.
    Chaque ligne représente une profondeur, chaque colonne correspond à une bin de vitesse (50 m/s).
    Chaque cellule de la matrice est le pourcentage d'occurrences de la vitesse donnée à la profondeur donnée.
    '''
    # Définir les bins
    bin_size = 50
    max_label = np.max(labels)
    bins = np.arange(0, max_label + bin_size, bin_size)

    # Créer la matrice
    distribution = np.zeros((labels[0].shape[0], len(bins) - 1))

    print(f'Nombre de labels: {len(labels)}')
    print(f'Longueur des labels: {labels[0].shape[0]}')
    print(f'Vitesse minimale : {np.min(labels)} m/s')
    print(f'Vitesse maximale : {max_label} m/s')

    print('------------------------------------')
    print('\nCRÉATION DE LA MATRICE DE DISTRIBUTION')
    for label in labels:
        for j, depth in enumerate(label):
            for k in range(len(bins) - 1):
                if bins[k] <= depth < bins[k + 1]:
                    distribution[j,k] += 1

    print(f'Taille de la matrice de distribution: {distribution.shape}')

    print('------------------------------------')
    print('\nCALCUL DES STATISTIQUES')
    # Statistiques
    occurrences = distribution.flatten()
    mean_occurrences = np.mean(occurrences)
    std_occurrences = np.std(occurrences)
    coefficient_variation = std_occurrences / mean_occurrences

    print(f'Moyenne des occurrences : {mean_occurrences}')
    print(f'Écart-type des occurrences : {std_occurrences}')
    print(f'Coefficient de variation : {coefficient_variation}')

    print('------------------------------------')
    print('\nAFFICHAGE DE LA MATRICE DE DISTRIBUTION')
    # Affichage de la matrice
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(distribution, cmap='viridis', aspect='auto')
    fig.colorbar(cax)

    ax.set_xlabel('Velocity bins (m/s)')
    ax.set_ylabel('Depth (m)')

    # Configurer les ticks des axes
    ax.set_yticks(np.linspace(0, distribution.shape[0] - 1, 6))
    ax.set_yticklabels(np.arange(0, 51, 10))

    xticks = np.linspace(0, len(bins) - 1, 10, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(bins[xticks])

    ax.set_title(f'Vs distribution inside the dataset {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'figures/{dataset_name}/distribution_{dataset_name}.pdf',bbox_inches='tight')
    plt.close()
    #plt.show()

    print(f'Le plot est disponible ici: figures/distribution_{dataset_name}.pdf')

    return distribution


def plot_distribution_oral(labels, dataset_name):
    '''
    Cette fonction crée une matrice de distribution des labels dans le dataset.
    Chaque ligne représente une profondeur, chaque colonne correspond à une bin de vitesse (50 m/s).
    Chaque cellule de la matrice est le pourcentage d'occurrences de la vitesse donnée à la profondeur donnée.
    '''
    # Définir les bins
    bin_size = 50
    max_label = np.max(labels)

    bins = np.arange(0, max_label + bin_size, bin_size)

    # Créer la matrice
    distribution = np.zeros((labels[0].shape[0], len(bins) - 1))
    print('distribution shape:', distribution.shape)

    print(f'Nombre de labels: {len(labels)}')
    print(f'Longueur des labels: {labels[0].shape[0]}')
    print(f'Vitesse minimale : {np.min(labels)} m/s')
    print(f'Vitesse maximale : {max_label} m/s')

    print('------------------------------------')
    print('\nCRÉATION DE LA MATRICE DE DISTRIBUTION')
    for label in labels:
        for j, depth in enumerate(label):
            for k in range(len(bins) - 1):
                if bins[k] <= depth < bins[k + 1]:
                    distribution[j,k] += 1

    # Calcul du vecteur moyen des labels à chaque profondeur
    average_label = np.mean(labels, axis=0)  # (N_profondeurs,)
    average_label_bins = np.digitize(average_label, bins) - 1  # Trouver l'indice du bin
    average_label_bins = np.clip(average_label_bins, 0, len(bins) - 2)  # S'assurer que c'est dans les limites

    print(f'Taille de la matrice de distribution: {distribution.shape}')

    print('------------------------------------')
    print('\nCALCUL DES STATISTIQUES')
    # Statistiques
    occurrences = distribution.flatten()
    mean_occurrences = np.mean(occurrences)
    std_occurrences = np.std(occurrences)
    coefficient_variation = std_occurrences / mean_occurrences

    print(f'Moyenne des occurrences : {mean_occurrences}')
    print(f'Écart-type des occurrences : {std_occurrences}')
    print(f'Coefficient de variation : {coefficient_variation}')

    print('------------------------------------')
    print('\nAFFICHAGE DE LA MATRICE DE DISTRIBUTION')
    # Affichage de la matrice
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(distribution, cmap='viridis', aspect='auto',extent=[0, distribution.shape[1], distribution.shape[0]-1,0])
    #plot average label in white, dashline:
    ax.plot(average_label_bins, np.arange(len(average_label)), color='white', linestyle='--', linewidth=2)
    # put a labl for colorbar
    cbar = plt.colorbar(cax)
    cbar.set_label('Occurences', rotation=270, labelpad=20, fontsize=15)

    ax.set_xlabel('Velocity bins (m/s)', fontsize=16)
    ax.set_ylabel('Depth (m)', fontsize=16)

    # Configurer les ticks des axes
    ax.set_yticks(np.linspace(0, distribution.shape[0] - 1, 6))
    ax.set_yticklabels(np.arange(0, 51, 10), fontsize=12)

    xticks = np.linspace(0, len(bins) - 1, 10, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(bins[xticks], fontsize=12)

    plt.tight_layout()
    plt.savefig(f'figures/{dataset_name}/distribution_{dataset_name}_oral.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()

    print(f'Le plot est disponible ici: figures/distribution_{dataset_name}_oral.pdf')

    return distribution
# Plot the distribution
#distribution=plot_distribution(labels, dataset_name)

# Create 2D histogramms associated with distribution 3D plot to give the occurences at a given depth
def plot_2D_histograms(labels, dataset_name):
    #another method to plot 2D histograms.
    # 1 on va définir un vecteur depth
    depths=[10,30,50]

    dz=0.5

    for depth in depths:
        # 2 on va lire tous les labels à la profondeur donnée:
        Vs_z_list = []
        for label in labels:
            #print('label shape:', label.shape)
            Vs_z= label[int(depth/dz)-1]
            Vs_z_list.append(Vs_z[0])

        print('Vs_z_list:', Vs_z_list)
        print('len Vs_z_list:', len(Vs_z_list))

        # 3 on va créer les bins
        Vmax=np.max(labels)
        bins = np.arange(0, Vmax + 50, 50)  # Ajuster les bins en fonction de la vitesse moyenne

        # 4 on va créer l'histogramme les colonnes seront rouges
        plt.hist(Vs_z_list, bins=bins, edgecolor='black', color='red')  # Utiliser Vs_z_list pour l'histogramme
        plt.xlabel(f'Vs{depth} (m/s)')
        plt.ylabel('Occurrences')
        plt.title(f'histogram showing the Vs distribution at {depth}m ({dataset_name})')
        plt.savefig(f'figures/{dataset_name}/histogram_Vs_{dataset_name}_{depth}m.pdf',bbox_inches='tight')
        #plt.show()
        plt.close()


#plot_2D_histograms(labels, dataset_name)

#create histograms of the value of thickness of the layers
# labels = Vs = f(depth)
def plot_histograms_thickness(labels, dataset_name):
    '''
    Cette fonction crée des histogrammes des épaisseurs des couches pour tous les modèles.
    '''
    print('------------------------------------')
    print('\nPLOT DES HISTOGRAMMES DES ÉPAISSEURS MOYENNES DES COUCHES')

    thickness_list = []

    # 1rst step: calculate the mean thickness
    for label in labels:
        thickness = []
        layer_thickness = 0
        for i in range(1, len(label)):
            if label[i] == label[i - 1]:
                layer_thickness += 0.5  # chaque label correspond à 0,25m
            else:
                thickness.append(layer_thickness)
                layer_thickness = 0.5  # réinitialiser pour la nouvelle couche
        thickness.append(layer_thickness)  # Ajouter la dernière couche

        thickness_list.append(thickness)
        # create one big vector with all the thicknesses
    thickness_list = np.concatenate(thickness_list).ravel()
    print('thickness list shape:', thickness_list.shape)

    # 2nd step : plot the histogram
    bins = np.arange(0, max(thickness_list) + 1, 1)  # Ajuster les bins en fonction des épaisseurs moyennes
    plt.hist(thickness_list, bins=bins, edgecolor='black')  # Utiliser mean_thickness_list pour l'histogramme

    # plot parameters
    plt.xlabel('Thickness of the layers (m)')
    plt.ylabel('Occurrences')
    plt.title(f'histogram showing the thickness of the layers ({dataset_name})')

    # display
    #plt.show()
    plt.savefig(f'figures/{dataset_name}/histogram_thickness_{dataset_name}.pdf',bbox_inches='tight')
    plt.close()

def plot_histograms_thickness_oral(labels, dataset_name):
    '''
    Cette fonction crée des histogrammes des épaisseurs des couches pour tous les modèles.
    '''
    print('------------------------------------')
    print('\nPLOT DES HISTOGRAMMES DES ÉPAISSEURS MOYENNES DES COUCHES')

    thickness_list = []

    # 1rst step: calculate the mean thickness
    for label in labels:
        thickness = []
        layer_thickness = 0
        for i in range(1, len(label)):
            if label[i] == label[i - 1]:
                layer_thickness += 0.5  # chaque label correspond à 0,25m
            else:
                thickness.append(layer_thickness)
                layer_thickness = 0.5  # réinitialiser pour la nouvelle couche
        thickness.append(layer_thickness)  # Ajouter la dernière couche

        thickness_list.append(thickness)
        # create one big vector with all the thicknesses
    thickness_list = np.concatenate(thickness_list).ravel()

    # 2nd step : plot the histogram
    bins = np.arange(0, max(thickness_list) + 1, 1)  # Ajuster les bins en fonction des épaisseurs moyennes
    plt.hist(thickness_list, bins=bins, edgecolor='black')  # Utiliser mean_thickness_list pour l'histogramme

    # plot parameters
    plt.xlabel('Thickness of the layers (m)', fontsize=15)
    plt.ylabel('Occurrences', fontsize=15)

    # display
    plt.savefig(f'figures/{dataset_name}/histogram_thickness_{dataset_name}_oral.pdf',format='pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
# Create the histogram
#plot_histograms_thickness(labels, dataset_name)

# Now we want to plot histogramms of the average number of layers in a model
def plot_histograms_layers(labels, dataset_name):
    '''
    Cette fonction crée des histogrammes du nombre de couches pour chaque modèle.
    '''
    print('------------------------------------')
    print('\nPLOT DES HISTOGRAMMES DU NOMBRE DE COUCHES')

    nb_layers_list = []

    # 1rst step: calculate the number of layers
    for label in labels:
        layers = 1
        for i in range(1, len(label)):
            if label[i] != label[i - 1]:
                layers += 1
        nb_layers_list.append(layers)

    # 2nd step : plot the histogram
    bins = np.arange(1, max(nb_layers_list) + 2, 1)  # Ajuster les bins pour inclure la dernière valeur
    counts, edges, patches = plt.hist(nb_layers_list, bins=bins, edgecolor='black', color='royalblue', align='left')

    # Annoter les barres avec les valeurs
    for count, edge in zip(counts, edges[:-1]):
        plt.text(edge + 0.5, count + 0.1, str(int(count)), ha='center', fontsize=12, fontweight='bold', color='black')

    # plot parameters
    plt.xticks(np.arange(1, max(nb_layers_list) + 1, 1))  # S'assurer que les ticks sont bien placés
    plt.xlabel('Nombre de couches')
    plt.ylabel('Occurrences')
    plt.title(f'Histogramme du nombre de couches par modèle ({dataset_name})')

    # Save & show
    plt.savefig(f'figures/{dataset_name}/histogram_layers_{dataset_name}.pdf', bbox_inches='tight')
    plt.close()

def plot_histograms_layers_oral(labels, dataset_name):
    '''
    Cette fonction crée des histogrammes du nombre de couches pour chaque modèle.
    '''
    print('------------------------------------')
    print('\nPLOT DES HISTOGRAMMES DU NOMBRE DE COUCHES')

    nb_layers_list = []

    # 1rst step: calculate the number of layers
    for label in labels:
        layers = 1
        for i in range(1, len(label)):
            if label[i] != label[i - 1]:
                layers += 1
        nb_layers_list.append(layers)

    # 2nd step : plot the histogram
    bins = np.arange(1, max(nb_layers_list) + 2, 1)  # Inclut bien toutes les valeurs possibles
    counts, edges, patches = plt.hist(nb_layers_list, bins=bins, edgecolor='black', color='royalblue', align='left')

    # Annoter les barres avec les valeurs
    for count, edge in zip(counts, edges[:-1]):
        plt.text(edge + 0.5, count + 0.1, str(int(count)), ha='center', fontsize=12, fontweight='bold', color='black')

    # plot parameters
    plt.xticks(np.arange(1, max(nb_layers_list) + 1, 1))  # Assure un affichage propre des valeurs
    plt.xlabel('Nombre de couches', fontsize=15)
    plt.ylabel('Occurrences', fontsize=15)
    plt.title(f'Histogramme du nombre de couches ({dataset_name})', fontsize=15)

    # Save & show
    plt.savefig(f'figures/{dataset_name}/histogram_layers_{dataset_name}_oral.pdf', bbox_inches='tight', format='pdf')
    plt.close()
# Create the histogram
#plot_histograms_layers(labels, dataset_name)

# histogram number of contacts (change in Vs) with depth
def plot_histograms_contacts(labels, dataset_name):
    '''
    Cette fonction crée des histogrammes du nombre de contacts (changement de Vs) avec la profondeur.
    '''
    print('------------------------------------')
    print('\nPLOT DES HISTOGRAMMES DU NOMBRE DE CONTACTS')

    contacts_list = []

    # 1rst step: calculate the number of contacts
    for label in labels:
        contacts = 0
        for i in range(1, len(label)):
            if label[i] != label[i - 1]:
                contacts += 1
        contacts_list.append(contacts)

    # 2nd step : plot the histogram
    bins = np.arange(0, max(contacts_list) + 1, 1)  # Ajuster les bins en fonction du nombre de contacts
    plt.hist(contacts_list, bins=bins, edgecolor='black')  # Utiliser contacts_list pour l'histogramme

    # plot parameters
    plt.xlabel('Number of contacts')
    plt.ylabel('Occurrences')
    plt.title(f'histogram showing the number of contacts ({dataset_name})')

    # display
    #plt.show()
    plt.savefig(f'figures/{dataset_name}/histogram_contacts_{dataset_name}.pdf',bbox_inches='tight')
    plt.close()

# Create the histogram
#plot_histograms_contacts(labels, dataset_name)

class VSz():
    def __init__(self, labels, dataset_name,z):
        self.labels = labels
        self.Vsz_list = []

    def calculate_vsz(self, labels, z):
        # 1ere étape: lire les labels jusqu'à la profondeur donnée
        zz = int(z / 0.5)
        labels_cut = []
        for label in labels:
            label_cut = label[:zz]
            labels_cut.append(label_cut)

        # 2eme étape: calculer le nombre de couches dans les labels tronqués
        all_h = []
        for label in labels_cut:
            thickness = []
            layer_thickness = 0.5  # commencer avec une épaisseur initiale
            for i in range(1, len(label)):
                if label[i] == label[i - 1]:
                    layer_thickness += 0.5  # chaque label correspond à 0.5m
                else:
                    thickness.append(layer_thickness)
                    layer_thickness = 0.5  # réinitialiser pour la nouvelle couche
            # Ajouter l'épaisseur de la dernière couche
            thickness.append(layer_thickness)
            all_h.append(thickness)

        # 3eme étape: calculer la vitesse de chaque couche
        Vs_z_list = []
        for label in labels_cut:
            Vs_label = []
            for i in range(1, len(label)):
                if label[i] != label[i - 1]:
                    Vs = label[i - 1]  # ou convertir en vitesse si nécessaire
                    Vs_label.append(Vs[0])  # vérifier que la structure de tes labels est correcte ici
            # Ajouter la vitesse de la dernière couche
            Vs_label.append(label[-1][0])
            Vs_z_list.append(Vs_label)

        # 4eme étape: calculer VsZ pour chaque modèle
        Vsz_list = []
        for i in range(len(Vs_z_list)):
            Vi = np.array(Vs_z_list[i])  # vitesses des couches
            hi = np.array(all_h[i])  # épaisseurs des couches

            # Vérification de la correspondance des tailles entre hi et Vi
            if len(hi) != len(Vi):
                print(f"Warning: mismatch between layer thicknesses and velocities for model {i}")
                continue

            # Formule pour VsZ: Vs(z) = z / sum(hi/Vi)
            Vsz = z / np.sum(hi / Vi)
            Vsz_list.append(Vsz)

        return Vsz_list

    def plot_histograms_VSz(self,labels, dataset_name, z,wave='Vs'):
        '''
        Cette fonction crée des histogrammes de la vitesse moyenne Vs(z) pour une profondeur donnée z.
        '''
        print('------------------------------------')
        print('\nPLOT DES HISTOGRAMMES DE LA VITESSE MOYENNE Vs(z)')

        Vsz_list= self.calculate_vsz(labels,z)

        # 5eme étape: plot de l'histogramme
        # creer les bins:
        Vmax = np.max(Vsz_list)
        bins = np.arange(0, Vmax + 50, 50)  # Ajuster les bins en fonction de la vitesse moyenne

        # plot l'histogramme
        plt.hist(Vsz_list, bins=bins, edgecolor='black')  # Utiliser Vs_z_list pour l'histogramme
        plt.xlabel(f'{wave}{z} (m/s)')
        plt.ylabel('Occurrences')
        plt.title(f'histogram showing the {wave}{z} distribution ({dataset_name})')
        plt.savefig(f'figures/{dataset_name}/histogram_{wave}_{dataset_name}_{z}m.pdf',bbox_inches='tight')
        #plt.show()
        plt.close()

    def plot_histograms_VSz_evolution_subfigures(self,labels, dataset_name, z):
        '''
        Cette fonction crée des sous-figures d'histogrammes pour la vitesse moyenne Vs(z)
        à différentes profondeurs, avec des axes x partagés et des sous-figures collées.
        L'axe y représente le pourcentage d'occurrences.
        '''
        print('------------------------------------')
        print('\nPLOT DES HISTOGRAMMES DE L\'ÉVOLUTION DE LA VITESSE MOYENNE Vs(z) AVEC POURCENTAGE D\'OCCURRENCES')

        # Crée une figure avec des sous-figures verticales
        fig, axes = plt.subplots(len(z), 1, sharex=True, figsize=(8, 12),
                                 gridspec_kw={'hspace': 0.06})  # hspace pour réduire l'espacement

        fig.subplots_adjust(top=0.95)

        colors = plt.cm.viridis(np.linspace(0, 1, len(z)))  # Palette de couleurs

        max_vs = 0
        for idx, depth in enumerate(z):
            zz = int(depth / 0.5)

            # 1ère étape : calcul de la vitesse moyenne Vs(z)
            Vs_z_list= self.calculate_vsz(labels,depth)

            # Mise à jour de la valeur maximale pour ajuster les bins
            max_vs = max(Vs_z_list)
            #print('max_vs:', max_vs)

            # 2ème étape : plot de l'histogramme pour cette profondeur avec occurrences
            bins = np.arange(0, max_vs + 50, 50)

            # Histogramme avec densité relative (pourcentage des occurrences)
            counts, bin_edges = np.histogram(Vs_z_list, bins=bins)
            total_counts = np.sum(counts)
            percentages = (counts / total_counts) * 100  # Calcul des pourcentages

            # Affichage sous forme d'histogramme
            axes[idx].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges),
                          alpha=0.7, edgecolor='black', color=colors[idx], align='edge')

            # Paramètres de la sous-figure
            axes[idx].set_ylabel(f'Occurences (%)\nVs(z={depth}m)', fontsize=10)
            axes[idx].grid(True)

            # Suppression des ticks x pour toutes les sous-figures sauf la dernière
            if idx < len(z) - 1:
                axes[idx].tick_params(labelbottom=False)

        # Paramètres de l'axe x partagé
        axes[-1].set_xlabel(f' Vs(z) (m/s)', fontsize=12)

        # Réduction de l'espace autour de la figure pour coller les sous-figures
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Titre global
        fig.suptitle(f'Evolution of Vs(z) distribution ({dataset_name})', fontsize=14)

        # Sauvegarde de la figure
        plt.savefig(f'figures/{dataset_name}/histogram_evolution_VSz_subfig_{dataset_name}.pdf',bbox_inches='tight')
        #plt.show()
        plt.close()

# Create the histogram for z=10m
#VSz_10=VSz(labels, dataset_name, 10)
#VSz_10.plot_histograms_VSz(labels, dataset_name, 10)
# Create the histogram for z=20m
#VSz_20=VSz(labels, dataset_name, 20)
#VSz_20.plot_histograms_VSz(labels, dataset_name, 20)
# Create the histogram for z=30m
#VSz_30=VSz(labels, dataset_name, 30)
#VSz_30.plot_histograms_VSz(labels, dataset_name, 30)
# Create the histogram for z=40m
#VSz_40=VSz(labels, dataset_name, 40)
#VSz_40.plot_histograms_VSz(labels, dataset_name, 40)
# Create the histogram for z=50m
#VSz_50=VSz(labels, dataset_name, 50)
#VSz_50.plot_histograms_VSz(labels, dataset_name, 50)

# Create the histogram for z=10m to 50m
#VSz_evolution=VSz(labels, dataset_name, [10,20,30,40,50])
#VSz_evolution.plot_histograms_VSz_evolution_subfigures(labels, dataset_name, [10,20,30,40,50])
class VSz_oral():
    def __init__(self, labels, dataset_name,z,wave='Vs'):
        self.labels = labels
        self.Vsz_list = []
        self.wave = wave

    def calculate_vsz(self, labels, z):
        # 1ere étape: lire les labels jusqu'à la profondeur donnée
        zz = int(z / 0.5)
        labels_cut = []
        for label in labels:
            label_cut = label[:zz]
            labels_cut.append(label_cut)

        # 2eme étape: calculer le nombre de couches dans les labels tronqués
        all_h = []
        for label in labels_cut:
            thickness = []
            layer_thickness = 0.5  # commencer avec une épaisseur initiale
            for i in range(1, len(label)):
                if label[i] == label[i - 1]:
                    layer_thickness += 0.5  # chaque label correspond à 0.5m
                else:
                    thickness.append(layer_thickness)
                    layer_thickness = 0.5  # réinitialiser pour la nouvelle couche
            # Ajouter l'épaisseur de la dernière couche
            thickness.append(layer_thickness)
            all_h.append(thickness)

        # 3eme étape: calculer la vitesse de chaque couche
        Vs_z_list = []
        for label in labels_cut:
            Vs_label = []
            for i in range(1, len(label)):
                if label[i] != label[i - 1]:
                    Vs = label[i - 1]  # ou convertir en vitesse si nécessaire
                    Vs_label.append(Vs[0])  # vérifier que la structure de tes labels est correcte ici
            # Ajouter la vitesse de la dernière couche
            Vs_label.append(label[-1][0])
            Vs_z_list.append(Vs_label)

        # 4eme étape: calculer VsZ pour chaque modèle
        Vsz_list = []
        for i in range(len(Vs_z_list)):
            Vi = np.array(Vs_z_list[i])  # vitesses des couches
            hi = np.array(all_h[i])  # épaisseurs des couches

            # Vérification de la correspondance des tailles entre hi et Vi
            if len(hi) != len(Vi):
                print(f"Warning: mismatch between layer thicknesses and velocities for model {i}")
                continue

            # Formule pour VsZ: Vs(z) = z / sum(hi/Vi)
            Vsz = z / np.sum(hi / Vi)
            Vsz_list.append(Vsz)

        return Vsz_list

    def plot_histograms_VSz(self,labels, dataset_name, z):
        '''
        Cette fonction crée des histogrammes de la vitesse moyenne Vs(z) pour une profondeur donnée z.
        '''
        print('------------------------------------')
        print('\nPLOT DES HISTOGRAMMES DE LA VITESSE MOYENNE Vs(z)')

        Vsz_list= self.calculate_vsz(labels,z)

        # 5eme étape: plot de l'histogramme
        # creer les bins:
        Vmax = np.max(Vsz_list)
        bins = np.arange(0, Vmax + 50, 50)  # Ajuster les bins en fonction de la vitesse moyenne

        # plot l'histogramme
        plt.hist(Vsz_list, bins=bins, edgecolor='black')  # Utiliser Vs_z_list pour l'histogramme
        plt.xlabel(f'{self.wave}{z} (m/s)')
        plt.ylabel('Occurrences')
        plt.title(f'histogram showing the {self.wave}{z} distribution ({dataset_name})')
        plt.savefig(f'figures/{dataset_name}/histogram_{self.wave}_{dataset_name}_{z}m.pdf',bbox_inches='tight')
        #plt.show()
        plt.close()

    def plot_histograms_VSz_evolution_subfigures(self,labels, dataset_name, z):
        '''
        Cette fonction crée des sous-figures d'histogrammes pour la vitesse moyenne Vs(z)
        à différentes profondeurs, avec des axes x partagés et des sous-figures collées.
        L'axe y représente le pourcentage d'occurrences.
        '''
        print('------------------------------------')
        print('\nPLOT DES HISTOGRAMMES DE L\'ÉVOLUTION DE LA VITESSE MOYENNE Vs(z) AVEC POURCENTAGE D\'OCCURRENCES')

        # Crée une figure avec des sous-figures verticales
        fig, axes = plt.subplots(len(z), 1, sharex=True, figsize=(8, 12),
                                 gridspec_kw={'hspace': 0.06})  # hspace pour réduire l'espacement

        fig.subplots_adjust(top=0.95)

        colors = plt.cm.viridis(np.linspace(0, 1, len(z)))  # Palette de couleurs

        max_vs = 0
        for idx, depth in enumerate(z):
            zz = int(depth /0.5)

            # 1ère étape : calcul de la vitesse moyenne Vs(z)
            Vs_z_list= self.calculate_vsz(labels,depth)


            # Mise à jour de la valeur maximale pour ajuster les bins
            max_vs = max(Vs_z_list)
            #print('max_vs:', max_vs)

            # 2ème étape : plot de l'histogramme pour cette profondeur avec occurrences
            bins = np.arange(0, max_vs + 50, 50)

            # Histogramme avec densité relative (pourcentage des occurrences)
            counts, bin_edges = np.histogram(Vs_z_list, bins=bins)
            total_counts = np.sum(counts)
            percentages = (counts / total_counts) * 100  # Calcul des pourcentages

            # Affichage sous forme d'histogramme
            axes[idx].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges),
                          alpha=0.7, edgecolor='black', color=colors[idx], align='edge')

            # Paramètres de la sous-figure
            axes[idx].set_ylabel(f'Occurences (%)\n{self.wave}(z={depth}m)', fontsize=15)
            axes[idx].grid(True)

            # Suppression des ticks x pour toutes les sous-figures sauf la dernière
            if idx < len(z) - 1:
                axes[idx].tick_params(labelbottom=False)

        # Paramètres de l'axe x partagé
        axes[-1].set_xlabel(f' {self.wave}(z) (m/s)', fontsize=15)

        # Réduction de l'espace autour de la figure pour coller les sous-figures
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Titre global
        fig.suptitle(f'Evolution of {self.wave}z : dataset préliminaire)', fontsize=18)

        # Sauvegarde de la figure
        plt.savefig(f'figures/{dataset_name}/histogram_evolution_{self.wave}z_subfig_{dataset_name}_oral.pdf',bbox_inches='tight')
        #plt.show()
        plt.close()
        print(f'evolution {self.wave}z oral done')

def plot_histograms_VSz_evolution_subfigures(labels, dataset_name, depths):
    '''
    Cette fonction crée des sous-figures d'histogrammes pour la vitesse moyenne Vs(z)
    à différentes profondeurs, avec des axes x partagés et des sous-figures collées.
    L'axe y représente le pourcentage d'occurrences.
    '''
    print('------------------------------------')
    print('\nPLOT DES HISTOGRAMMES DE L\'ÉVOLUTION DE LA VITESSE MOYENNE Vs(z) AVEC POURCENTAGE D\'OCCURRENCES')

    # Crée une figure avec des sous-figures verticales
    fig, axes = plt.subplots(len(depths), 1, sharex=True, figsize=(8, 12),
                             gridspec_kw={'hspace': 0.06})  # hspace pour réduire l'espacement

    fig.subplots_adjust(top=0.95)

    colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))  # Palette de couleurs

    max_vs = 0
    for idx, z in enumerate(depths):
        zz = int(z / 0.5)
        Vs_z_list = []

        # 1ère étape : calcul de la vitesse moyenne Vs(z)
        VSZ=VSz(labels, dataset_name, z)
        Vs_z_list= VSZ.calculate_vsz(labels=labels,z=z)

        # Mise à jour de la valeur maximale pour ajuster les bins
        max_vs = max(max_vs, max(Vs_z_list))

        # 2ème étape : plot de l'histogramme pour cette profondeur avec occurrences
        bins = np.arange(0, max_vs + 50, 50)

        # Histogramme avec densité relative (pourcentage des occurrences)
        counts, bin_edges = np.histogram(Vs_z_list, bins=bins)
        total_counts = np.sum(counts)
        percentages = (counts / total_counts) * 100  # Calcul des pourcentages

        # Affichage sous forme d'histogramme
        axes[idx].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges),
                      alpha=0.7, edgecolor='black', color=colors[idx], align='edge')

        # Paramètres de la sous-figure
        axes[idx].set_ylabel(f'Occurences (%)\nVs(z={z}m)', fontsize=10)
        axes[idx].grid(True)

        # Suppression des ticks x pour toutes les sous-figures sauf la dernière
        if idx < len(depths) - 1:
            axes[idx].tick_params(labelbottom=False)

    # Paramètres de l'axe x partagé
    axes[-1].set_xlabel('Average velocity Vs(z) (m/s)', fontsize=12)

    # Réduction de l'espace autour de la figure pour coller les sous-figures
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Titre global
    fig.suptitle(f'Evolution of Vs(z) distribution ({dataset_name})', fontsize=14)

    # Sauvegarde de la figure
    plt.savefig(f'figures/{dataset_name}/histogram2_evolution_VSz_subfig_{dataset_name}.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_histograms_averagedz(labels, dataset_name):
    '''
    Cette fonction crée des histogrammes de la moyenne des dérivées de Vs pour chaque modèle.
    '''
    print('------------------------------------')
    print('\nPLOT DES HISTOGRAMMES DE LA MOYENNE DES DÉRIVÉES DE Vs')

    mean_dz_list = []

    # 1re étape: calculer la moyenne des dérivées
    for label in labels:
        label_1d = label.flatten()  # Transformer en 1D
        if len(label_1d) > 1:
            dz = np.diff(label_1d)
            if len(dz) > 0:
                mean_dz = np.mean(dz) #pk faire la moyenne ?
                mean_dz_list.append(mean_dz)

    # Vérifier si mean_dz_list n'est pas vide
    if len(mean_dz_list) == 0:
        print("Pas de dérivées valides à afficher.")
        return

    # 2e étape : ajuster la taille des bins en fonction de la distribution des données
    bin_width = (max(mean_dz_list) - min(mean_dz_list)) / 20  # Diviser l'écart en 20 bins
    bins = np.arange(min(mean_dz_list), max(mean_dz_list) + bin_width, bin_width)

    # Créer l'histogramme
    plt.hist(mean_dz_list, bins=bins, edgecolor='black')

    # Paramètres du plot (english)
    plt.xlabel('Average derivative of Vs')
    plt.ylabel('Occurrences')
    plt.title(f'Histogram showing Average derivative of Vs ({dataset_name})')

    # Afficher le graphique
    #plt.show()
    plt.savefig(f'figures/{dataset_name}/histogram_averagedz_{dataset_name}.pdf',bbox_inches='tight')
    plt.close()

# Create the histogram
#plot_histograms_averagedz(labels, dataset_name)

def plot_histograms_dz(labels, dataset_name):
    '''
    Cette fonction crée des histogrammes de la moyenne des dérivées de Vs pour chaque modèle.
    '''
    print('------------------------------------')
    print(f'\nPLOT DES HISTOGRAMMES DES DÉRIVÉES DE Vs - {dataset_name}')

    dz_list = []

    # 1re étape: calculer la moyenne des dérivées
    for label in labels:
        label_1d = label.flatten()  # Transformer en 1D
        if len(label_1d) > 1:
            dz = np.diff(label_1d)
            dz_list.append(dz)  # ajouter le vecteur dz à la liste

    # Vérifier si dz_list n'est pas vide
    if len(dz_list) == 0:
        print("Pas de dérivées valides à afficher.")
        return

    # Fusionner tous les vecteurs dz en un seul
    dz_list = np.concatenate(dz_list).ravel()
    total_occurences = len(dz_list)

    # Calculer xmin et xmax pour 3% des valeurs
    sorted_dz = np.sort(dz_list)
    lower_limit_index = int(total_occurences * 0.03)
    upper_limit_index = total_occurences - lower_limit_index

    # Définir xmin et xmax
    xmin = sorted_dz[lower_limit_index]
    xmax = sorted_dz[upper_limit_index - 1]

    # Filtrer dz_list entre xmin et xmax
    filtered_dz = dz_list[(dz_list >= xmin) & (dz_list <= xmax)]
    total_filtered = len(filtered_dz)

    # Calculer le nombre de bins avec la contrainte de 50%
    max_occurrences_per_bin = total_occurences * 0.5 / 100  # 50% des occurrences totales
    num_bins = max(1, min(total_filtered // int(max_occurrences_per_bin) if max_occurrences_per_bin > 0 else total_filtered, total_filtered))

    # Création de l'histogramme
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_dz, bins=num_bins, weights=np.ones_like(filtered_dz) * 100 / total_filtered)  # Histogramme en pourcentage
    plt.title(f'Histogram of Vs Derivatives for {dataset_name}')
    plt.xlabel('Vs Derivative')
    plt.ylabel('Occurrences (%)')
    plt.xlim(xmin, xmax)  # Limiter l'axe des x
    plt.grid(True)

    # Afficher l'histogramme
    #plt.show()
    plt.savefig(f'figures/{dataset_name}/histogram_dz_{dataset_name}.pdf',bbox_inches='tight')
    plt.close()

#plot_histograms_dz(labels, dataset_name)

# create a figure to plot maximum 25 dispersion images
def plot_disp_images(shotgathers, dataset_name):
    '''
    This function creates a figure with 25 shot gathers and associated dispersion images.
    Each shotgather image is a 2D matrix where the lines and columns correspond to time samples and traces respectively.
    '''

    print('------------------------------------')
    print('\nPLOT THE SHOT GATHERS')
    print('shape shotgathers:', shotgathers[0].shape)
    # calculate the number of shotgathers
    nb_images = len(shotgathers)
    for max in [25, 16, 9]:
        if nb_images > max:
            nb_images = max
            dim = int(np.sqrt(max))
            break
    print(f'Number of shot gathers: {nb_images}')

    # choose images randomly so choose nb_images random indexes among the shotgathers
    indexes = np.random.choice(len(shotgathers), nb_images, replace=False)

    # Convert the image from grid points into real units
    time_vector = np.linspace(0, 1.5, shotgathers[0].shape[1])
    print(f'Time vector: {time_vector.shape}')
    nb_traces = shotgathers[0].shape[2]
    print(f'Number of traces: {nb_traces}')

    # Create the figure
    fig, axs = plt.subplots(dim, dim, figsize=(nb_images, nb_images))

    # Add the main title with some padding
    fig.suptitle(f'Shot gathers of the dataset {dataset_name}', fontsize=22, y=0.95)

    for i in range(nb_images):
        indice = indexes[i]
        ax = axs[i // dim, i % dim]
        ax.imshow(shotgathers[indice][0], aspect='auto', cmap='gray',
                  extent=[0, nb_traces, time_vector[-1], time_vector[0]])
        ax.set_title(f'Shot gather {indice}', fontsize=12)

        # Display the x and y axis labels
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Time (s)')

        # Optional: Adjust axis visibility if you want to keep gridlines
        # ax.axis('on')

    # Adjust layout to give space for the title
    plt.subplots_adjust(top=0.9)

    # Save and show the figure
    plt.savefig(f'figures/{dataset_name}/shotgathers_{dataset_name}.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()

    # Print the location of the saved figure
    print(f'The plot is available here: figures/{dataset_name}/shotgathers_{dataset_name}.pdf')

    print('------------------------------------')
    print('\nPLOT THE DISPERSION IMAGES')
    print('Loading dispersion parameters')
    fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name)
    print('shape shotgathers:', shotgathers[0].shape)
    #calculate the number of shotgathers
    nb_images = len(shotgathers)
    for max in [25,16,9]:
        if nb_images > max:
            nb_images = max
            dim= int(np.sqrt(max))
            break
    print(f'Number of dispersion images: {nb_images}')

    # Convert the image from grid points into real units
    time_vector = np.linspace(0, 1.5, shotgathers[0].shape[1])
    print(f'Time vector: {time_vector.shape}')
    nb_traces = shotgathers[0].shape[2]
    print(f'Number of traces: {nb_traces}')

    # Create the figure
    fig, axs = plt.subplots(dim, dim, figsize=(nb_images+3, nb_images+3))

    # Add the main title with some padding
    fig.suptitle(f'Dispersion images of the dataset {dataset_name}', fontsize=22, y=0.95)

    for i in range(nb_images):
        indice = indexes[i]
        #print('shape shotgathers:', shotgathers[indice].shape)
        disp=dispersion(shotgathers[indice][0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
        ax = axs[i // dim, i % dim]
        ax.imshow(disp,aspect='auto',cmap='jet')
        xticks_positions = np.linspace(0, disp.shape[1] - 1, 5).astype(int)  # 5 positions de ticks
        xticks_labels = np.round(np.linspace(np.min(c), np.max(c), 5)).astype(
            int)  # Labels correspondants aux vitesses de phase
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticks_labels, fontsize=7)

        ax.set_title(f'Dispersion image {indice}', fontsize=10)

        # Display the x and y axis labels
        ax.set_xlabel('phase velocity', fontsize=7)
        ax.set_ylabel('frequence', fontsize=7)

    # Adjust layout to give space for the title
    plt.subplots_adjust(top=0.9)

    # Save and show the figure
    plt.savefig(f'figures/{dataset_name}/disp_images_{dataset_name}.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()


    # Print the location of the saved figure
    print(f'The plot is available here: figures/{dataset_name}/disp_images_{dataset_name}.pdf')

# Plot the dispersion images
#plot_disp_images(shotgathers, dataset_name)

def plot_combined_figure(shotgathers, labels, dataset_name):
    # Parameters
    dz = 0.5
    max_label = np.max(labels)  # Assuming labels are normalized, scale them properly
    fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name,max_c=3.5)

    # Create a figure with GridSpec (2 main columns, 3 sub-columns in each main column, 5 rows)
    fig = plt.figure(figsize=(15, 25))
    gs = fig.add_gridspec(7, 3, wspace=0.3, hspace=0.5)  # 6 rows, 3 sub-columns

    # Time vector and number of traces for the shot gathers
    time_vector = np.linspace(0, 1.5, shotgathers[0].shape[1])
    nb_traces = shotgathers[0].shape[2]

    # Randomly select 5 indices to plot
    indexes = random.sample(range(len(shotgathers)), 7)

    for i, idx in enumerate(indexes):
        # Subplot for the label (first sub-column in each main column)
        ax_label = fig.add_subplot(gs[i, 0])
        depth_vector = np.arange(labels[idx].shape[0]) * dz
        ax_label.plot(labels[idx], depth_vector, label='True label')
        ax_label.invert_yaxis()
        ax_label.set_xlabel('Vs (m/s)')
        ax_label.set_ylabel('Depth (m)')
        ax_label.set_title(f'Vs Depth {idx}')
        ax_label.legend(fontsize=8)

        # Subplot for the shot gather (second sub-column)
        ax_shot = fig.add_subplot(gs[i, 1])
        ax_shot.imshow(shotgathers[idx][0], aspect='auto', cmap='gray',
                       extent=[0, nb_traces, time_vector[-1], time_vector[0]])
        ax_shot.set_xlabel('Distance (m)')
        ax_shot.set_ylabel('Time (s)')
        ax_shot.set_title(f'Shot gather {idx}')

        # Subplot for the dispersion image (third sub-column)
        ax_disp = fig.add_subplot(gs[i, 2])
        disp = dispersion(shotgathers[idx][0].T, dt, x, c, epsilon=1e-6, fmax=fmax).numpy().T
        disp = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
        ax_disp.imshow(disp, aspect='auto', cmap='jet')
        xticks_positions = np.linspace(0, disp.shape[1] - 1, 5).astype(int)
        xticks_labels = np.round(np.linspace(np.min(c), np.max(c), 5)).astype(int)
        ax_disp.set_xticks(xticks_positions)
        ax_disp.set_xticklabels(xticks_labels, fontsize=7)
        ax_disp.set_xlabel('Phase velocity (m/s)')
        ax_disp.set_ylabel('Frequency (Hz)')
        ax_disp.set_title(f'Dispersion {idx}')

    # Main title for the figure
    fig.suptitle(f'Combined plot of Shot gathers, Labels, and Dispersion Images for {dataset_name}', fontsize=22)

    # Save and show the figure
    plt.savefig(f'figures/{dataset_name}/combined_plot_{dataset_name}.pdf', format='pdf', dpi=300)
    #plt.show()
    plt.close()

    # Print the location of the saved figure
    print(f'The plot is available here: figures/{dataset_name}/combined_plot_{dataset_name}.pdf')

# Plot the combined figure
#plot_combined_figure(shotgathers, labels, dataset_name)

def plot_combined_figure_oral(shotgathers, labels, dataset_name):
    # Parameters
    dz = 0.5
    max_label = np.max(labels)  # Assuming labels are normalized, scale them properly
    fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name, max_c=3.5)

    nsamples=3
    # Create a figure with GridSpec (4 rows, 3 columns)
    fig, axes = plt.subplots(nsamples, 3, figsize=(18, 15), gridspec_kw={'wspace': 0.4, 'hspace': 0.2})

    # Time vector and number of traces for the shot gathers
    time_vector = np.linspace(0, 1.5, shotgathers[0].shape[1])
    nb_traces = shotgathers[0].shape[2]

    # Randomly select 4 indices to plot
    indexes = random.sample(range(len(shotgathers)), nsamples)

    # Titles for the columns
    column_titles = ["Profil de Vs", "Tir sismique", "Image de dispersion"]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=20)

    for row_idx, idx in enumerate(indexes):
        # Subplot for the label (first column)
        ax_label = axes[row_idx, 0]
        depth_vector = np.arange(labels[idx].shape[0]) * dz
        ax_label.plot(labels[idx], depth_vector, label='True label', linewidth=2)
        ax_label.invert_yaxis()
        #set x_lims:
        ax_label.set_xlim([0, max_label])
        if row_idx == (nsamples-1):  # Show X-axis label only on the last row
            ax_label.set_xlabel('Vs (m/s)', fontsize=17)
        else:
            ax_label.set_xticklabels([])
        ax_label.set_ylabel('Depth (m)', fontsize=17)
        ax_label.tick_params(axis='y', labelsize=14)
        ax_label.legend(fontsize=14)

        # Subplot for the shot gather (second column)
        ax_shot = axes[row_idx, 1]
        ax_shot.imshow(shotgathers[idx][0], aspect='auto', cmap='gray',
                       extent=[0, nb_traces, time_vector[-1], time_vector[0]])
        if row_idx == (nsamples-1):  # Show X-axis label only on the last row
            ax_shot.set_xlabel('Distance (m)', fontsize=17)
        else:
            ax_shot.set_xticklabels([])
        ax_shot.set_ylabel('Time (s)', fontsize=17)
        ax_shot.tick_params(axis='y', labelsize=14)

        # Subplot for the dispersion image (third column)
        ax_disp = axes[row_idx, 2]
        disp = dispersion(shotgathers[idx][0].T, dt, x, c, epsilon=1e-03, fmax=fmax).numpy().T
        disp = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
        ax_disp.imshow(disp, aspect='auto', cmap='jet')
        xticks_positions = np.linspace(0, disp.shape[1] - 1, 5).astype(int)
        xticks_labels = np.round(np.linspace(np.min(c), np.max(c), 5)).astype(int)
        ax_disp.set_xticks(xticks_positions)
        ax_disp.set_xticklabels(xticks_labels if row_idx == (nsamples-1) else [], fontsize=12)
        if row_idx == (nsamples-1):  # Show X-axis label only on the last row
            ax_disp.set_xlabel('Phase velocity (m/s)', fontsize=17)
        else:
            ax_disp.set_xticklabels([])
        ax_disp.set_ylabel('Frequency (Hz)', fontsize=17)
        ax_disp.tick_params(axis='y', labelsize=17)

    # Save and show the figure
    plt.savefig(f'figures/{dataset_name}/combined_plot_{dataset_name}_oral.pdf', format='pdf', dpi=300,bbox_inches='tight')
    #plt.show()
    plt.close()

    # Print the location of the saved figure
    print(f'The plot is available here: figures/{dataset_name}/combined_plot_{dataset_name}_oral.pdf')




#create seed:
np.random.seed(42)
# Prepare oral figures
plot_distribution_oral(labels, dataset_name)
plot_histograms_layers_oral(labels, dataset_name)
plot_histograms_thickness_oral(labels, dataset_name)
VSz_evolution_oral=VSz_oral(labels, dataset_name, [10,20,30,40,50])
VSz_evolution_oral.plot_histograms_VSz_evolution_subfigures(labels, dataset_name, [10,20,30,40,50])
plot_combined_figure_oral(shotgathers, labels, dataset_name)

#do the same for Vp:
# Prepare oral figures
plot_distribution_oral(labelsVP, dataset_name)
plot_histograms_layers_oral(labelsVP, dataset_name)
plot_histograms_thickness_oral(labelsVP, dataset_name)
VPz_evolution_oral=VSz_oral(labelsVP, dataset_name, [10,20,30,40,50],wave='Vp')
VPz_evolution_oral.plot_histograms_VSz_evolution_subfigures(labelsVP, dataset_name, [10,20,30,40,50])
plot_combined_figure_oral(shotgathers, labelsVP, dataset_name)

