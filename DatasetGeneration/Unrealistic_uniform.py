import matplotlib.pyplot as plt

from ModelGenerator_modified import (Sequence, Stratigraphy,
                            Property, Lithology, ModelGenerator, Deformation)
import os
# visible devices = 2 and 3:
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from GeoDataset_modified import GeoDataset
from SeismicGenerator_modified import SeismicGenerator, plot_shotgather
from SeismicGenerator_modified import SeismicAcquisition
from GraphIO_modified import Vsdepth, ShotGather
import argparse
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
from tabulate import tabulate
import subprocess


#define parser
parser = argparse.ArgumentParser()
parser.add_argument('-s','--size_dataset',type=int,default=100,
                    help='Number of models to generate')
parser.add_argument('-data','--dataset_name',type=str,default='Uniform_dataset',
                    help='Name of the dataset')
parser.add_argument('--erase','-e',type=bool,default=True,
                    help='Erase the previous dataset')
args = parser.parse_args()


#####################################################################################
#verify if Datasets folder exists, if not create it:
if not os.path.exists('Datasets/'):
    os.mkdir('Datasets/')

if args.erase:
    # erase the previous dataset
    if os.path.exists(f'Datasets/{args.dataset_name}/'):
        os.system(f'rm Datasets/{args.dataset_name}/*/*')
        print(f'rm Datasets/{args.dataset_name}/*/*')
        print('---------------------------------------------------')
        print('\n')

#create folder for the figures if does not exist:
path_figures=f'../figures/{args.dataset_name}'
if not os.path.exists(path_figures):
    os.mkdir(path_figures)
    print('Creating folder for the figures')
    print('---------------------------------------------------')
    print('\n')


#########################################################################################
import numpy as np
from scipy.stats.qmc import Halton

def generate_stratified_models_halton(num_models=100, Vp_range=(500, 5000), Vs_range=(50, 3000), min_layers=2,
                                      max_layers=10):
    """
    Génère des modèles synthétiques stratifiés avec un gradient positif de vitesse en utilisant les séquences de Halton.

    Arguments:
    - num_models : Nombre de modèles à générer
    - Vp_range : Intervalle de Vp (min, max)
    - Vs_range : Intervalle de Vs (min, max)
    - min_layers : Nombre minimum de couches par modèle
    - max_layers : Nombre maximum de couches par modèle

    Retourne:
    - models : Liste contenant les modèles générés [(épaisseurs, Vp, Vs) pour chaque modèle]
    """

    models = []
    sampler = Halton(d=3 * max_layers, scramble=True)  # d=3 car on génère Vp et Vs et thickness

    for _ in range(num_models):
        # 1️⃣ Choisir un nombre aléatoire de couches
        num_layers = np.random.randint(min_layers, max_layers + 1)

        # 2️⃣ Générer des valeurs Halton (normalisées entre 0 et 1)
        halton_samples = sampler.random(n=num_layers)  # Générer `num_layers` valeurs


        # 3️⃣ Mapper les valeurs Halton vers les plages Vp et Vs
        Vp_values = Vp_range[0] + halton_samples[:, 0] * (Vp_range[1] - Vp_range[0])
        Vs_values = Vs_range[0] + halton_samples[:, 1] * (Vs_range[1] - Vs_range[0])

        # 4️⃣ Forcer le gradient positif en triant les valeurs
        Vp_values = np.sort(Vp_values)
        Vs_values = np.sort(Vs_values)

        # 5️⃣ Générer les densités pour chaque couche en utilisant la formule de Gardner $\rho = 0.31 * V_p^{0.25}$
        rho_values = 0.31 * Vp_values ** 0.25

        # 6️⃣ Générer des valeurs de facteur de qualité Q grâce à la loi de Brocher Q=0.1 Vp^1.5
        Q_range = (7, 500)
        Q_values = Q_range[0] + halton_samples[:, 0] * (Q_range[1] - Q_range[0])
        Q_values = np.sort(Q_values)

        # 5️⃣ Générer des épaisseurs aléatoires totalisant 50m.
        # Nous souhaitons une épaisseur minimale de 1m.
        z_toits = (1/50, 1)
        z_toits = z_toits[0] + halton_samples[:, 2] * (z_toits[1] - z_toits[0])
        thickness_values= z_toits * 50
        layer_thicknesses = np.diff(np.insert(np.cumsum(thickness_values), 0, 0))

        # 6️⃣ Stocker le modèle
        models.append((layer_thicknesses, Vp_values, Vs_values,rho_values,Q_values))


    return models


# Test : Générer 5 modèles
models = generate_stratified_models_halton(num_models=args.size_dataset)

#if == MAIN:
if __name__ == "__main__":
    # Exemple d'affichage d'un modèle
    for i, (thickness, Vp, Vs, rho, Q) in enumerate(models[:1]):  # Afficher les 2 premiers modèles
        print(f"\n🔹 Modèle {i + 1}")
        print(f"Épaisseurs: {thickness.round(2)}")
        print(f"Vp: {Vp.round(1)} m/s")
        print(f"Vs: {Vs.round(1)} m/s")
        print(f"rho: {rho.round(1)} kg/m^3")
        print(f"Q: {Q.round(1)}")

        # Calcul de la profondeur cumulée
        depth = np.cumsum(thickness)

        # Ajouter la profondeur finale (50m) à la fin
        depth = np.append(depth, 50)

        # Ajouter un zéro au début des vitesses pour commencer à la surface
        Vp = np.insert(Vp, 0, 0)
        Vs = np.insert(Vs, 0, 0)

        # Ajouter la profondeur 0 au début pour correspondre à la surface
        depth = np.insert(depth, 0, 0)

        # Inverser la profondeur pour que la surface soit en haut
        depth = depth[::-1]

        # Plot Vp vs depth using stairs (x axis=Vp, y axis=depth):
        plt.stairs(Vp, depth, label=f'Model {i + 1}')
        plt.xlabel('Depth (m)')
        plt.ylabel('Vp (m/s)')
        plt.title('Vp vs Depth')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

        # Plot Vs vs depth using stairs (x axis=Vs, y axis=depth):
        plt.stairs(Vs, depth, label=f'Model {i + 1}')
        plt.ylabel('Depth (m)')
        plt.xlabel('Vs (m/s)')
        plt.title('Vs vs Depth')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()


    # check parameters distribution over the dataset:
    # count occurences for each parameter:
    Vp_values = []
    Vs_values = []
    rho_values = []
    Q_values = []

    count_layers = 0
    thickness_values = []
    for i, (thickness, Vp, Vs, rho, Q) in enumerate(models):
        print(f"\n🔹 Modèle {i + 1}")
        Vp_values.extend(vp for vp in Vp)
        Vs_values.extend(vs for vs in Vs)
        rho_values.extend(rho for rho in rho)
        Q_values.extend(q for q in Q)
        thickness_values.extend(thickness)
        count_layers += len(thickness)
    print('---------------------------------------------------')
    print('\n')
    print('VP values:', Vp_values)
    print('---------------------------------------------------')
    print('\n')
    print('VS values:', Vs_values)
    print('---------------------------------------------------')
    print('\n')
    print('Rho values:', rho_values)
    print('---------------------------------------------------')
    print('\n')
    print('Q values:', Q_values)
    print('---------------------------------------------------')
    print('\n')
    print('Thickness values:', thickness_values)
    print('---------------------------------------------------')


    # plot histograms for each parameter:
    #hist of number of occurences for each parameter:
    plt.hist(Vp_values, bins=10, edgecolor='blue', density=False)
    plt.title(f'Histogram of Vp values')
    plt.xlabel('Vp values(m/s)')
    plt.ylabel('Occurencies')
    plt.show()
    plt.close()

    plt.hist(Vs_values, bins=10, edgecolor='blue', density=False)
    plt.title(f'Histogram of Vs values')
    plt.xlabel('Vs values(m/s)')
    plt.ylabel('Occurencies')
    plt.show()
    plt.close()

    plt.hist(rho_values, bins=10, edgecolor='blue', density=False)
    plt.title(f'Histogram of Rho values')
    plt.xlabel('Rho values(kg/m^3)')
    plt.ylabel('Occurencies')
    plt.show()
    plt.close()

    plt.hist(Q_values, bins=10, edgecolor='blue', density=False)
    plt.title(f'Histogram of Q values')
    plt.xlabel('Q values')
    plt.ylabel('Occurencies')
    plt.show()
    plt.close()

    plt.hist(thickness_values, bins=10, edgecolor='blue', density=False)
    plt.title(f'Histogram of Thickness values')
    plt.xlabel('Thickness values(m)')
    plt.ylabel('Occurencies')
    plt.show()
    plt.close()




# Questions:
# 1. que penser de la distribution des paramètres ? Tous uniformes ?
# 2. Comment améliorer le côté réaliste des sho gathers ?
# 3. Quel pourrait être le problème avec les courbes de dstribution ?




