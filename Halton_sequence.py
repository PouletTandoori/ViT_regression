from scipy.stats import qmc
import numpy as np

#define the number of dimensions and applay scrambling to shuffle the sequence
sampler = qmc.Halton(d=1, scramble=True)
sample = sampler.random(n=5)
#print(np.round(sample,3))

param_min, param_max = 0, 50

sample=param_min + (param_max - param_min) * sample

#print final output rounded to 3 decimal places
print(np.round(sample,3))

#print sum of the sequence
print(np.sum(sample))

def generate_halton_thicknesses(total, n):
    """
    Génère `n` épaisseurs à l'aide d'une séquence de Halton dont la somme est `total`.

    :param total: Somme totale des épaisseurs.
    :param n: Nombre d'épaisseurs à générer.
    :return: Liste des épaisseurs.
    """
    # Génère une séquence de Halton de taille n-1
    sampler = qmc.Halton(d=1, scramble=True)  # d=1 pour une dimension
    cuts = sampler.random(n - 1).flatten()  # Génère n-1 valeurs entre 0 et 1
    # Multiplie par le total pour avoir des points dans [0, total]
    cuts = np.sort(cuts * total)
    # Ajoute les limites 0 et total
    points = np.concatenate(([0], cuts, [total]))
    # Calcule les différences entre les points
    thicknesses = np.diff(points)
    return thicknesses

# Exemple d'utilisation
thicks = generate_halton_thicknesses(total=50, n=5)
print("Épaisseurs générées :", thicks)
print("Somme des épaisseurs :", np.sum(thicks))

