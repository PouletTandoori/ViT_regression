from scipy.stats.qmc import Halton

class HaltonSequenceGenerator:
    """
    Classe pour générer une séquence de Halton multidimensionnelle.

    :param intervals: Liste de tuples [(val_min, val_max), ...] définissant les
                      intervalles de chaque dimension.
    :param scramble: Booléen indiquant si la séquence doit être mélangée (scrambling).
    """
    def __init__(self, intervals, scramble=True, integer=False):
        self.intervals = intervals
        self.n_dimensions = len(intervals)
        self.scramble = scramble
        self.halton = Halton(d=self.n_dimensions, scramble=self.scramble)
        self.integer = integer

    def generate(self, n_samples):
        """
        Génère une séquence de Halton pour les dimensions et intervalles spécifiés.

        :param n_samples: Nombre de points à générer dans la séquence.
        :return: Liste des points générés avec les valeurs mappées dans les intervalles.
        """
        # Génération de la séquence brute de Halton
        raw_samples = self.halton.random(n_samples)


        # Mise à l'échelle dans les intervalles spécifiés
        scaled_samples = []
        for i, (val_min, val_max) in enumerate(self.intervals):
            scaled_samples.append(raw_samples[:, i] * (val_max - val_min) + val_min)
        #print('scaled_samples=',scaled_samples[0].astype(int))

        if self.integer == True:
            #scaled_samples = list, chaque élément est un tableau et chaque valeur doit être entière
            scaled_samples_int = [scaled_samples[0].astype(int)]
            return list(zip(*scaled_samples_int))

        return list(zip(*scaled_samples))  # Retourne une liste de tuples (point)

# Exemple d'utilisation
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # Définir les intervalles [min, max] pour chaque dimension
    intervals = [(0, 1)] * 7  # 7 dimensions entre 0 et 1

    # Initialiser le générateur
    halton_gen = HaltonSequenceGenerator(intervals, integer=False)

    # Générer 100000 échantillons
    samples = np.array(halton_gen.generate(100000))  # (100000, 6)

    # Vérifier la distribution en traçant un histogramme pour chaque dimension
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(6):  # Pour chaque dimension
        axes[i].hist(samples[:, i], bins=20, edgecolor='blue', density=True)
        axes[i].set_title(f'Histogram for Dimension {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.show()



