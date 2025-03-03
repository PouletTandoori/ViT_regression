#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate seismic models and labels for training.
"""

import copy

import numpy as np
import h5py as h5
from scipy.signal.windows import gaussian
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from matplotlib import animation
from Halton_Sequence import HaltonSequenceGenerator

class ModelGenerator:
    """
    Generate a layered model.

    This class can read and write to a file the parameters needed to generate
    random models.
    """

    def __init__(self):
        # Number of grid cells in X direction.
        self.NX = 256
        # Number of grid cells in Z direction.
        self.NZ = 256
        # Grid spacing in X, Y, Z directions (in meters).
        self.dh = 10.0

        # Minimum thickness of a layer (in grid cells).
        self.layer_dh_min = 50
        # Minimum thickness of a layer (in grid cells).
        self.layer_dh_max = 1e9
        # Minimum number of layers.
        self.layer_num_min = 5
        # Fix the number of layers if not 0.
        self.num_layers = 0

        # If true, first layer dip is 0.
        self.dip_0 = True
        # Maximum dip of a layer.
        self.dip_max = 0
        # Maximum dip difference between two adjacent layers.
        self.ddip_max = 5

        # Change between two layers.
        # Add random noise two a layer (% or velocity).
        self.max_texture = 0
        # Range of the filter in x for texture creation.
        self.texture_xrange = 0
        # Range of the filter in z for texture creation.
        self.texture_zrange = 0
        # Zero-lag correlation between parameters, same for each
        self.corr = 0.6

        # Minimum fault dip.
        self.fault_dip_min = 0
        # Maximum fault dip.
        self.fault_dip_max = 0
        # Minimum fault displacement.
        self.fault_displ_min = 0
        # Maximum fault displacement.
        self.fault_displ_max = 0
        # Bounds of the fault origin location.
        self.fault_x_lim = [0, self.NX]
        self.fault_y_lim = [0, self.NZ]
        # Maximum quantity of faults.
        self.fault_nmax = 1
        # Probability of having faults.
        self.fault_prob = 0

        self.thick0min = None
        self.thick0max = None
        self.layers = None

        self._properties = None
        self._stratigraphy = None

        self.Halt_seq = None

    @property
    def properties(self):
        if self._properties is None:
            self._stratigraphy, self._properties = self.build_stratigraphy()
        return self._properties

    @property
    def stratigraphy(self):
        if self._stratigraphy is None:
            self._stratigraphy, self._properties = self.build_stratigraphy()
        return self._stratigraphy

    #setter to stratigraphy
    @stratigraphy.setter
    def stratigraphy(self, value):
        self._stratigraphy = value
        self._properties = self._stratigraphy.properties()

    def save_parameters_to_disk(self, filename):
        """
        Save all parameters to disk.

        :param filename: Name of the file for saving parameters.
        """
        with h5.File(filename, 'w') as file:
            for item in self.__dict__:
                file.create_dataset(item, data=self.__dict__[item])

    def read_parameters_from_disk(self, filename):
        """
        Read all parameters from a file.

        :param filename: Name of the file containing parameters.
        """
        with h5.File(filename, 'r') as file:
            for item in self.__dict__:
                try:
                    self.__dict__[item] = file[item][()]
                except KeyError:
                    pass

    def generate_model(self, stratigraphy=None, thicks=None, dips=None,
                       boundaries=None, gradxs=None, texture_trends=None,
                       seed=None,Halt_seq=None):
        """

        :param stratigraphy: A `Stratigraphy` object.
        :param thicks: A list of layer thicknesses. If not provided, create
                       random thicknesses. See ModelParameters for variables
                       controlling the random generation process.
        :param dips: A list of layer dips. If not provided, create
                     random dips.
        :param boundaries: A list of arrays containing the position of the top
                           of the layers. If none, generated randomly
        :param gradxs: A list of the linear trend of each property in each
                       layer. If None, no trend in x is added and if "random",
                       create random gradients in each layer.
        :param texture_trends: A list of the of array depicting the alignment
                               of the texture within a layer. If None, will
                               follow the top boundary of the layer.
        :param seed: A seed for random generators.

        :return:
            props2d: A list of 2D property arrays
            layerids: A 2D array containing layer ids
            layers: A list of Layer objects
            Halt_seq_reduced: 2D array containing Reduced halton sequence
        """
        if stratigraphy is None:
            stratigraphy = self.stratigraphy
        else:
            self.stratigraphy = stratigraphy

        if seed is not None:
            np.random.seed(seed)

        if self.Halt_seq is None:
            print('[ModelGenerator] Halton seq is None')
        else:
            print('[ModelGenerator] Halton seq shape = ',self.Halt_seq.shape)
            Halt_seq = self.Halt_seq


        if stratigraphy is None:
            stratigraphy = Stratigraphy()
        if thicks is None:
            if boundaries is None:
                #print('Halton sequence = ',Halt_seq)
                thicks,nlayer = random_thicks(self.NZ,
                                       self.layer_dh_min,
                                       self.layer_dh_max,
                                       self.layer_num_min,
                                       self.num_layers,
                                       thick0min=self.thick0min,
                                       thick0max=self.thick0max,
                                       Halton_seq=Halt_seq)
            else:
                thicks = [0 for _ in range(len(boundaries))]
        if dips is None:
            dips = random_dips(len(thicks), self.dip_max,
                               self.ddip_max, dip_0=self.dip_0)

        layers = stratigraphy.build_stratigraphy(thicks, dips,Halton_seq=Halt_seq)
        if boundaries is None:
            layers = generate_random_boundaries(self.NX, layers)
        else:
            for ii, layer in enumerate(layers):
                layer.boundary = boundaries[ii]
        if texture_trends is not None:
            for ii, layer in enumerate(layers):
                layer.texture_trend = texture_trends[ii]

        props2d, layerids = gridded_model(self.NX, self.NZ, layers,
                                          self.texture_zrange,
                                          self.texture_xrange,
                                          self.corr)
        faults = Faults(dip_min=self.fault_dip_min, dip_max=self.fault_dip_max,
                        displ_min=self.fault_displ_min,
                        displ_max=self.fault_displ_max, dh=self.dh,
                        x_lim=self.fault_x_lim, y_lim=self.fault_y_lim,
                        nmax=self.fault_nmax, prob=self.fault_prob)
        props2d, layerids = faults.add_faults(props2d, layerids)

        names = [prop.name for prop in layers[0].lithology]
        propdict = {name: prop for name, prop in zip(names, props2d)}

        # remove 1 row:
        Halt_seq_reduced = Halt_seq[1:,:]

        #créer un tuple de sorties : propdict, layerids, layers, Halt_seq_reduced
        outputs= (propdict, layerids, layers, Halt_seq_reduced)


        return outputs

    def build_stratigraphy(self):
        """
        Build the stratigraphy object that controls model creation.

        :returns:
            strati: A `Stratigraphy` object.
            properties: A dict of properties with key-values pairs `name:
                        [vmin, vmax]` where `vmin` and `vmax` are the minimum
                        and maximum values that can take a property. Each
                        property returned by generate model should be found in
                        this dictionary.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def plot_model(self, props2d, layers, animated=False, figsize=(16, 8)):
        """
        Plot the properties of a generated gridded model.

        :param props2d: The dictionary of properties from the output of
                        generate_model.
        :param layers: A list of layers from the output of generate_model.
        :param animated: If true, the plot can be animated.
        :param figsize: A tuple providing the size of the figure to create.

        :return:
            ims: A list of `pyplot` images.
            fig: A `Figure` object.
        """
        names = list(props2d.keys())
        minmax = {name: [np.inf, -np.inf] for name in names}
        for layer in layers:
            for prop in layer.lithology:
                if prop.name in minmax:
                    if minmax[prop.name][0] > prop.min:
                        minmax[prop.name][0] = prop.min
                    if minmax[prop.name][1] < prop.max:
                        minmax[prop.name][1] = prop.max
        for name in names:
            if minmax[name][0] is np.inf:
                minmax[name] = [np.min(props2d[name]) / 10,
                                np.min(props2d[name]) * 10]

        fig, axs = plt.subplots(1, len(names), figsize=figsize, squeeze=False)
        axs = axs.flatten()
        ims = [axs[ii].imshow(props2d[name], animated=animated, aspect='auto',
                              cmap='inferno', vmin=minmax[name][0],
                              vmax=minmax[name][1])
               for ii, name in enumerate(names)]

        for ii, ax in enumerate(axs):
            ax.set_title(names[ii])
            plt.colorbar(ims[ii], ax=ax, orientation="horizontal", pad=0.16,
                         fraction=0.15)
        plt.tight_layout()

        return fig, ims

    def animated_dataset(self, *args, nframes=10, path=None,**kwargs):
        """
        Produce an animation of a dataset.

        Show the input data, and the different labels for each example.

        :param phase: Which dataset, from either train, test or validate.
        """
        props2d, _, layers, _ = self.generate_model(*args, **kwargs)
        names = list(props2d.keys())
        fig, ims = self.plot_model(props2d, layers, animated=True)

        def init():
            for im, name in zip(ims, names):
                im.set_array(props2d[name])
            return ims

        def animate(t):
            props2d, _, layers,_ = self.generate_model(*args, **kwargs)
            for im, name in zip(ims, names):
                im.set_array(props2d[name])
            return ims

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=nframes, interval=3000,
                                       blit=True, repeat=True)
        if path is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=1, metadata=dict(artist='ModelGenerator'),
                            bitrate=1800)

            anim.save(f'{path}/AnimatedDataset.mp4', writer=writer)

        #plt.show()

class Stratigraphy(object):

    def __init__(self, sequences=None, defaultprops=None, Halton_seq=None):
        """
        A series of Sequences.

        When building a model the layered model will contain the sequences in
        order.

        :param sequences: A list of Sequence objects
        """
        if sequences is None:
            litho = Lithology(properties=defaultprops)
            sequences = [Sequence(lithologies=[litho])]

        self.sequences = sequences
        self.layers = None
        self.Halton_seq = Halton_seq

    def properties(self):
        """
        Summarize the properties in Stratigraphy.

        :return: A dict containing all properties contained in Stratigraphy
                 with minimum and maximum values {p.name: [min, max]}.
        """
        props = {p.name: [9999, 0]
                 for p in self.sequences[0].lithologies[0]}
        for seq in self.sequences:
            for lith in seq:
                for p in lith.properties:
                    if props[p.name][0] > p.min:
                        props[p.name][0] = p.min
                    if props[p.name][1] < p.max:
                        props[p.name][1] = p.max
        return props

    def build_stratigraphy(self, thicks, dips, Halton_seq):

        # verify if Halton sequence is defined or not:
        if Halton_seq is None or Halton_seq.shape[1] < 2:
            Halton_seq=self.Halton_seq


        #print(f"🔍 Halton_seq shape: {Halton_seq.shape}")

        layers = []
        seqid = 0
        seqid0 = 0
        seqthick = 0

        # Sélection des séquences lithologiques valides
        sequences = [s for s in self.sequences if s.skipprob < np.random.rand()]
        seqiter = iter(sequences[0])

        # Définition des épaisseurs minimales et maximales des séquences

        sthicks_min = [s.thick_min for s in sequences]
        sthicks_max = [s.thick_max for s in sequences]
        sthicks_min[-1] = sthicks_max[-1] = 1e09

        seq = sequences[0]
        lith = None
        seq_nlay = 0

        # on prend seulement les lignes qui correspondent aux nombre de lithologies:
        halton_ordered = Halton_seq[:len(thicks)]
        # Pour encourager un gradient de vitesse, on va mettre dans l'ordre croissant les valeurs de Halton: (colonnes triées du haut vers le bas)
        # à partir de la 3ème colonne et pour autant de ligne que de couches:
        for ii in range(halton_ordered.shape[0]):
            halton_ordered=np.sort(halton_ordered, axis=0)


        prof=0
        # Boucle sur les couches
        for ii, (t, di) in enumerate(zip(thicks, dips)):
            seqthick0 = seqthick
            seqthick += t

            # Changement de séquence si nécessaire
            if seq_nlay >= seq.nmin and (seqthick0 > sthicks_min[seqid] or seqthick >= sthicks_max[seqid]):
                seq_nlay = 0
                if seqid < len(sequences) - 1:
                    seqid0 = seqid
                    seqid += 1
                    seqiter = iter(sequences[seqid])
                    seq = sequences[seqid]

            seq_nlay += 1
            lith0 = lith
            lith = next(seqiter)

            #profondeur de la iieme couche:
            prof+=thicks[ii]
            #create a coef to multiply the Halton sequence value by prof, between 1 and 1.2 depending on the prof value (between 0 and 100):
            coef=1+(prof/100)*0.3

            # Utilisation de la séquence de Halton pour les propriétés
            properties = []
            halton_values = halton_ordered[ii, 2:]  # On prend les valeurs à partir de la 3ème colonne


            #print('halton_values shape=',halton_values.shape)

            # pour une ligne de Halton, on a 4 valeurs:(Vp, Vs, Rho, Q)
            #parcourir les lignes et ensuite parcourir les colonnes:
            jj=0
            print(f'profondeur de la {ii}ème couche: {prof}, donc coef={coef}')
            for prop in lith:
                #give the name of the property:
                if prop.name=='vp':
                    #random value:
                    val=prop.min + halton_values[jj] * (prop.max - prop.min)*coef


                    vp=val
                    jj+=1
                elif prop.name=='vs':
                    #random value:
                    val=prop.min + halton_values[jj] * (prop.max - prop.min)*coef
                    vs=val
                    jj+=1
                elif prop.name=='vpvs':
                    #dot not augment jj:

                    val=vp / vs

                else:
                    #random value:
                    val=prop.min + halton_values[jj] * (prop.max - prop.min)
                    jj+=1

                properties.append(val)

        # On va essayer d'intégrer un gradient de vitesses avec la profondeur:
            # repérer le début de la couche par rapport au modèle:
            #print(f"🔍 ii: {ii}, seqthick: {seqthick}, seqthick0: {seqthick0}")
            #


            # Création et stockage de la couche
            layers.append(Layer(ii, t, di, seq, lith, gradx=None, properties=properties))

        self.layers = layers
        return layers

    def summary(self):
        x = PrettyTable()
        x.add_column("Layer no", [la.idnum for la in self.layers])
        x.add_column("Sequence",  [la.sequence.name for la in self.layers])
        x.add_column("Lithology",  [la.lithology.name for la in self.layers])
        x.add_column("Thickness",  [la.thick for la in self.layers])
        x.add_column("Dip",  [la.dip for la in self.layers])
        for ii in range(len(self.layers[0].lithology.properties)):
            x.add_column(self.layers[0].lithology.properties[ii].name,
                         [np.round(la.properties[ii],2) for la in self.layers])
        print(x)


class Sequence(object):

    def __init__(self, name="Default", lithologies=None, ordered=False,
                 proportions=None, thick_min=0, thick_max=1e9, nmax=9999,
                 nmin=1, deform=None, skip_prob=0, accept_decrease=1):
        """
        A sequence of Lithology objects.

        It can be ordered or random, meaning that when iterated upon, the
        Sequence object will provide either the given lithologies in order, or
        provide a random draw of the lithologies, with a probability of a
        lithology to be drawn given by proportions.

        :param name: Name of the sequence.
        :param lithologies: A list of Lithology objects.
        :param ordered: If True, the Sequence provides the lithology in the
                        order within lithologies. If False, the Sequence
                        provides random lithologies drawn from the list.
        :param proportions: A list of proportions of each lithology in the
                            sequence. Must sum to 1.
        :param thick_min: The minimum thickness of the sequence.
        :param thick_max: The maximum thickness of the sequence.
        :param nmax: Maximum number of lithologies that can be drawn from a
                     sequence, when ordered is False.
        :param nmin: Minimum number of lithologies that can be drawn from a
                     sequence, when ordered is False.
        :param deform: A Deformation object that generate random deformation of
                       a boundary.
        :param skip_prob: The probability that this sequence is skipped.
        :param accept_decrease: The probability to accept a decrease of a
                                property, for properties with
                                filter_decrease=True.
        """
        self.name = name
        if lithologies is None:
            lithologies = [Lithology()]
        self.lithologies = lithologies
        if proportions is not None:
            if np.sum(proportions) - 1.0 > 1e-6:
                raise ValueError("The sum of proportions should be 1")
            if len(proportions) != len(lithologies):
                raise ValueError("Lengths of proportions and lithologies "
                                 "should be equal")
        self.proportions = proportions
        self.ordered = ordered
        if ordered:
            nmax = len(lithologies)
        self.nmax = nmax
        self.nmin = nmin
        self.thick_max = thick_max
        self.thick_min = thick_min
        self.deform = deform
        self.skipprob = skip_prob
        self.accept_decrease = accept_decrease

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.nmax:
            if self.ordered:
                out = self.lithologies[self.n]
            else:
                out = np.random.choice(self.lithologies, p=self.proportions)
            self.n += 1
            return out
        else:
            raise StopIteration


class Lithology(object):

    def __init__(self, name="Default", properties=None):
        """
        A collection of Property objects.

        For example, a Lithology could be made up of vp and rho for an acoustic
        media and describe unconsolidated sediments seismic properties for a
        marine survey.

        :param name: Name of a lithology
        :param properties: A list of Property objects
        """
        self.name = name
        if properties is None:
            properties = [Property()]
        self.properties = properties
        for prop in properties:
            setattr(self, prop.name, prop)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.properties):
            self.n += 1
            return self.properties[self.n-1]
        else:
            raise StopIteration


class Layer(object):

    def __init__(self, idnum, thick, dip, sequence, lithology, properties,
                 boundary=None, gradx=None, texture_trend=None):
        """
        A specific layer within a model.

        Provide a description of its lithology, its thickness, dip and the
        deformation of its upper boundary.

        :param idnum: The identification number of the layer.
        :param thick: The thickness of the layer.
        :param dip: The dip of the layer.
        :param sequence: A Sequence object to which the layer belongs.
        :param lithology: A Lithology object to which the layer belongs.
        :param properties: A list of value of the layer properties.
        :param boundary: An array of the position of the top of the layer.
        :param gradx: A list of horizontal increase of each property.
        """
        self.idnum = idnum
        self.thick = int(thick)
        self.dip = dip
        self.sequence = sequence
        self.lithology = lithology
        self.properties = properties
        names = [prop.name for prop in lithology]
        for name, prop in zip(names, self.properties):
            setattr(self, name, prop)
        if gradx is None:
            gradx = [0 for _ in self.properties]
        self.gradx = gradx
        self.boundary = boundary
        self.texture_trend = texture_trend


class Deformation:

    def __init__(self, max_deform_freq=0, min_deform_freq=0, amp_max=0,
                 max_deform_nfreq=20, prob_deform_change=0.3,
                 cumulative=False):
        """
        Create random deformations of a boundary with harmonic functions.

        :param max_deform_freq: Maximum frequency of the harmonic components.
        :param min_deform_freq: Minimum frequency of the harmonic components.
        :param amp_max: Maximum amplitude of the deformation.
        :param max_deform_nfreq: Number of frequencies.
        :param cumulative: If True, deformation of consecutive layers
                           are added together (are correlated).
        """
        self.max_deform_freq = max_deform_freq
        self.min_deform_freq = min_deform_freq
        self.amp_max = amp_max
        self.max_deform_nfreq = max_deform_nfreq
        self.prob_deform_change = prob_deform_change
        self.cumulative = cumulative

    def create_deformation(self, nx):
        """
        Create random deformations of a boundary with harmonic functions.

        :param nx: Number of points of the boundary.

        :return: An array containing the deformation function.
        """
        x = np.arange(0, nx)
        deform = np.zeros(nx)
        if self.amp_max > 0 and self.max_deform_freq > 0:
            nfreqs = np.random.randint(self.max_deform_nfreq)
            vmin = np.log(self.max_deform_freq)
            vmax = np.log(self.min_deform_freq)
            freqs = np.exp(np.random.uniform(vmin, vmax, size=nfreqs))
            phases = np.random.rand(nfreqs) * np.pi * 2
            amps = np.random.rand(nfreqs)
            for ii in range(nfreqs):
                deform += amps[ii] * np.sin(freqs[ii] * x + phases[ii])

            ddeform = np.max(np.abs(deform))
            if ddeform > 0:
                deform = deform / ddeform * self.amp_max * np.random.rand()

        return deform


class Faults:

    def __init__(self, dip_min=0, dip_max=0, displ_min=0, displ_max=0, dh=10.0,
                 x_lim=None, y_lim=None, nmax=1, prob=0):
        """
        Create random faults in a 2D gridded model.

        :param dip_min: Minimum dip, as measured in degrees from the
                        horizontal axis.
        :param dip_max: Maximum dip, as measured in degrees from the
                        horizontal axis.
        :param displ_min: Minimum absolute displacement, in meters. A positive
                          displacement brings the top layer upwards.
        :param displ_max: Maximum absolute displacement, in meters. A positive
                          displacement brings the top layer upwards.
        :param dh: Grid cell size, in meters.
        :param x_lim: Bounds `[x_min, x_max]` of the fault origin location.
                      Defaults to the model's boundaries.
        :param y_lim: Bounds `[y_min, y_max]` of the fault origin location.
                      Defaults to the model's boundaries. `y` is measured from
                      the surface.
        :param nmax: Maximum quantity of faults.
        :param prob: Either the scalar probability of having at least one fault
                     or a list of probabilities for each quantity of faults
                     in `range(1, nmax+1)`. In either case, the remaining
                     probability is associated to not having any fault.
        """
        self.dip_min = np.deg2rad(dip_min)
        self.dip_max = np.deg2rad(dip_max)
        self.displ_min = displ_min
        self.displ_max = displ_max
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.dh = dh
        self.nmax = nmax
        if isinstance(prob, list):
            assert len(prob) == nmax
            prob = [1-sum(prob), *prob]
        else:
            prob = [1 - prob, *[prob/nmax]*nmax]
        self.prob = prob

    def add_faults(self, props2d, layerids):
        n = np.random.choice(self.nmax+1, p=self.prob)
        for _ in range(n):
            props2d, layerids = self.add_fault(props2d, layerids)
        return props2d, layerids

    def add_fault(self, props2d, layerids):
        dip = np.random.uniform(self.dip_min, self.dip_max)
        dip *= np.random.choice([-1, 1])
        displ = np.random.uniform(self.displ_min, self.displ_max)
        displ /= self.dh
        vdispl = displ * np.sin(abs(dip))
        vdispl = int(round(vdispl))
        if vdispl == 0:
            return props2d, layerids

        x_min, x_max = self.x_lim or (0, layerids.shape[1])
        y_min, y_max = self.y_lim or (0, layerids.shape[0])
        if y_min == y_max:
            y_max += 1
        if x_min == x_max:
            x_max += 1
        y = layerids.shape[0] - np.random.randint(y_min, y_max)
        x = np.random.randint(x_min, x_max)

        grid_idx = np.meshgrid(*(np.arange(s) for s in layerids.shape[::-1]))
        grid_idx = np.array(grid_idx).reshape([2, -1]).T
        grid_idx -= [x, y]
        line_idx = np.cos(dip), np.sin(dip)
        is_over = np.cross(line_idx, grid_idx) < 0
        is_over = is_over.reshape(layerids.shape)

        arrays = [*props2d, layerids]
        for i, arr in enumerate(arrays):
            if vdispl > 0:
                displ_arr = np.pad(arr, ((0, vdispl), (0, 0)), mode='edge')
                displ_arr = displ_arr[vdispl:]
            else:
                displ_arr = np.pad(arr, ((-vdispl, 0), (0, 0)), mode='edge')
                displ_arr = displ_arr[:vdispl]

            upper, lower = displ_arr, arr
            arrays[i] = np.where(is_over, upper, lower)

        *props2d, layerids = arrays
        return props2d, layerids


class Diapir(Lithology):

    def __init__(self, *args, loc_min=None, loc_max=None, height_min=0,
                 height_max=1, width_min=0, width_max=1, prob=1, **kwargs):
        """
        Add a diapir-shaped deformation to layer boundaries.

        :param height_min: Minimum height, in grid cell units.
        :param height_max: Maximum height, in grid cell units.
        :param width_min: Minimum width at half-maximum, in grid cell units.
        :param width_max: Maximum width at half-maximum, in grid cell units.
        :param prob: Probability of adding a diapir to a layer.
        """
        super().__init__(*args, **kwargs)
        self.loc_min = loc_min
        self.loc_max = loc_max
        self.height_min = height_min
        self.height_max = height_max
        self.width_min = width_min
        self.width_max = width_max
        self.prob = prob

    def add_diapir(self, layer):
        if np.random.random_sample() > self.prob:
            return
        nx = len(layer.boundary)
        if self.width_min != self.width_max:
            width = np.random.randint(self.width_min, self.width_max)
        else:
            width = self.width_min
        if self.height_min != self.height_max:
            height = np.random.randint(self.height_min, self.height_max)
        else:
            height = self.height_min
        loc_max = nx - width - 1
        if self.loc_max is not None:
            assert self.loc_max <= loc_max
            loc_max = self.loc_max
        loc_min = width
        if self.loc_min is not None:
            assert self.loc_min >= loc_min
            loc_min = self.loc_min
        loc = np.random.randint(loc_min, loc_max+1)
        x_start, x_end = loc - width, loc + width

        k = 10**np.random.uniform(.5, 1.5)
        x = np.linspace(0, 2*np.pi, 2*width+1)
        diapir = (1-np.arctan(k*np.cos(x))/np.arctan(k)) / 2
        diapir *= height
        layer.boundary[x_start:x_end+1] -= diapir.astype(int)
        layer.boundary = np.clip(layer.boundary, 0, None)


class Property(object):
    def __init__(self, name="Default", vmin=1000, vmax=5000, texture=0,
                 trend_min=0, trend_max=0, gradx_min=0, gradx_max=0,
                 dzmin=None, dzmax=None, filter_decrease=False):
        """
        Describe one material property of a Lithology object.

        Provides the maximum and minimum value that the property can take
        within a Lithology. For example, a Property could describe the
        P-wave velocity.

        :param name: Name of the property.
        :param vmin: Minimum value of the property.
        :param vmax: Maximum value of the property.
        :param texture: Maximum percentage of change of random fluctuations
                        within a layer of the property.
        :param trend_min: Minimum value of the linear trend in z in a layer.
        :param trend_max: Maximum value of the linear trend in z in a layer.
        :param gradx_min: Minimum value of the linear trend in x in a layer.
        :param gradx_max: Maximum value of the linear trend in x in a layer.
        :param dzmax: Maximum change between two consecutive layers with
                      the same lithology.
        :param dzmin: Minimum change between two consecutive layers with
                      the same lithology. Defaults to `-dzmax`.
        :param filter_decrease: If true, accept a decrease of this property
                                according to the Sequence's accept_decrease.
        """
        self.name = name
        self.min = vmin
        self.max = vmax
        self.texture = texture
        self.trend_min = trend_min
        self.trend_max = trend_max
        self.gradx_min = gradx_min
        self.gradx_max = gradx_max
        self.dzmin = dzmin
        self.dzmax = dzmax
        self.filter_decrease = filter_decrease


import numpy as np

import numpy as np

def random_thicks(nz, thickmin, thickmax, nmin, nlayer,
                  thick0min=None, thick0max=None, Halton_seq=None):
    """
    Génère une séquence de couches avec différentes épaisseurs en assurant une épaisseur minimale.

    :param nz: Nombre de points en Z de la grille.
    :param thickmin: Épaisseur minimale d'une couche.
    :param thickmax: Épaisseur maximale du modèle.
    :param nmin: Nombre minimum de couches.
    :param nlayer: Nombre de couches à créer.
    :param thick0min: Si défini, la première épaisseur est tirée entre thick0min et thick0max.
    :param thick0max: Voir `thick0min`.
    :param Halton_seq: Séquence de Halton pour la génération pseudo-aléatoire.

    :return: Liste des épaisseurs des couches et le nombre de couches.
    """
    dh=0.5

    if Halton_seq is None or Halton_seq.shape[1] < 2:
        print("⚠️ Warning: Halton sequence non fournie. Génération d'une séquence aléatoire.")
        Halton_seq = np.array(HaltonSequenceGenerator(intervals=[(0, 1)]*6).generate(100000))

        print(f"🔍 Halton_seq shape: {Halton_seq.shape}")

    thickmax = min(int(nz / nmin), thickmax)
    if thickmax < thickmin:
        print("⚠️ Warning: épaisseur max < épaisseur min. Correction automatique.")

    # Déterminer le nombre de couches
    nlmin = nmin #=1
    halton_nlayer = Halton_seq[0, 0]
    #halton entre 0 et 1. On va faire en sorte que si halton_nlayer<0.25, nlayer=1, si 0.25<halton_nlayer<0.5, nlayer=2, si 0.5<halton_nlayer<0.75, nlayer=3, si 0.75<halton_nlayer<1, nlayer=4:
    if halton_nlayer <= 0.25:
        nlayer = 1
    elif halton_nlayer > 0.25 and halton_nlayer <= 0.5:
        nlayer = 2
    elif halton_nlayer > 0.5 and halton_nlayer <= 0.75:
        nlayer = 3
    elif halton_nlayer > 0.75:
        nlayer = 4

    #nlayer = np.clip(nlayer, nlmin, nlmax)
    print('Halton Nlayer = ', halton_nlayer, ' et nlayer = ', nlayer)

    #print(f'📏 Nombre de couches: {nlayer}, épaisseur max du modèle: {thickmax}')

    if nlayer > 1:
        halton_toits = Halton_seq[:nlayer-1, 1]  # On prend `nlayer-1` valeurs
        toits = thickmin + halton_toits * (thickmax - thickmin)/dh
        toits = np.round(toits).astype(int)

        # Ajouter 0 et thickmax
        toits = np.concatenate(([0], toits, [thickmax/dh]))
        thicks = np.diff(toits)

        # Assurer que chaque épaisseur est au moins `thickmin`
        while np.any(thicks < thickmin):
            for i in range(len(thicks)):
                if thicks[i] < thickmin:
                    # Augmenter cette épaisseur et ajuster les autres
                    delta = thickmin - thicks[i]
                    thicks[i] = thickmin
                    idx_adj = np.random.choice(np.where(thicks > thickmin)[0], 1)  # Trouver une couche ajustable
                    thicks[idx_adj] -= delta  # Réduire une autre épaisseur pour compenser
    else:
        thicks = np.array([thickmax/dh])

    # Ajuster la première épaisseur si `thick0min` et `thick0max` sont définis
    if thick0min is not None and thick0max is not None:
        thicks[0] = np.random.uniform(thick0min, thick0max)

    #print(f'📊 Épaisseurs des couches: {thicks}')

    return thicks, nlayer





def random_dips(n_dips, dip_max, ddip_max, dip_0=True):
    """
    Generate a random sequence of dips of layers.

    :param n_dips: Number of dips to generate.
    :param dip_max: Maximum dip.
    :param ddip_max: Maximum change of dip.
    :param dip_0: If true, the first dip is 0.

    :return: A list containing the dips of the layers.
    """
    dips = np.zeros(n_dips)
    if not dip_0:
        dips[1] = np.random.uniform(-dip_max, dip_max)
    for ii in range(2, n_dips):
        dips[ii] = dips[ii - 1] + np.random.uniform(-ddip_max, ddip_max)
        if np.abs(dips[ii]) > dip_max:
            dips[ii] = np.sign(dips[ii]) * dip_max

    return dips


def generate_random_boundaries(nx, layers):
    """
    Randomly generate a boundary for each layer.

    Generation is based on the thickness, dip and deformation properties of the
    sequence to which a layer belong.

    :return: The list of layers with the boundary property filled randomly.
    """
    top = layers[0].thick
    seq = layers[0].sequence
    de = np.zeros(nx, dtype=int)
    for layer in layers[1:]:
        if layer.boundary is None:
            boundary = top
            theta = layer.dip / 360 * 2 * np.pi
            boundary += np.array([int(np.tan(theta) * (jj - nx / 2))
                                  for jj in range(nx)], dtype=int)
            if layer.sequence != seq:
                prob = 1
                seq = layer.sequence
            else:
                prob = np.random.rand()
            if seq.deform is not None and prob < seq.deform.prob_deform_change:
                if seq.deform.cumulative:
                    de += seq.deform.create_deformation(nx).astype(int)
                else:
                    de = seq.deform.create_deformation(nx)
            boundary += de.astype(int)
            boundary = np.clip(boundary, 0, None)
            layer.boundary = boundary
            top += layer.thick

    return layers


def gridded_model(nx, nz, layers, lz, lx, corr):
    """
    Generate a gridded model from a model depicted by a list of Layers objects.

    Add a texture in each layer.

    :param nx: Grid size in X.
    :param nz: Grid size in Z.
    :param layers: A list of Layer objects.
    :param lz: The coherence length in z of the random heterogeneities.
    :param lx: The coherence length in x of the random heterogeneities.
    :param corr: Zero-lag correlation between each property.

    :return:
        props2d: A list of 2D grid of the properties.
        layerids: A grid of layer id numbers.
    """
    # Generate the 2D model, from top thicks to bottom
    npar = len(layers[0].properties)
    props2d = [np.full([nz, nx], p) for p in layers[0].properties]
    layerids = np.zeros([nz, nx])

    addtext = False
    addtrend = False
    for layer in layers:
        for prop in layer.lithology.properties:
            if lx > 0 and lz > 0 and prop.texture > 0:
                addtext = True
            if np.abs(prop.trend_min) > 1e-6 or np.abs(prop.trend_max) > 1e-6:
                addtrend = True

    if addtext:
        textures = random_fields(npar, 2 * nz, 2 * nx, lz=lz, lx=lx, corr=corr)
        for n in range(npar):
            textamp = layers[0].lithology.properties[n].texture
            if textamp > 0:
                textures[n] = textures[n] / np.max(textures[n])
                props2d[n] += textures[n][:nz, :nx] * textamp

    for layer in layers[1:]:
        trends = [None for _ in range(npar)]
        if addtrend is not None:
            for n in range(npar):
                tmin = layer.lithology.properties[n].trend_min
                tmax = layer.lithology.properties[n].trend_max
                trends[n] = np.random.uniform(tmin, tmax) #### modified

        top = np.max(layer.boundary)
        if layer.texture_trend is not None:
            texture_trend = -layer.texture_trend
            texture_trend -= np.min(texture_trend)
            if top + int(np.max(texture_trend)) + nz > 2 * nz:
                top = 2 * nz - int(np.max(texture_trend)) - nz
        else:
            texture_trend = None

        if isinstance(layer.lithology, Diapir):
            layer.lithology.add_diapir(layer)

        for jj, z in enumerate(layer.boundary):
            for n in range(npar):
                prop = layer.properties[n]
                grad = layer.gradx[n]
                props2d[n][z:, jj] = prop + grad * jj
            layerids[z:, jj] = layer.idnum
            if addtext:
                if layer.texture_trend is not None:
                    b1 = top + z + int(texture_trend[jj])
                else:
                    b1 = top
                b2 = b1 + nz - z
                for n in range(npar):
                    textamp = layer.lithology.properties[n].texture
                    if textamp > 0:
                        props2d[n][z:, jj] += textures[n][b1:b2, jj] * textamp
            if addtrend is not None:
                for n in range(npar):
                    props2d[n][z:, jj] += (trends[n] * np.arange(z, nz))

    # for n in range(npar):
    #     vmin = layers[0].lithology.properties[n].min
    #     vmax = layers[0].lithology.properties[n].max
    #     props2d[n][props2d[n] < vmin] = vmin
    #     props2d[n][props2d[n] > vmax] = vmax

    return props2d, layerids


def random_fields(nf, nz, nx, lz=2, lx=2, corr=None):
    """
    Create a random model with bandwidth limited noise.

    :param nf: Number of fields to generate.
    :param nz: Number of cells in z.
    :param nx: Number of cells in x.
    :param lz: High frequency cut-off size in z.
    :param lx: High frequency cut-off size in x.
    :param corr: Zero-lag correlation between 1 field and subsequent fields.
    """
    corrs = [1.0] + [corr for _ in range(nf-1)]

    noise0 = np.random.random([nz, nx])
    noise0 = noise0 - np.mean(noise0)
    noises = []
    for ii in range(nf):
        noisei = np.random.random([nz, nx])
        noisei = noisei - np.mean(noisei)
        noise = corrs[ii] * noise0 + (1.0-corrs[ii]) * noisei
        noise = np.fft.fft2(noise)
        noise[0, :] = 0
        noise[:, 0] = 0

        maskz = gaussian(nz, lz)**2
        maskz = np.roll(maskz, [int(nz / 2), 0])
        if lx > 0:
            maskx = gaussian(nx, lx)**2
            maskx = np.roll(maskx, [int(nx / 2), 0])
            noise *= maskx
        noise = noise * np.reshape(maskz, [-1, 1])

        noise = np.real(np.fft.ifft2(noise))
        noise = noise / np.max(noise)
        if lx == 0:
            noise = np.stack([noise[:, 0] for _ in range(nx)], 1)

        noises.append(noise)

    return noises


if __name__ == '__main__':
    gen = ModelGenerator()
    stratigraphy = Stratigraphy()
    gen.animated_dataset(stratigraphy)