"""
Define the base class for building a dataset.

The `GeoDataset` class is the main interface to define and build a Geophysical
dataset for deep neural network training.

See `DefinedDataset` for examples on how to use this class.
"""

import os
import gc
import fnmatch
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from natsort import natsorted as sorted
from Halton_Sequence import HaltonSequenceGenerator


from DatasetGenerator_modified import DatasetGenerator



class GeoDataset:
    """
    Base class of a dataset.

    Define a specific dataset by inheriting from this class and changing the
    model parameters.
    """
    basepath = os.path.abspath("Datasets")

    # Seed of the 1st model generated. Seeds for subsequent models are
    # incremented by 1.
    seed0 = 0

    def __init__(self, trainsize=1000, validatesize=0, testsize=100,
                 toinputs=None, tooutputs=None):
        """
        Initialize a GeoDataset.

        :param trainsize: Number of examples in the training dataset.
        :param validatesize: Number of examples in the validation dataset.
        :param testsize: Number of examples in the test dataset.
        :param toinputs: Dict containing name: name of graphio listing all
                         desired inputs.
        :param tooutputs: Dict containing name: name of graphio listing all
                         desired outputs.

        """
        self.trainsize = trainsize
        self.validatesize = validatesize
        self.testsize = testsize
        self.toinputs = toinputs
        self.tooutputs = tooutputs
        print('[GeoDataset] Outputs:',self.tooutputs)

        (self.model, self.physic, self.graphios) = self.set_dataset()
        self.generator = DatasetGenerator(model=self.model,
                                          physic=self.physic,
                                          graphios=self.graphios,
                                          Halton_seq=None)

        # Paths of the test, train and validation dataset.
        self.datatrain = os.path.join(self.basepath, self.name, "train")
        self.datavalidate = os.path.join(self.basepath, self.name, "validate")
        self.datatest = os.path.join(self.basepath, self.name, "test")

        # List of examples found in the dataset paths.
        self.files = {"train": [], "validate": [], "test": []}
        self.shuffled = None
        self._shapes = None

    @property
    def name(self):
        if hasattr(self, "__name"):
            return self.__name
        else:
            return type(self).__name__

    @name.setter
    def name(self, name):
        self.__name = name

    def set_dataset(self):
        """
        Define the parameters of a dataset.

        Override this method to set the parameters of a dataset.

        :return:
            model: A `ModelGenerator` object that generates models.
            physic: An `Physic` object that set data creation.
            graphios: The parallel to `inputs` for graph graphios.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")

    def _getfilelist(self, phase=None):
        """
        Search for examples found in the dataset directory.
        """
        phases = {"train": self.datatrain,
                  "validate": self.datavalidate,
                  "test": self.datatest}
        if phase is not None:
            phases = {phase: phases[phase]}

        for el in phases:
            try:
                files = fnmatch.filter(os.listdir(phases[el]), 'example_*')
                files = sorted(files)
                self.files[el] = [os.path.join(phases[el], f) for f in files]
            except FileNotFoundError:
                pass

    def generate_dataset(self, nproc=1):
        """
        Generate the training, testing and validation datasets with GPUs.
        """

        #generate Halton sequence:
        intervals = [(0, 1)] * 6  # 6 dimensions between 0 and 1
        Halton_sequence=HaltonSequenceGenerator(intervals, integer=False)
        # Generate 100 times more samples than the number of examples:
        n_samples=25*(self.trainsize+self.validatesize+self.testsize)
        print('Generating Halton sequence: ',n_samples,' samples')
        Halton_seq=np.array(Halton_sequence.generate(n_samples))

        print('shape Halton sequence before generation:',Halton_seq.shape)
        print('Halton sequence:',Halton_seq[:20])

        print('#############################################')
        print('Generating training dataset....')
        seed0 = self.seed0
        self.generator.Halt_seq=Halton_seq
        samples=self.generator.generate_dataset(self.datatrain, self.trainsize,
                                        seed0=seed0, nproc=nproc)

        print('Halton seq at the end of training dataset generation:',samples.shape)
        print('#############################################')

        print('Generating validation dataset....')
        seed0 += self.trainsize
        self.generator.Halt_seq=samples
        samples2=self.generator.generate_dataset(self.datavalidate, self.validatesize,
                                        seed0=seed0, nproc=nproc)

        print('Halton seq at the end of validation dataset generation:',samples2.shape)
        print('#############################################')

        print('Generating testing dataset....')
        seed0 += self.validatesize
        self.generator.Halt_seq=samples2
        samples3=self.generator.generate_dataset(self.datatest, self.testsize,
                                        seed0=seed0, nproc=nproc)

        print('Halton seq at the end of testing dataset generation:',samples3.shape)

    def get_example(self, filename=None,  phase="train", shuffle=True, preprocess=False):
        """
        Read an example from a file and apply preprocessing if desired.


        :param filename: If provided, get the example in filename. If None, get
                         a random example for a file list provided by phase.
        :param phase: Either "train", "test" or "validate". Get an example from
                      the "phase" dataset.
        :param shuffle: If True, draws randomly an example, else give examples
                        in order.
        :param preprocess: If True, apply preprocessing to the data.

        :return:
            inputs: A dictionary of inputs' name-values pairs.
            labels: A dictionary of labels' name-values pairs.
            weights: A dictionary of weights' name-values pairs.
            filename: The filename of the example.
        """

        self.phase = phase
        if filename is None:
            if not self.files[phase] or self.shuffled != shuffle:
                self._getfilelist(phase=phase)
                print('files:',self.files[phase])
                # if .lock in the name of the file, remove it:
                self.files[phase] = [f for f in self.files[phase] if '.lock' not in f]
                if not self.files[phase]:
                    raise FileNotFoundError
                if shuffle:
                    np.random.shuffle(self.files[phase])
                self.shuffled = shuffle
            filename = self.files[phase].pop()

        toread = list(set(list(self.toinputs.values())
                          + list(self.tooutputs.values())))
        features, weights = self.generator.read(filename, toread)
        if preprocess:
            for key in features:
                processed = self.graphios[key].preprocess(features[key],
                                                          weights[key])
                features[key] = processed[0]
                weights[key] = processed[1]

        inputs = {name: features[key] for name, key in self.toinputs.items()}
        labels = {name: features[key] for name, key in self.tooutputs.items()}
        weights = {name: weights[key] for name, key in self.tooutputs.items()}

        return inputs, labels, weights, filename

    def plot_example(self, filename=None, phase='train', plot_preds=False,
                     apply_weights=True, preprocess=False, pred_dir=None, ims=None):
        """
        Plot the data and the labels of an example.

        :param filename: If provided, get the example in filename. If None, get
                         a random example for a file list provided by phase.
        :param phase: Either "train", "test" or "validate". Get an example from
                      the "phase" dataset.
        :param plot_preds: Whether or not to plot predictions.
        :param apply_weights: Whether to feed the weights to all `plot`
                              functions the images or to show the weights on
                              another row.
        :param preprocess: If True, apply preprocessing to the data.
        :param pred_dir: The name of the subdirectory within the dataset test
                         directory to restore the predictions from. Defaults to
                         the name of the network class. This should typically
                         be the network's name.
        :param ims: List of return values of plt.imshow to update.
        """

        (inputs, labels,
         weights, filename) = self.get_example(filename=filename, phase=phase,
                                               preprocess=preprocess)

        rows = [inputs, labels]
        inputs_meta = {name: deepcopy(self.graphios[key]) for name, key in self.toinputs.items()}
        outputs_meta = {name: deepcopy(self.graphios[key]) for name, key in self.tooutputs.items()}
        for el in inputs_meta.values():
            el.meta_name = "Inputs"
        for el in outputs_meta.values():
            el.meta_name = "Outputs"
        rows_meta = [inputs_meta]
        if outputs_meta:
            rows_meta.append(outputs_meta)
        if not apply_weights:
            rows.append(weights)
            weights_meta = deepcopy(outputs_meta)
            for output in weights_meta.values():
                output.meta_name = "Weights"
            rows_meta.append(weights_meta)

        if plot_preds:
            prednames = list(self.tooutputs.keys())
            preds = self.generator.read_predictions(filename, pred_dir,
                                                    prednames)
            rows.append(preds)
            preds_meta = deepcopy(outputs_meta)
            for output in preds_meta.values():
                output.meta_name = "Predictions"
            rows_meta.append(preds_meta)

        nrows = len(rows)
        ims_per_row = [sum(row[name].naxes for name in row)
                       for row in rows_meta]
        qty_ims = sum(ims_per_row)
        ncols = np.lcm.reduce(ims_per_row)

        if ims is None:
            fig = plt.figure(figsize=[16, 8], constrained_layout=False)
            gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
            axs = []
            for i, (row, ims_in_row) in enumerate(zip(rows, ims_per_row)):
                ncols_per_im = ncols // ims_in_row
                for j_min in range(0, ncols, ncols_per_im):
                    ax = fig.add_subplot(gs[i, j_min:j_min+ncols_per_im])
                    axs.append(ax)
            ims = [None for _ in range(qty_ims)]
        else:
            fig = None
            axs = [None for _ in range(qty_ims)]

        n = 0
        for row, row_meta in zip(rows, rows_meta):
            for colname in row:
                print('colname:', colname)

                naxes = row_meta[colname].naxes
                input_ims = ims[n:n+naxes]
                input_axs = axs[n:n+naxes]
                try:
                    colweights = weights[colname] if apply_weights else None
                except KeyError:
                    colweights = None
                data = row[colname]
                try:
                    data = row_meta[colname].postprocess(data)
                except AttributeError:
                    pass
                output_ims = row_meta[colname].plot(data,
                                                    weights=colweights,
                                                    axs=input_axs,
                                                    ims=input_ims)
                for im in output_ims:
                    ims[n] = im
                    n += 1

        return fig, axs, ims

    def animate(self, phase='train', plot_preds=False, apply_weights=True,
                preprocess=True, pred_dir=None):
        """
        Produce an animation of a dataset.

        Show the input data and the labels for each example.

        :param phase: Which dataset to animate. Either `"train"`, `"test"` or
                      `"validate"`.
        :param plot_preds: Whether or not to plot predictions.
        :param apply_weights: Whether to feed the weights to all `plot`
                              functions the images or to show the weights on
                              another row.
        :param preprocess: If True, apply preprocessing to the data.
        :param pred_dir: The name of the subdirectory within the dataset test
                         directory to restore the predictions from. Defaults to
                         the name of the network class. This should typically
                         be the network's name.
        """
        fig, axs, ims = self.plot_example(phase=phase,
                                          plot_preds=plot_preds,
                                          apply_weights=apply_weights,
                                          pred_dir=pred_dir)
        plt.tight_layout()

        def init():
            self.plot_example(phase=phase,
                              ims=ims,
                              plot_preds=plot_preds,
                              apply_weights=apply_weights, pred_dir=pred_dir)
            return ims

        def animate(_):
            self.plot_example(phase=phase,
                              ims=ims,
                              plot_preds=plot_preds,
                              apply_weights=apply_weights, pred_dir=pred_dir)
            return ims

        _ = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(self.files[phase]),
                                    interval=3000, blit=True, repeat=True)
        #plt.show()
        gc.collect()