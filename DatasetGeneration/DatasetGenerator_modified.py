"""
Produce a dataset on multiple GPUs.

Used by the `GeoFlow.GeoDataset.GeoDataset` class.
"""

import os
from multiprocessing import Process, Queue
import queue
from typing import Dict

import numpy as np
import h5py as h5
from filelock import FileLock, Timeout

from ModelGenerator_modified import ModelGenerator
from GeoFlow.Physics.Physic import Physic, PhysicError
from GeoFlow.GraphIO import GraphIO
from tqdm import tqdm


class DatasetGenerator:
    """
    Generate a complete dataset.
    """

    def __init__(self, model: ModelGenerator, physic: Physic,
                 graphios: Dict[str, GraphIO], Halton_seq=None):
        """
        Generate a dataset as implied by the arguments.

        :param model: A `SimpleElastic` that can create the earth
                      properties.
        :param physic: A Physic object that can generate geophysical data.
        :param graphios: A dict of GraphIO that generate the inputs and labels
                        of the network from the generated data and earth
                        properties.
        """
        self.Halt_seq = None
        self.model = model
        self.graphios = graphios
        self.physic = physic
        self.Halt_seq = Halton_seq


    def new_example(self, seed, Halton_Seq=None):
        """
        Generate one example

        :param seed: Seed of the model to generate.
        """
        self.Halt_seq = Halton_Seq
        self.model.Halt_seq = self.Halt_seq
        if self.Halt_seq is None:
            print('[DatasetGenerator3] Halton Seq is None')


        try:
            results = self.model.generate_model(seed=seed)
            props, layerids, layers, Halt_seq_reduced = results
        except Exception as e:
            print(f'ERREUR [DatasetGenerator] new_example: {e}')
            # afficher le chemin jusqu'à generate_model:
            print('Chemin:', self.model.generate_model)
            quit()

        self.Halt_seq = Halt_seq_reduced

        data = self.physic.compute_data(props)

        features = {}
        weights = {}
        for name in self.graphios:
            label, weight = self.graphios[name].generate(data, props)
            features[name] = label
            weights[name] = weight

        return features, weights,self.Halt_seq

    def read(self, filename: str, toread: list = None):
        """
        Read one example from hdf5 file.

        :param filename: Name of the file.
        :param toread: List of features to read.

        :returns:
            features: A dictionary of toread' name-data pairs.
            weights: A dictionary of weights' name-values pairs.
        """
        print('Nom fichier:', filename)
        with h5.File(filename, "r") as file:
            if toread is None:
                toread =  list(self.graphios.keys())
                print('[DatasetGenerator;read] toread:', toread)
            weights = {}
            features = {}
            for key in toread:
                features[key] = file[key][:]
                if key+"_w" in file.keys():
                    weights[key] = file[key+"_w"][:]
                else:
                    weights[key] = None

        return features, weights

    def read_predictions(self, filename: str, load_dir: str, tooutputs: list = None):
        """
        Read one example's predictions from hdf5 file.

        :param filename: Name of the file.
        :param load_dir: The name of the subdirectory within the dataset test
                         directory to restore the predictions from. Defaults to
                         the name of the network class. This should typically
                         be the network's name.
        :param tooutputs: List of graphios to read predictions

        :returns:
            preds: A dictionary of predictions' name-values pairs.
        """
        directory, filename = os.path.split(filename)
        filename = os.path.join(directory, load_dir, filename)
        preds = {}
        with h5.File(filename, "r") as file:
            if tooutputs is None:
                tooutputs = self.graphios
            for key in tooutputs:
                name = key + "_pred"
                if name in file.keys() and file[name].size > 1:
                    preds[key] = file[key + "_pred"][:]
        return preds

    def write(self, exampleid, savedir, features, weights, filename=None):
        """
        Write one example in hdf5 format.

        :param exampleid: The example ID number.
        :param savedir The directory in which to save the example.
        :param features: A dictionary of graph inputs-outputs' name-values pairs.
        :param weights:  A dictionary of graph weights' name-values pairs.
        :param filename: If provided, save the example in filename.
        """
        if filename is None:
            filename = os.path.join(savedir, "example_%d" % exampleid)
        else:
            filename = os.path.join(savedir, filename)

        with h5.File(filename, "w") as file:
            for name in features:
                file[name] = features[name]
            for name in weights:
                if weights[name] is not None:
                    file[name+"_w"] = weights[name]

    def write_predictions(self, exampleid, savedir, preds, filename=None):
        """
        :param exampleid: The example ID number.
        :param savedir The directory in which to save the example.
        :param preds: A dictionary of graph outputs' name-values pairs.
        :param filename: If provided, save the example in filename.
        """
        if filename is None:
            filename = os.path.join(savedir, "example_%d" % exampleid)
        else:
            filename = os.path.join(savedir, filename)

        with h5.File(filename, "w") as file:
            for name in preds:
                sname = name + "_pred"
                if sname in file.keys():
                    del file[sname]
                file[sname] = preds[sname]

    def generate_dataset(self,
                         savepath: str,
                         nexamples: int,
                         seed0: int = None,
                         nproc: int = 1):
        """
        Create a dataset on multiple GPUs.

        :param savepath: Root path of the dataset.
        :param nexamples: Quantity of examples to generate.
        :param seed0: First seed of the first example in the dataset. Seeds are
                      incremented by 1 at each example.
        :param nproc: Number of parralel generators to use. The Physics class
                      handle which ressource should be used
        :param Halton_seq: Halton sequence to use for stratified sampling. Size (nexamples, 6). Used as Queue

        :returns:
            Halton_seq: The Halton sequence after the last example. Values used are removed from the sequence.
        """
        try:
            os.makedirs(savepath)
        except FileExistsError:
            pass

        exampleids = Queue()
        print(f"Generating {nexamples} examples")
        for el in np.arange(seed0, seed0 + nexamples):
            exampleids.put(el)
        generators = []
        for i in range(nproc):
            sg = self.__class__(model=self.model, physic=self.physic.copy(i),
                                graphios=self.graphios)
            if self.Halt_seq is None:
                print('[DatasetGenerator1] Halton Queue is None')
            thisgen = DatasetProcess(savepath, sg, exampleids, self.Halt_seq)
            self.Halt_seq = thisgen.run()
            thisgen.start()
            generators.append(thisgen)
        for gen in generators:
            gen.join()

        return self.Halt_seq



class DatasetProcess(Process):
    """
    Create a new process to generate seismic data.
    """

    def __init__(self,
                 savepath: str,
                 data_generator: DatasetGenerator,
                 seeds: Queue,
                 Halton_seq= None):
        """
        Initialize a `DatasetGenerator` object.

        :param savepath: Path at which to create the dataset.
        :param data_generator: A `DatasetGenerator` object to create examples.
        :param seeds: A `Queue` containing the seeds of models to generate.
        """
        super().__init__()

        self.savepath = savepath
        self.data_generator = data_generator
        self.seeds = seeds
        self.Halton_queue = Halton_seq
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

    def run(self):
        """
        Start the process to generate data.
        """
        # Obtenir le nombre total de seeds
        total_seeds = self.seeds.qsize()

        if self.Halton_queue is None:
            print('[DatasetProcess2] Halton Queue is None')

        # Utiliser une barre de progression
        with tqdm(total=total_seeds, desc="Generating dataset",colour='green') as pbar:

            while not self.seeds.empty():
                try:
                    # Récupérer un seed depuis la queue
                    seed = self.seeds.get(timeout=1)
                except queue.Empty:
                    print("WARNING: Queue is empty")
                    break

                filename = f"example_{seed}"
                filepath = os.path.join(self.savepath, filename)

                # Vérifier si le fichier existe déjà
                if not os.path.isfile(filepath):
                    try:
                        with FileLock(filepath + '.lock', timeout=0):
                            print('filename:', filename)
                            feats, weights,self.Halton_queue = self.data_generator.new_example(seed, self.Halton_queue)
                            print('[DatasetProcess2] Halton Queue reduced shape:', self.Halton_queue.shape)
                            self.data_generator.write(seed, self.savepath, feats,
                                                      weights, filename=filename)

                    except Timeout:
                        print(f"WARNING: {filepath} is locked")
                        continue
                    except (ValueError, PhysicError) as error:
                        print(f"WARNING: {error}")
                        os.remove(filepath + '.lock')
                else:
                    os.remove(filepath + '.lock')

                # Mettre à jour la barre de progression après traitement
                pbar.update(1)
            pbar.close()
        return self.Halton_queue




