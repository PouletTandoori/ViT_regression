from ModelGenerator_modified import (Sequence, Stratigraphy,
                            Property, Lithology, ModelGenerator, Deformation)
import os
# visible devices = 2 and 3:
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from GeoDataset_modified import GeoDataset
from SeismicGenerator_modified import SeismicGenerator, plot_shotgather
from SeismicGenerator_modified import SeismicAcquisition
from GraphIO_modified import Vsdepth, ShotGather, Vpdepth
import argparse
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
from tabulate import tabulate
import subprocess


#define parser
parser = argparse.ArgumentParser()
parser.add_argument('-tr_s','--trainsize',type=int,default=50,
                    help='Number of files to be generated in the training folder,'
                         '\n validation and test folders will have 30% of this number')
parser.add_argument('-data','--dataset_name',type=str,default='Halton_debug',
                    help='Name of the dataset, ex: Halton_debug')
parser.add_argument('--erase','-e',type=bool,default=True,
                    help='Erase the previous dataset: Default=True')
args = parser.parse_args()

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

#Define parameters for the dataset:
train_size= args.trainsize
validate_size= int(train_size*0.3)
test_size= int(train_size*0.3)

#Define parameters for aquisition geometry:
length_profile=96
dh = 0.5                                # Grid spacing in meters# Length of the line of receivers: 96m (=m*Zmax with 1<m<3)
minimal_offset=length_profile*dh        # Minimal offset: generally L/4
Profile=length_profile+minimal_offset   # Length of the profile: 96+24=120m
number_receivers=96                     # Number of receivers: 96
receiver_spacing=1                      # Receiver spacing: 1m (Zmin/k with 0.3<k<1.0)
Zmax=50                                 # Maximum depth of the profile: 50m

#Define parameters for the modelisation of the seismic data:
nab = 16                                            # Number of padding cells for the absorbing boundary
peak_freq = 10.0                                    # Peak frequency of the wavelet in Hertz
df = 2                                              # Frequency deviation for randomness
bonus_cells = 0                                     # Number of bonus cells to avoid disposal to close to the boundary

dt = 0.00007                                        # Time sampling interval in seconds
resampling = 10                                     # Resampling factor for the time axis
NbCellsX= int(Profile/dh)   # Number of grid cells in x direction (domain size + boundary)
NbCellsZ= int(Zmax/dh)                      # Number of grid cells in z direction (domain size + boundary)



class ViscoElasticModel(ModelGenerator):

    def __init__(self,):
        super().__init__()
        # Grid spacing in X, Y, Z directions (in meters).
        self.dh = 0.25
        # Number of grid cells in X direction.
        self.NX = NbCellsX
        # Number of grid cells in Z direction.
        self.NZ = NbCellsZ

        # Minimum thickness of a layer (in grid cells).
        self.layer_dh_min = 1/self.dh
        # Maximum thickness of a layer (in grid cells).
        self.layer_dh_max = 12.5/self.dh
        # Minimum number of layers.
        self.layer_num_min = 1
        # Fix the number of layers if not 0.
        self.num_layers = 0

        # If true, first layer dip is 0.
        self.dip_0 = True
        # Maximum dip of a layer.
        self.dip_max = 0
        # Maximum dip difference between two adjacent layers.
        self.ddip_max = 0

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
        self.fault_dip_max = 45
        # Minimum fault displacement.
        self.fault_displ_min = 0
        # Maximum fault displacement.
        self.fault_displ_max = 10
        # Bounds of the fault origin location.
        self.fault_x_lim = [0, self.NX]
        self.fault_y_lim = [0, self.NZ]
        # Maximum quantity of faults.
        self.fault_nmax = 1
        # Probability of having faults.
        self.fault_prob = 0.0

        self.thick0min = None
        self.thick0max = 50
        self.layers = None

        self._properties = None
        self._stratigraphy = None

    def VpVs_from_VpandVs(self, vp_min, Vs_min, vp_max, Vs_max):
        """
        Compute Vp/Vs min and max ratios from Vp and Vs.
        """
        vpvs_min = vp_min / Vs_max
        vpvs_max = vp_max / Vs_min
        return vpvs_min, vpvs_max

    def Summary_lithologies(self, strati, properties):
        print("Summary of the lithologies:\n")

        # Collect data for the table
        table_data = []
        for seq in strati.sequences:
            for lith in seq.lithologies:
                vp_min = next(prop.min for prop in lith.properties if prop.name == "vp")
                vp_max = next(prop.max for prop in lith.properties if prop.name == "vp")
                vs_min = next(prop.min for prop in lith.properties if prop.name == "vs")
                vs_max = next(prop.max for prop in lith.properties if prop.name == "vs")
                vpvs_min = next(prop.min for prop in lith.properties if prop.name == "vpvs")
                vpvs_max = next(prop.max for prop in lith.properties if prop.name == "vpvs")
                rho_min = next(prop.min for prop in lith.properties if prop.name == "rho")
                rho_max = next(prop.max for prop in lith.properties if prop.name == "rho")
                q_min = next(prop.min for prop in lith.properties if prop.name == "q")
                q_max = next(prop.max for prop in lith.properties if prop.name == "q")

                # Add a row for the lithology
                table_data.append([
                    lith.name, vp_min, vp_max,
                    vs_min, vs_max,
                    vpvs_min, vpvs_max,
                    rho_min, rho_max,
                    q_min, q_max
                ])

        # Define headers
        headers = [
            "Lithology", "vp_min", "vp_max",
            "vs_min", "vs_max",
            "vpvs_min", "vpvs_max",
            "rho_min", "rho_max",
            "q_min", "q_max"
        ]

        # Print the table
        print(tabulate(table_data, headers=headers, floatfmt=".2f", tablefmt="fancy_grid"))
        print('---------------------------------------------------')
        print('\n')

    def build_stratigraphy(self):
        lithologies = {}

        '''# Organic soils
        name = "Organic soils"
        vp = Property("vp", vmin=300, vmax=700)
        vs = Property("vs", vmin=100, vmax=300)
        print('Organic soils:')
        print('Vs_min:',vs.min,' Vs_max:',vs.max,' Vp_min:',vp.min,' Vp_max:',vp.max)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        print('Vp/Vs_min:',vpvs.min,' Vp/Vs_max:',vpvs.max)
        rho = Property("rho", vmin=500, vmax=1500)
        q = Property("q", vmin=5, vmax=20)
        organic_soils = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        # Dry sands
        name = "Dry Sands"
        vp = Property("vp", vmin=400, vmax=1200)
        vs = Property("vs", vmin=100, vmax=500)
        #print('Vs_min:',vs.min,' Vs_max:',vs.max,' Vp_min:',vp.min,' Vp_max:',vp.max)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        #print('Vp/Vs_min:',vpvs.min,' Vp/Vs_max:',vpvs.max)
        rho = Property("rho", vmin=1700, vmax=1900)
        q = Property("q", vmin=13, vmax=63)
        dry_sands = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Wet sands
        name = "Wet sands"
        vp = Property("vp", vmin=1500, vmax=2000)
        vs = Property("vs", vmin=400, vmax=600)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1800, vmax=2000)
        q = Property("q", vmin=13, vmax=63)
        wet_soils = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # silts
        name = "Silts"
        vp = Property("vp", vmin=1400, vmax=2100)
        vs = Property("vs", vmin=300, vmax=800)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1300, vmax=2200)
        q = Property("q", vmin=13, vmax=63)
        silts = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        # Clay
        name = "Clay"
        vp = Property("vp", vmin=1100, vmax=2500)
        vs = Property("vs", vmin=80, vmax=800)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1200, vmax=2000)
        q = Property("q", vmin=7, vmax=14)
        clay = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])

        '''#tills
        name = "Tills"
        vp = Property("vp", vmin=1600, vmax=3100)
        vs = Property("vs", vmin=300, vmax=1100)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1600, vmax=2100)
        q = Property("q", vmin=256, vmax=430)
        tills = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# sandstone
        name = "Sandstone" #grès
        vp = Property("vp", vmin=2000, vmax=3500)
        vs = Property("vs", vmin=800, vmax=1800)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2100, vmax=2400)
        q = Property("q", vmin=70, vmax=150)
        sandstone = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# Dolomite
        name = "Dolomite"  # dolomie
        vp = Property("vp", vmin=2500, vmax=6500)
        vs = Property("vs", vmin=1900, vmax=3600)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2300, vmax=2900)
        q = Property("q", vmin=100, vmax=600)
        dolomite = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# Limestone
        name = "Limestone"  # calcaire
        vp = Property("vp", vmin=2300, vmax=2600)
        vs = Property("vs", vmin=1100, vmax=1300)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=2600, vmax=2700)
        q = Property("q", vmin=100, vmax=600)
        limestone = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''

        '''# Shale
        name = "Shale"
        vp = Property("vp", vmin=2000, vmax=5000)
        vs = Property("vs", vmin=1000, vmax=2000)
        vpvs_min, vpvs_max = self.VpVs_from_VpandVs(vp.min, vs.min, vp.max, vs.max)
        vpvs = Property("vpvs", vmin=vpvs_min, vmax=vpvs_max)
        rho = Property("rho", vmin=1700, vmax=3300)
        q = Property("q", vmin=10, vmax=70)
        shale = Lithology(name=name, properties=[vp,vs, vpvs, rho, q])'''


        # set everything to zero if you want a tabular model.
        deform = Deformation(max_deform_freq=0.00,
                             min_deform_freq=0.0000,
                             amp_max=0,
                             max_deform_nfreq=0,
                             prob_deform_change=0)


        fine_sediments_seq = Sequence(name="fine_sediments_seq",
                                    lithologies=[dry_sands, wet_soils, silts, clay],
                                    thick_min=50,
                                    thick_max=50,
                                    deform=deform,
                                    skip_prob=0.0,
                                    ordered=False,
                                    accept_decrease=0.2,
                                    nmax=4
                                    )

        # dry soil, saturated soil, altered rock, granite
        sequences = [fine_sediments_seq]
        strati = Stratigraphy(sequences=sequences)

        properties = strati.properties()

        return strati, properties


    def generate_model(self, seed=235):
        props2D, layerids, layers,Halton_reduced = super().generate_model(seed=seed)
        props2D["vs"] = props2D["vp"] / props2D["vpvs"]

        return props2D, layerids, layers,Halton_reduced

#Display the summary of the lithologies:
model=ViscoElasticModel()
#model.Summary_lithologies(model.stratigraphy, model.properties)

#define MyAcquisition class using SeismicAcquisition class from SeismicGenerator.py
class MyAcquisition(SeismicAcquisition):
    '''
    We want to make 1.5s long seismograms with 0.02 ms sampling rate.
    If we want a minimum depth investigation of 2m and a maximum depth investigation of 50m, we need to have a grid spacing of 2/8 m = 0.25m.
    We want to have 50m long line of receivers with 96 receivers spaced by 1m.

    We will create a grid of 100m long * 50m depth grid cells with a grid spacing of 0.25m.

    '''

    def __init__(self, dh: float = dh, nx: int = NbCellsX, nz: int = NbCellsZ):
        super().__init__(dh=dh, nx=nx, nz=nz)

        # Time sampling for seismogram (in seconds).
        self.dt = dt
        # Number of times steps.
        self.NT = int(1.5/dt)  # around 1.5 sec
        # Peak frequency of input wavelet (in Hertz).
        self.peak_freq = peak_freq
        self.configuration = 'inline'
        # Minimum position of receivers (-1 = minimum of grid).
        self.gmin = None
        # Maximum position of receivers (-1 = maximum of grid).
        self.gmax = None

#plot the aquisition geometry:
aquire=MyAcquisition()
#aquire.plot_acquisition_geometry(model, path=path_figures)

class SimpleDataset(GeoDataset):
    name = args.dataset_name

    def set_dataset(self, *args, **kwargs):
        model=ViscoElasticModel()
        acquire = MyAcquisition()
        physic = SeismicGenerator(acquire=acquire)
        graphios = {ShotGather.name: ShotGather(model=model, acquire=acquire),
                    Vsdepth.name: Vsdepth(model=model, acquire=acquire),
                    Vpdepth.name: Vpdepth(model=model, acquire=acquire)}
        for name in graphios:
            graphios[name].train_on_shots = True

        return model, physic, graphios


dataset = SimpleDataset(trainsize=train_size, validatesize=validate_size, testsize=test_size,
                        toinputs={ShotGather.name: ShotGather.name},
                        tooutputs={Vsdepth.name: Vsdepth.name, Vpdepth.name: Vpdepth.name})

#create ModelAnimated:
print('Creating model visualisation:')
#dataset.model.animated_dataset(nframes=min(args.trainsize+args.validatesize+args.testsize,50),path=path_figures)
#generate dataset:
print('\n')
print('---------------------------------------------------')
print('Generating dataset:')
dataset.generate_dataset(nproc=3)
print('Dataset created')
print('---------------------------------------------------')
print('\n')

# erase all .lock files in train, validate and test folders:
os.system(f'rm Datasets/{args.dataset_name}/train/*.lock')
os.system(f'rm Datasets/{args.dataset_name}/validate/*.lock')
os.system(f'rm Datasets/{args.dataset_name}/test/*.lock')
print('Erasing lock files')


#smallest between 5 and the sum of the sizes of the training, validation, and test sets
nb_examples = min(5, args.trainsize + validate_size + test_size)
# Collect multiple examples from the dataset
examples = [dataset.get_example(phase="train") for _ in range(nb_examples)]
# Unpack the examples into separate lists for inputs, labels, weights, and filenames
inputspre_list, labelspre_list, weightspre_list, filename_list = zip(*examples)

#plot the example:
plot_shotgather(inputspre_list, labelspre_list,dz=dh,dt=dt,resample=resampling, nb_examples=nb_examples)

#change direction to /home/rbertille/data/pycharm/ViT_project/pycharm_ViT to make post processing
print('Post processing:')
os.chdir('/home/rbertille/data/pycharm/ViT_project/pycharm_ViT')
subprocess.run(['python', 'verify_data.py', '--dataset_name', args.dataset_name, '--clean', 'True'])
subprocess.run(['python','DistributionDataset.py','--dataset_name',args.dataset_name])