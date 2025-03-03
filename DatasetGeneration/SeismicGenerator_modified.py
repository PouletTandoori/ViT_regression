# -*- coding: utf-8 -*-
"""
Generate the labels (seismic data) using SeisCL.
Requires the SeisCL python interface.

Modified from the original version by Rémy Bertille
Change Seismic Aquisition class to allow more personalization

"""

import numpy as np
import os
from matplotlib import pyplot as plt
from SeisCL.SeisCL import SeisCL, SeisCLError
from GeoFlow.SeismicUtilities.Numpy import random_wavelet_generator
from GeoFlow.Physics import Physic, PhysicError
from GeoFlow.GraphIO import Vsdepth, ShotGather, Vpdepth,Dispersion


'''class SeismicAcquisition:
    """
    Define all parameters needed to model seismic data with SeisCL
    """

    def __init__(
        self,
        nab=16,                # Number of padding cells for the absorbing boundary
        bonus_cells=20,        # Number of padding cells in addition to the absorbing boundary
        dh=0.25,               # Grid spacing in meters
        nx=400 + 2 * 16,       # Number of grid cells in x direction (domain size + boundary)
        nz=200 + 2 * 16,       # Number of grid cells in z direction (domain size + boundary)
        ny=None,               # Number of grid cells in y direction (3D case), None for 2D
        fs=True,               # Enable free surface condition at the top
        NT=100000,             # Number of time steps in the simulation
        dt=0.00002,            # Time sampling interval in seconds
        peak_freq=10,          # Peak frequency of the wavelet in Hertz
        wavefuns=[1],          # Source wavelet function selection
        df=0,                  # Frequency deviation for randomness
        tdelay=None,           # Time delay for the source (computed dynamically if None)
        resampling=100,        # Resampling factor for the time axis
        source_depth=0,        # Depth of the source in meters
        receiver_depth=0,      # Depth of the receivers in meters
        dg=1,                  # Receiver spacing in meters
        ds=2,                  # Source spacing in 2D
        gmin=None,             # Minimum position of receivers
        gmax=None,             # Maximum position of receivers
        minoffset=0,           # Minimum offset for recording
        sourcetype=2,          # Source type (pressure source or force)
        rectype=1,             # Recording type (pressure or velocities)
        abs_type=2,            # Type of absorbing boundary (1 for CPML, 2 for Cerjan)
        singleshot=True,       # Enable single-shot simulation
        configuration='inline',# Geophone configuration: inline or full
        L=0,                   # Number of attenuation mechanisms (L=0 for elastic)
        FL=np.array(15),       # Frequencies for attenuation mechanisms
    ):
        # Initialisation des attributs
        self.nab = nab
        self.bonus_cells = bonus_cells
        self.dh = dh
        self.nx = nx
        self.nz = nz
        self.ny = ny
        self.fs = fs
        self.NT = NT
        self.dt = dt
        self.peak_freq = peak_freq
        self.wavefuns = wavefuns
        self.df = df
        self.tdelay = tdelay or 2.0 / (peak_freq - df)  # Default delay calculation
        self.resampling = resampling
        self.source_depth = source_depth
        self.receiver_depth = receiver_depth
        self.dg = dg
        self.ds = ds
        self.gmin = gmin
        self.gmax = gmax
        self.minoffset = minoffset
        self.sourcetype = sourcetype
        self.rectype = rectype
        self.abs_type = abs_type
        self.singleshot = singleshot
        self.configuration = configuration
        self.L = L
        self.FL = FL

    def set_rec_src(self):
        """
        Provide the sources' and receivers' positions for SeisCL.

        Override to change which data is modelled if needed.

        :return:
            src_pos: Source array.
            rec_pos: Receiver array.
        """
        assert self.configuration in ['inline', 'full']

        ng = 96 # Number of geophones

        if self.singleshot:
            # Add just one source on the left side
            left_side = self.nab + self.bonus_cells # unit=cells
            sx = np.array([left_side * self.dh]) # unit=meters
        elif self.configuration == 'inline':
            # Compute several sources
            left_side = self.nab + self.bonus_cells  # unit=cells
            start_idx = left_side # unit=cells
            if self.gmin is not None and self.gmin < 0:
                start_idx += -self.gmin # unit=cells
            end_idx = self.nx - self.nab # unit=cells
            if self.gmax is not None and self.gmax > 0:
                end_idx += -self.gmax # unit=cells
            sx = np.arange(start_idx, end_idx, self.ds) * self.dh # unit=meters
        elif self.configuration == 'full':
            # Compute several sources
            left_side = self.nab + self.bonus_cells  # unit=cells
            start_idx = left_side
            right_side= self.nx - self.nab - self.bonus_cells # unit=cells
            end_idx = right_side # unit=cells
            sx = np.arange(start_idx, end_idx, self.ds) * self.dh # unit=meters
        sz = np.full_like(sx, self.source_depth) # unit=meters
        sid = np.arange(0, sx.shape[0])

        src_pos = np.stack([sx,
                            np.zeros_like(sx),
                            sz,
                            sid,
                            np.full_like(sx, self.sourcetype)], axis=0)

        # Add receivers (consider sx as origin)
        if self.gmin is not None:
            gmin = self.gmin
        else:
            if self.configuration == 'inline':
                gmin = (self.minoffset / self.dh) # unit=cells

            elif self.configuration == 'full':
                self.bonus_cells + self.nab + self.minoffset # unit=cells

        if self.gmax is not None:
            gmax = self.gmax
        else:
            if self.configuration == 'inline':
                gmax = gmin + (ng * self.dg) / self.dh
            elif self.configuration == 'full':
                gmax = gmin + ng * self.dg # unit=cells


        gx0 = np.arange(gmin * self.dh, gmax* self.dh, self.dg ) # unit=meters
        gsid = np.concatenate([np.full_like(gx0, s) for s in sid], axis=0)
        if self.configuration == 'inline':
            gx = np.concatenate([s + gx0 for s in sx], axis=0)
        elif self.configuration == 'full':
            gx = np.concatenate([gx0 for _ in sx], axis=0)
        gz = np.full_like(gx, self.receiver_depth)
        gid = np.arange(0, len(gx))

        rec_pos = np.stack([gx,
                            np.zeros_like(gx),
                            gz,
                            gsid,
                            gid,
                            np.full_like(gx, 2),
                            np.zeros_like(gx),
                            np.zeros_like(gx)], axis=0,)

        print('dg:', self.dg, 'm')
        print('gmin:', gmin * self.dh, 'm')
        print('gmax:', gmax * self.dh, 'm')

        print('src_pos:', src_pos[0][0], 'm')

        print('rec_pos:', rec_pos[0])
        print('len rec_pos:', len(rec_pos[0]))
        print('-----------------------------------')
        print('\n')

        return src_pos, rec_pos

    def source_generator(self):
        return random_wavelet_generator(self.NT, self.dt, self.peak_freq,
                                        self.df, self.tdelay)

    def plot_acquisition_geometry(self, model, path=None):
        src_pos, rec_pos = self.set_rec_src()
        qty_srcs = src_pos.shape[1]
        src_x, _, _, src_id, _ = src_pos
        rec_x, _, _, rec_src_id, _, _, _, _ = rec_pos
        props, _, _ = model.generate_model()
        vs = props['vs']
        x_max = self.nx * self.dh
        z_max = self.nz * self.dh

        # Adjust the figure height based on the number of sources
        height_geometry = 12 * (qty_srcs + 1) / 72
        fig, axs = plt.subplots(
            nrows=2,
            sharex=True,
            figsize=(8, 8 + height_geometry),
            gridspec_kw={'height_ratios': [height_geometry, 8], 'hspace': 0}
        )

        # Plot sources and receivers
        axs[0].scatter(src_x, src_id, marker='x', s=12, label="Sources", color='red')
        axs[0].scatter(rec_x, rec_src_id, marker='v', s=10, label="Receivers", color='blue')
        axs[0].set_axis_off()
        axs[0].set_xlim([0, x_max])
        axs[0].set_ylim([-.5, qty_srcs - .5])

        # Plot velocity model
        axs[1].imshow(vs, origin='upper', extent=[0, x_max, z_max, 0], aspect='auto')
        axs[1].set_ylabel("Depth (m)")
        axs[1].set_xlabel("Position (m)")

        # Create a single legend for the figure
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.9, 0.01), ncol=2,fontsize=14)

        # Adjust layout and save/show the figure
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for the legend
        if path is not None:
            plt.savefig(f'{path}/acquisition_geometry.pdf', format='pdf', dpi=300)
        plt.show()
        plt.close()'''


class SeismicAcquisition:
    """
    Define all parameters needed to model seismic data with SeisCL

    min depth investigation: around 1m
    max depth investigation: around 50 m

    Length of the line of receivers: 96m (=m*Zmax with 1<m<3)
    Number of receivers: 96
    Receiver spacing: 1m (Zmin/k with 0.3<k<1.0)
    distance between the source and the receivers: 1m (dg)
    """

    def __init__(self, dh: float = 0.25, nx: int = 400, nz: int = 200, ny: int = None):

        # Number of padding cells of absorbing boundary.
        self.nab = 16

        self.dh = dh  # grid spacing, in meters
        self.nx = nx + 2 * self.nab  # number of grid cells in x direction
        self.nz = nz + 1 * self.nab  # number of grid cells in z direction
        self.ny = ny

        # Whether free surface is turned on the top face.
        self.fs = True
        # Number of times steps.
        self.NT = 100000
        # Time sampling for seismogram (in seconds).
        self.dt = 0.00002
        # Peak frequency of input wavelet (in Hertz).
        self.peak_freq = 10.0  # could be higher
        # Source wave function selection.
        self.wavefuns = [1]
        # Frequency of source peak_freq +- random(df).
        self.df = 2
        # Delay of the source.
        self.tdelay = 2.0 / (self.peak_freq - self.df)
        # Resampling of the shots time axis.
        self.resampling = 100  # ex: choose 10 to divide time axis by 10
        # Depth of sources (m).
        self.source_depth = 0
        # Depth of receivers (m).
        self.receiver_depth = 0
        # Receiver interval in meters
        self.dg = 1
        # Source interval (in 2D).
        self.ds = 2
        # Minimum position of receivers (-1 = minimum of grid).
        self.gmin = None
        # Maximum position of receivers (-1 = maximum of grid).
        self.gmax = None
        self.minoffset = (nx - 2 * self.nab)/4
        # Integer used by SeisCL for pressure source (100) or force in z (2).
        self.sourcetype = 2
        # Integer used by SeisCL indicating which type of recording. Either
        # 2) pressure or 1) velocities.
        self.rectype = 1  # if we choose velocities, then we obtain 2 shotgathers instead of 1
        # Absorbing boundary type:
        # 1) CPML or 2) absorbing layer of Cerjan.
        self.abs_type = 2

        self.singleshot = True
        # Whether to fill the surface with geophones or to use inline spread.
        # Either `'full'` or `'inline'`.
        self.configuration = 'inline'
        # Number of attenuation mechanism (L=0 elastic)
        self.L = 0
        # Frequencies of the attenuation mechanism
        self.FL = np.array(15)

    def set_rec_src(self):

        '''print('nab in set_rec_src:', self.nab)
        print('nx in set_rec_src:', self.nx)
        print('nz in set_rec_src:', self.nz)'''

        dg = self.dg / self.dh
        # convert into grid points
        # print('grid points dg:',dg)
        ng = 96  # Quantity of geophones.

        # Add receiver.
        if self.gmin:
            gmin = self.gmin
        else:
            gmin = self.nab + self.minoffset

        if self.gmax:
            gmax = self.gmax
        else:
            gmax = gmin + ng * dg

        # Add sources.
        sx = [gmin * self.dh - self.minoffset * self.dh]
        '''print('grd pts sx:', sx)
        print('grd pts gmin:', gmin)
        print('grd pts gmax:', gmax)'''

        # Set source.
        sz = np.full_like(sx, self.source_depth)
        sid = np.arange(0, len(sx))

        src_pos = np.stack([sx,
                            np.zeros_like(sx),
                            sz,
                            sid,
                            np.full_like(sx, self.sourcetype)], axis=0)

        # Set receivers
        gx0 = np.arange(gmin, gmax, dg) * self.dh
        gx = np.concatenate([gx0 for _ in sx], axis=0)
        gsid = np.concatenate([np.full_like(gx0, s) for s in sid], axis=0)
        gz = np.full_like(gx, self.receiver_depth)
        gid = np.arange(0, len(gx))

        rec_pos = np.stack([gx,
                            np.zeros_like(gx),
                            gz,
                            gsid,
                            gid,
                            np.full_like(gx, 2),
                            np.zeros_like(gx),
                            np.zeros_like(gx)], axis=0)

        '''print('dg:', dg)
        print('gmin:', gmin)
        print('gmax:', gmax)

        print('src_pos:', src_pos[0])
        print('rec_pos:', rec_pos[0])'''

        return src_pos, rec_pos

    def source_generator(self):
        print('SOURCE GENERATOR:')
        print('NT=', self.NT)
        print('dt=', self.dt)
        return random_wavelet_generator(self.NT, self.dt, self.peak_freq,
                                        self.df, self.tdelay)

    def plot_acquisition_geometry(self, model, path=None):
        src_pos, rec_pos = self.set_rec_src()
        qty_srcs = src_pos.shape[1]
        src_x, _, _, src_id, _ = src_pos
        rec_x, _, _, rec_src_id, _, _, _, _ = rec_pos
        props, _, _ = model.generate_model()
        vs = props['vs']
        x_max = self.nx * self.dh
        z_max = self.nz * self.dh

        # Adjust the figure height based on the number of sources
        height_geometry = 12 * (qty_srcs + 1) / 72
        fig, axs = plt.subplots(
            nrows=2,
            sharex=True,
            figsize=(8, 8 + height_geometry),
            gridspec_kw={'height_ratios': [height_geometry, 8], 'hspace': 0}
        )

        # Plot sources and receivers
        axs[0].scatter(src_x, src_id, marker='x', s=12, label="Sources", color='red')
        axs[0].scatter(rec_x, rec_src_id, marker='v', s=10, label="Receivers", color='blue')
        axs[0].set_axis_off()
        axs[0].set_xlim([0, x_max])
        axs[0].set_ylim([-.5, qty_srcs - .5])

        # Plot velocity model
        axs[1].imshow(vs, origin='upper', extent=[0, x_max, z_max, 0], aspect='auto')
        axs[1].set_ylabel("Depth (m)")
        axs[1].set_xlabel("Position (m)")

        # Create a single legend for the figure
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.9, 0.01), ncol=2,fontsize=14)

        # Adjust layout and save/show the figure
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for the legend
        if path is not None:
            plt.savefig(f'{path}/acquisition_geometry.pdf', format='pdf', dpi=300)
        #plt.show()
        plt.close()

class SeismicGenerator(Physic):
    """
    Generate seismic data with SeisCL.
    """

    def __init__(self, acquire: SeismicAcquisition, pid: int=0, ):
        """
        Interface to SeisCL to generate seismic data.

        :param acquire: An SeismicAcquisition object that defines the seismic acquisition.
        :param pid : GPU pid to use.
        """
        super().__init__(pid)

        self.acquire = acquire
        workdir = "workdir_seismic" + str(pid)
        if acquire.ny is not None:
            N = np.array([acquire.nz, acquire.nx, acquire.ny])
            nd = 3
        else:
            N = np.array([acquire.nz, acquire.nx])
            nd = 2

        # Assign the GPU to SeisCL.
        nouse = np.arange(0, 16)
        nouse = nouse[nouse != self.pid]

        self.src_pos_all, self.rec_pos_all = acquire.set_rec_src()

        self.seiscl = SeisCL(workdir=workdir,
                             N=N,
                             ND=nd,
                             dh=acquire.dh,
                             nab=acquire.nab,
                             dt=acquire.dt,
                             NT=acquire.NT,
                             f0=acquire.peak_freq,
                             seisout=acquire.rectype,
                             freesurf=int(acquire.fs),
                             abs_type=acquire.abs_type,
                             L=acquire.L,
                             FL=acquire.FL,
                             no_use_GPUs = nouse,
                             rec_pos_all=self.rec_pos_all,
                             src_pos_all=self.src_pos_all,
                             )
        self.resampling = acquire.resampling

        #self.wavelet_generator = acquire.source_generator()

    def compute_data(self, props: dict):
        """
        Compute the seismic data of a model.

        :param props: A dictionary of properties' name-values pairs.

        :return: An array containing the modeled seismic data.
        """
        self.seiscl.src_all = None  # Reset source to generate new random source.
        if not os.path.isdir(self.seiscl.workdir):
            print("pid:" + self.pid + ", workdir is " + self.seiscl.workdir)
            os.mkdir(self.seiscl.workdir)
        try:

            self.seiscl.set_forward(self.src_pos_all[3, :], props,
                                    withgrad=False)
            self.seiscl.execute()
            data = self.seiscl.read_data()
        except SeisCLError as e:
            raise PhysicError("SeisCL error: " + str(e))
        data = [el[::self.resampling, :] for el in data]
        # concatenate all the recorded components on axis 1
        data = np.concatenate(data, axis=1)
        return data

    def copy(self, pid):
        """
        Copy the physic object.

        :returns:
            physic: A new physic object.
        """
        selfcopy = self.__class__(self.acquire, pid=pid)
        return selfcopy


def plot_shotgather(liste_inputspre, liste_labelspre,dz=1,dt=0.00002,resample=100, path=None, nb_examples=1):

    # Initialize figure with multiple rows for examples
    fig, axs = plt.subplots(nb_examples, 2, figsize=(10, 6 * nb_examples))
    if nb_examples == 1:
        axs = [axs]  # Ensure axs is iterable when nb_examples is 1

    for i in range(nb_examples):
        print(f'Example {i + 1}:')
        inputspre = liste_inputspre[i]
        labelspre = liste_labelspre[i]
        print('shape inputspre:', inputspre[ShotGather.name].shape)

        # Check for NaN values in input and label
        if np.isnan(inputspre[ShotGather.name]).any():
            print(f'Warning: NaN values detected in ShotGather for example {i + 1}')
        if np.isnan(labelspre[Vsdepth.name]).any():
            print(f'Warning: NaN values detected in Vsdepth for example {i + 1}')

        time_vector = np.arange(inputspre[ShotGather.name].shape[0]) * (dt * resample)
        nb_traces = int(inputspre[ShotGather.name].shape[1] / 2)
        print('nb_traces:', nb_traces)

        depth_vector = np.arange(labelspre[Vsdepth.name].shape[0]) * dz

        # Print shape of the shot gather
        print('shape of the shot gather:', inputspre[ShotGather.name][:, nb_traces:].shape)

        '''# Normalize each trace individually
        for trace in range(inputspre[ShotGather.name].shape[1]):
            inputspre[ShotGather.name][:, trace] = (
                (inputspre[ShotGather.name][:, trace] - np.min(inputspre[ShotGather.name][:, trace]))
                / (np.max(inputspre[ShotGather.name][:, trace]) - np.min(inputspre[ShotGather.name][:, trace]))
            )'''

        #define shot gather:
        shotgather = inputspre[ShotGather.name][:, nb_traces:]

        # Plot only the second half of the traces
        axs[i][0].imshow(shotgather,
                         extent=[1, nb_traces, time_vector[-1], time_vector[0]],
                         aspect='auto', cmap='gray')
        axs[i][0].set_title(f'Shot Gather num {i + 1}')
        axs[i][0].set_ylabel('Time (s)', fontsize=15)
        axs[i][0].set_xlabel('Traces', fontsize=15)

        axs[i][1].plot(labelspre[Vsdepth.name], depth_vector, label='Vs', linewidth=1)
        axs[i][1].invert_yaxis()
        axs[i][1].set_ylabel('z (m)', fontsize=15)
        axs[i][1].set_xlabel('Vs (m/s)', fontsize=15)
        axs[i][1].set_title(f'Vs = f(z) - Exemple {i + 1}')

    plt.tight_layout()
    if path is not None:
        plt.savefig(f'{path}/Shotgather_proposal.pdf', format='pdf', dpi=300)
    #plt.show()
    plt.close()


