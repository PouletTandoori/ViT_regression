import tensorflow as tf
import math
import numpy as np
import torch

def acquisition_parameters(dataset_name,max_c=3.5,display=False):
    '''
    define some parameters
    '''
    #print('Dataset:', dataset_name)
    if dataset_name == 'Dataset1Dsmall' or dataset_name == 'Dataset1Dbig' or dataset_name == 'TutorialDataset':
        # define some parameters
        fmax = 35  # maximum frequency
        dt = 0.00002 * 100  # timestep * resampling
        src_pos = [28.]  # source position
        rec_pos = np.array([29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46.,
                            47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
                            65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76.])  # receivers positions
        dg = np.abs(rec_pos[0] - rec_pos[1])  # receiver spacing
        off0 = np.abs(rec_pos[0] - src_pos[0])  # minimum offset
        off1 = np.abs(rec_pos[-1] - rec_pos[0])  # maximum offset
        ng = rec_pos.shape[-1]  # number of receivers
        offmin, offmax = np.min([off0, off1]), np.max([off0, off1])
        x = rec_pos - src_pos[0]  # offsets
        c = np.logspace(1.5, 3.9, num=200)  # phase velocities

    elif dataset_name == 'Dataset1Dhuge_96tr' or dataset_name == 'Dataset1Dsimple':
        print('Considering 96 traces shot gathers')

        fmax = 35  # maximum frequency
        dt = 0.00002 * 100  # timestep * resampling
        src_pos = [3.]  # source position
        rec_pos = np.array([4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
                            17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28.,
                            29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
                            41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52.,
                            53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
                            65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76.,
                            77., 78., 79., 80., 81., 82., 83., 84., 85., 86., 87., 88.,
                            89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99.]
)  # receivers positions
        dg = np.abs(rec_pos[0] - rec_pos[1])  # receiver spacing
        off0 = np.abs(rec_pos[0] - src_pos[0])  # minimum offset
        off1 = np.abs(rec_pos[-1] - rec_pos[0])  # maximum offset
        ng = rec_pos.shape[-1]  # number of receivers
        offmin, offmax = np.min([off0, off1]), np.max([off0, off1])
        x = rec_pos - src_pos[0]  # offsets
        c = np.logspace(2, np.log10(max_c), num=224)  # phase velocities

    elif dataset_name == 'Halton_debug' or dataset_name == 'For_inversion_example' or dataset_name == 'Halton_Dataset' :

        fmax = 35  # maximum frequency
        dt = 0.00007  * 100  # timestep * resampling
        src_pos = [8.]  # source position

        rec_pos = np.array([ 40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.,  52.,  53.,
            54.,  55.,  56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.,  64.,  65.,  66.,  67.,
            68.,  69.,  70.,  71.,  72.,  73.,  74.,  75. , 76.,  77.,  78.,  79.,  80.,  81.,
            82.,  83.,  84.,  85.,  86.,  87. , 88.,  89.,  90.,  91.,  92.,  93.,  94.,  95.,
            96.,  97. , 98. , 99. ,100., 101., 102., 103. ,104., 105., 106., 107., 108., 109.,
            110., 111., 112., 113., 114., 115., 116., 117., 118., 119., 120., 121., 122., 123.,
            124., 125., 126., 127., 128., 129. ,130. ,131. ,132. ,133. ,134. ,135.])  # receivers positions
        dg = np.abs(rec_pos[0] - rec_pos[1])  # receiver spacing
        off0 = np.abs(rec_pos[0] - src_pos[0])  # minimum offset
        off1 = np.abs(rec_pos[-1] - rec_pos[0])  # maximum offset
        ng = rec_pos.shape[-1]  # number of receivers
        offmin, offmax = np.min([off0, off1]), np.max([off0, off1])
        x = rec_pos - src_pos[0]  # offsets
        c = np.logspace(1.5, max_c, num=224)  # phase velocities

    else:
        print('Be careful the parameters for generating the dispersion images may be wrong !')
        # define some parameters
        fmax = 25  # maximum frequency
        dt = 0.00002 * 100  # timestep * resampling
        src_pos = [28.]  # source position
        rec_pos = np.array([29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46.,
                            47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
                            65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76.])  # receivers positions
        dg = np.abs(rec_pos[0] - rec_pos[1])  # receiver spacing
        off0 = np.abs(rec_pos[0] - src_pos[0])  # minimum offset
        off1 = np.abs(rec_pos[-1] - rec_pos[0])  # maximum offset
        ng = rec_pos.shape[-1]  # number of receivers
        offmin, offmax = np.min([off0, off1]), np.max([off0, off1])
        x = rec_pos - src_pos[0]  # offsets
        c = np.logspace(1.5, 3.5, num=224)  # phase velocities

    if display == True:
        print('fmax:', fmax, '\ndt:', dt, '\nsrc_pos:', src_pos, '\nrec_pos:', rec_pos, '\ndg:', dg, '\noff0:', off0, '\noff1:', off1, '\nng:', ng, '\noffmin:', offmin, '\noffmax:', offmax, '\nx:', x, '\nc:', c)

    return fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c

def phase_shift_sum_base(d, f, x, c, adjoint=False):
    """
    Apply the phase shift summation to a signal in the frequency domain

    :param d: Data with shape [..., nx, nf] if adjoint is False,
              or [..., nc, nt] if adjoint is True.
    :param f: Frequency vector with nf components
    :param x: A 1D array containing the full offset of the nx traces
    :param c: A 1D array containing the nc velocities
    :param adjoint: If True, goes from the velocity-freq to offset-freq domain,
                    if False, goes from offset-freq to velocity-freq domain.

    :return:
        m: Phase-shifted summation of the signal. If adjoint, the dimensions
           is [..., nx, nf], else is [..., nc, nf]
    """

    nd = tf.rank(d)
    inds = tf.concat([tf.ones(nd-1, dtype=tf.int32), [-1]], axis=0)
    f = tf.cast(tf.reshape(f, inds), d.dtype)
    #print('f2',f)

    c = tf.cast(c, d.dtype)
    nc = c.shape[-1]

    inds = tf.concat([tf.ones(nd - 2, dtype=tf.int32), [-1, 1]], axis=0)
    x = tf.cast(x, d.dtype)
    nx = x.shape[-1]

    if adjoint:
        c = tf.reshape(c, inds)
        m = tf.TensorArray(d.dtype, size=nx)
        for ix in tf.range(nx):
            i = tf.cast(tf.complex(0.0, -1.0), d.dtype)
            delta = tf.exp(i * 2 * math.pi * f * x[ix] / c)
            m = m.write(ix, tf.reduce_sum(delta * d, axis=-2))
    else:
        x = tf.reshape(x, inds)
        m = tf.TensorArray(d.dtype, size=nc)
        for ic in tf.range(nc):
            i = tf.cast(tf.complex(0.0, 1.0), d.dtype)
            delta = tf.exp(i * 2 * math.pi * f * x / c[ic])
            m = m.write(ic, tf.reduce_sum(delta * d, axis=-2))

    leading = tf.range(1, nd - 1)
    trailing = tf.constant([-1]) + tf.rank(f)
    new_order = tf.concat([leading, [0], trailing], axis=0)

    return tf.transpose(m.stack(), new_order)

def phase_shift_sum(d, f, x, c, adjoint=False):
    """
    Apply the phase shift summation to a signal in the frequency domain,
    and provide a custom gradient.

    :param d: Data with shape [..., nx, nf] if adjoint is False,
              or [..., nc, nt] if adjoint is True.
    :param f: Frequency vector with nf components
    :param x: A 1D array containing the full offset of the nx traces
    :param c: A 1D array containing the nc velocities
    :param adjoint: If True, goes from the velocity-freq to offset-freq domain,
                    if False, goes from offset-freq to velocity-freq domain.

    :return:
        m: Phase-shifted summation of the signal. If adjoint, the dimensions
           is [..., nx, nf], else is [..., nc, nf]
    """
    dout = phase_shift_sum_base(d, f, x, c, adjoint=adjoint)
    return dout

    # Note: The gradient function is omitted here

def linear_radon_freq(d, dt, x, c, fmax=None, norm=False, epsilon=0.001):

    nt = d.shape[-1]
    d_fft = tf.signal.rfft(d)
    if norm: d_fft /= tf.cast(tf.abs(d_fft) + epsilon*tf.math.reduce_max(tf.abs(d_fft)),dtype=tf.complex64)
    fnyq = 1.00 / (nt*dt) * (nt//2+1)
    if fmax is None:
        fmax = fnyq
    if fmax > fnyq:
        raise ValueError("fmax=%f is greater than nyquist=%f"
                         % (fmax, 0.5 / dt))
    f = tf.range(fmax, delta=1.00 / (nt*dt))
    #print('f:', f)
    nf = f.shape[-1]

    d_fft = d_fft[..., :nf]

    m = phase_shift_sum(d_fft, f, x, c)

    return m

def dispersion(d, dt, x, c, fmax=None,epsilon=0.001):
    '''
    calculate the dispersion image using phase shift method
    :param d: shotgather
    :param dt: timestep
    :param x: position of the receivers
    :param c:
    :param fmax:
    :param epsilon:
    :return:
    '''
    #print('d:', d.shape)
    #print('d0*dt:', d.shape[1]*dt)
    return tf.abs(linear_radon_freq(d, dt, x, c, fmax=fmax, norm=True,epsilon=epsilon))

def prepare_disp_for_NN(disp):
    # normalize the disp values
    disp = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
    #store shape:
    shape0,shape1= np.shape(disp)
    print('disp shape:', np.shape(disp))
    # add padding to the disp to reach 224,224
    disp = np.pad(disp, ((0, 224 - disp.shape[0]), (0, 224 - disp.shape[1])), 'constant', constant_values=(0, 0))
    #print('disp after reshape shape:', np.shape(disp))

    # transform into tensor
    disp = torch.tensor(disp, dtype=torch.float32)
    #print('disp shape tensor:', np.shape(disp))

    # transform into 3 channels
    disp = disp.repeat(3, 1, 1)
    #print('disp shape tensor:', np.shape(disp))

    return disp,shape0,shape1

# test the programm for Halton_debug dataset (if main ....)
if __name__ == '__main__':
    # Use PytorchDataset to load the data and directly create dataloaders with shot gathers and Vs profiles
    from PytorchDataset import create_dataloaders

    # defines some variables:
    data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_ViT/DatasetGeneration/Datasets/'
    dataset_name = 'Halton_debug'

    # data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
    # dataset_name = 'Dataset1Dsimple'

    # create pytorch dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        data_path=data_path,
        dataset_name=dataset_name,
        batch_size=1,
        use_dispersion=True,
        data_augmentation=False
    )

    # select only 1 exemle1 for this exercice:
    train_dataloader.dataset.data = train_dataloader.dataset.data[:1]
    # and associated labels:
    train_dataloader.dataset.labels = train_dataloader.dataset.labels[:1]
    # define some parameters
    fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters('Halton_debug', max_c=3.5)
    print('fmax:', fmax, 'dt:', dt, 'src_pos:', src_pos, 'rec_pos:', rec_pos, 'dg:', dg, 'off0:', off0, 'off1:', off1, 'ng:', ng, 'offmin:', offmin, 'offmax:', offmax, 'x:', x, 'c:', c)

    from matplotlib import pyplot as plt
    # Then create dispersion images:
    for i in range(len(train_dataloader)):
        sample = train_dataloader.dataset[i]
        inputs = sample['data']
        # create dispersion images:
        disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=fmax).numpy().T
        disp, shape1, shape2 = prepare_disp_for_NN(disp)
        # plot them:
        plt.figure()
        plt.imshow(disp[0][:shape1,:shape2], aspect='auto', cmap='jet')
        xticks_positions = np.linspace(0, disp.shape[2] - 1, 5).astype(int)
        xticks_labels = np.round(np.linspace(np.min(c), np.max(c), 5)).astype(int)
        plt.xticks(xticks_positions, xticks_labels)
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.title('Dispersion image')
        plt.show()
        # stop after 1 example:
        if i == 1:
            break


