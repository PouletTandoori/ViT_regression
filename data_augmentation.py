import torch
import random
import numpy as np

class DeadTraces:
    """
    Applies dead traces to a shotgather. A dead trace is a trace with all values set to 1.
    """

    def __init__(self, dead_trace_ratio=0.04):
        self.dead_trace_ratio = dead_trace_ratio

    def add_dead_traces(self, image):
        '''
        Replace some random traces with dead traces in the data:
        a dead trace is a trace (=column) with all values set to 1.

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels,
                      H is the height, and W is the width.

        :return: augmented_data: 3D tensor of shape (C, H, W) with some dead traces.
        '''

        # Ensure the image has three dimensions (C, H, W)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension if missing

        # Make a copy of the original image
        augmented_data = image.clone()

        # Count columns (traces) in the data
        num_columns = augmented_data.shape[2]
        num_dead_traces = int(round(num_columns * self.dead_trace_ratio))

        # Choose the indices of the dead traces randomly
        dead_traces_indices = random.sample(range(num_columns), num_dead_traces)

        # Set the values of the dead traces to 1
        for i in dead_traces_indices:
            augmented_data[:, :, i] = 1

        return augmented_data

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Call the add_dead_traces method using the dead_trace_ratio
        return self.add_dead_traces(sample)

class MissingTraces:
    """
    Applies missing traces to a shotgather. We establish a missing trace is a trace with all values set to the average value of the trace.
    """
    def __init__(self, missing_trace_ratio=0.04):
        self.missing_trace_ratio = missing_trace_ratio

    def Missing_traces(self,image, missing_trace_ratio):
        ''''
        Replace some random traces by missing traces to the data: a missing trace is a trace (=column) with all values set to the average value of the trace

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels, H is the height and W is the width
        :param missing_trace_ratio: From 0 to 1, the ratio of missing traces to add

        :return: augmented_data: 3D tensor of shape (C, H, W) with some missing traces
        '''

        img2D = 0
        # verify if image is (h,w) or (c,h,w):
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            img2D = 1

        # Make a copy of the original image
        augmented_data = image.clone() if isinstance(image, torch.Tensor) else image.copy()

        # count colums in the data
        num_columns = augmented_data[0].shape[1]
        # print('nb traces=',num_columns)
        # choose missing_trace_ratio % of the traces to be dead traces randomly
        num_missing_traces = int(num_columns * missing_trace_ratio)
        # print('nb missing traces',num_dead_traces)
        # choose the indices of the dead traces
        missing_traces_indices = random.sample(range(num_columns), num_missing_traces)
        # print('indices of the missing traces:',missing_traces_indices)

        average_value = augmented_data[:, :, missing_traces_indices].mean()

        # set the values of the dead traces to average
        augmented_data[:, :, missing_traces_indices] = average_value
        if img2D==1:
            augmented_data=augmented_data.squeeze(0)

        return augmented_data

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.Missing_traces(sample, self.missing_trace_ratio)

class GaussianNoise:
    """
    Adds Gaussian noise to a shotgather.
    """
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean

    def add_gaussian_noise(self, image, mean, std):
        '''
        Add Gaussian noise to the data

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels, H is the height and W is the width
        :param mean: mean of the Gaussian noise
        :param std: standard deviation of the Gaussian noise

        :return: augmented_data: 3D tensor of shape (C, H, W) with Gaussian noise
        '''
        # Make a copy of the original image
        augmented_data = image.clone() if isinstance(image, torch.Tensor) else image.copy()

        # Add Gaussian noise
        noise = torch.randn(augmented_data.shape) * std + mean
        augmented_data += noise

        return augmented_data

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.add_gaussian_noise(sample, self.mean, self.std)

class TraceShift:
    """
    Shifts traces in a shot to simulate topography effects.
    """

    def __init__(self, shift_ratio=0.1):
        self.shift_ratio = shift_ratio

    def shift_traces(self, image, shift_ratio):
        '''
        Shift traces (columns) in the data by a random integer value between
        -max_shift and +max_shift to create topography effects.
        If empty spaces are created, they are filled with the average value of the trace.

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels,
                      H is the height, and W is the width.
        :param shift_ratio: From 0 to 1, the ratio of traces to shift.

        :return: augmented_data: 3D tensor of shape (C, H, W) with some shifted traces.
        '''

        # Verify if image is (H, W) or (C, H, W):
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension if missing

        # Make a copy of the original image
        augmented_data = image.clone()

        # Count columns in the data
        num_columns = augmented_data.shape[2]
        num_rows = augmented_data.shape[1]
        max_shift = int(shift_ratio * num_rows)

        # Loop over traces, apply a random shift for each trace:
        for i in range(num_columns):
            # Get the trace
            trace = augmented_data[:, :, i]
            # Get the shift value (random integer between -max_shift and max_shift)
            shift = random.randint(-max_shift, max_shift)

            # Shift the trace
            shifted_trace = torch.roll(trace, shift, dims=1)  # Shift vertically
            # Fill empty spaces with the average value of the trace
            if shift > 0:
                shifted_trace[:, :shift] = trace[:, :shift].mean()
            elif shift < 0:
                shifted_trace[:, shift:] = trace[:, shift:].mean()

            augmented_data[:, :, i] = shifted_trace

        return augmented_data

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.shift_traces(sample, self.shift_ratio)

class TraceShift2:
    """
    Shifts a subset of contiguous traces in a shot to simulate topography effects.
    """

    def __init__(self, shift_ratio=0.1, contiguous_ratio=0.2):
        """
        :param shift_ratio: Ratio of traces to shift (from 0 to 1).
        :param contiguous_ratio: Ratio of contiguous traces to shift (from 0 to 1).
        """
        self.shift_ratio = shift_ratio
        self.contiguous_ratio = contiguous_ratio

    def create_grouped_vector(self,length=100, num_groups=None, group_size_range=(5, 15)):
        """
        Crée un vecteur de 0 et de 1 avec des 1 regroupés en 2 à 4 groupes.

        :param length: Longueur du vecteur.
        :param num_groups: Nombre de regroupements de 1 (2 à 4).
        :param group_size_range: Tuple définissant la taille minimale et maximale des groupes.

        :return: Vecteur numpy de 0 et 1 avec les 1 regroupés.
        """
        if num_groups is None:
            num_groups = random.choice([2,3,4])  # Choisit aléatoirement 2 à 4 groupes

        vector = np.zeros(length, dtype=int)

        group_positions = sorted(
            random.sample(range(length), num_groups))  # Choisit aléatoirement les positions de départ des groupes

        for pos in group_positions:
            group_size = random.randint(*group_size_range)  # Taille aléatoire pour chaque groupe
            end_pos = min(pos + group_size, length)  # S'assure que le groupe reste dans les limites du vecteur
            vector[pos:end_pos] = 1

        return vector

    def shift_traces(self, image, shift_ratio, contiguous_ratio):
        '''
        Shift a subset of contiguous traces in the data by a random integer value between
        -max_shift and +max_shift to create topography effects.
        If empty spaces are created, they are filled with the average value of the trace.

        :param image: 3D tensor of shape (C, H, W) where C is the number of channels,
                      H is the height, and W is the width.
        :param shift_ratio: From 0 to 1, the ratio of traces to shift.
        :param contiguous_ratio: From 0 to 1, the ratio of contiguous traces to shift.

        :return: augmented_data: 3D tensor of shape (C, H, W) with some shifted traces.
        '''

        # Verify if image is (H, W) or (C, H, W):
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension if missing

        # Make a copy of the original image
        augmented_data = image.clone()

        # Count columns in the data
        num_columns = augmented_data.shape[2]
        num_rows = augmented_data.shape[1]
        max_shift = int(shift_ratio * num_rows)

        #determine the size of the sequences to shift
        group_size_range = (int(contiguous_ratio * num_columns/2), int(num_columns*contiguous_ratio*2))

        # create a vector to determine which column is shifted or not (1=shifted, 0=not shifted)
        shift_indices=self.create_grouped_vector(length=num_columns, num_groups=None, group_size_range=group_size_range)

        # Loop over traces, apply a random shift for each trace:
        for i in range(num_columns):
            if shift_indices[i]==0:
                continue
            else:
                # Get the trace
                trace = augmented_data[:, :, i]
                # Get the shift value (random integer between -max_shift and max_shift)
                direction_shift= random.choice([-1,1])
                if direction_shift==1:
                    shift = random.randint(0, max_shift)
                else:
                    shift = random.randint(-max_shift,0)

                # Shift the trace
                shifted_trace = torch.roll(trace, shift, dims=1)  # Shift vertically
                # Fill empty spaces with the average value of the trace
                if shift > 0:
                    shifted_trace[:, :shift] = trace[:, :shift].mean()
                elif shift < 0:
                    shifted_trace[:, shift:] = trace[:, shift:].mean()

                augmented_data[:, :, i] = shifted_trace

        return augmented_data



    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.shift_traces(sample, self.shift_ratio, self.contiguous_ratio)

