#Here you can find the implementation of the traing methods used in this project

import torch
from PhaseShiftMethod import *
import torch.optim as optim
from Utilities import JeffLoss
import matplotlib.pyplot as plt

def training(dataset_name='Dataset1Dhuge_96tr',device=None, lr=0.1, nepochs=1, early_stopping_patience=2,alpha=None,beta=None,model=None, dispersion_arg=False, train_dataloader=None, val_dataloader=None, max_label=None):
    '''
    Train a neural network model using the provided parameters and return the necessary information for plotting learning curves, and the trained model.

    :param dataset_name: name of the dataset used for training, str
    :param device: cpu or gpu used for training, torch.device
    :param lr: learning rate, float
    :param nepochs: maximum number of epochs, int
    :param early_stopping_patience: maximum number of epochs without improvement in validation loss, int
    :param alpha: strenght of the l1 loss, float
    :param beta: strenght of the blocky term, float
    :param model: NN architecture used for training, torch.nn.Module
    :param dispersion_arg: inputs=dispersion image, bool (if False, inputs=shot gathers)
    :param train_dataloader: training dataloader; torch.utils.data.DataLoader
    :param val_dataloader: validation dataloader; torch.utils.data.DataLoader
    :param max_label: maximum velocity value for the labels, int

    :return:
    train_losses: training losses; vector
    val_losses: validation losses; vector
    epochs_count_train: epochs count for training; vector
    epochs_count_val: epochs count for validation; vector
    device: device used for training; torch.device
    model: trained model; torch.nn.Module

    '''
    print('\nTRAINING LOOP:')
    #verify all parameters are correctly defined:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        raise ValueError("Model must be provided.")
    if alpha is None or beta is None:
        alpha = 0;beta = 0
    if train_dataloader is None or val_dataloader is None:
        raise ValueError("Data loaders must be provided.")
    if max_label is None:
        print('If max_label is not defined, it will be set to 6000.')
        max_label = 6000



    print('Device: ', device)
    model = model.to(device)

    # Load the acquisition parameters if the dispersion image is used
    if dispersion_arg is not False:
        fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name,max_c=max_label)

    # Initializations
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f'Optimizer:{optimizer} Learning rate: {lr}')
    print('Early stopping is used with patience:', early_stopping_patience)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    epochs_count_train = []
    epochs_count_val = []

    print(f'Number of epochs: {nepochs}')
    for epoch in range(nepochs):
        epoch_losses = []
        model.train()
        for step in range(len(train_dataloader)):
            sample = train_dataloader.dataset[step]
            inputs = sample['data']
            if dispersion_arg is not False:
                disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=fmax).numpy().T
                disp,_,_ = prepare_disp_for_NN(disp)
                disp = disp.unsqueeze(0).to(device)
            labels = sample['label']
            #print('labels:',labels.shape)
            inputs = inputs.unsqueeze(0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if dispersion_arg is not False:
                outputs = model(disp)
            else:
                outputs = model(inputs)
            criterion = JeffLoss(l1=alpha,beta=beta)
            loss = criterion(outputs.float(), labels.permute(1, 0).float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        #calculate validation loss every 5 epochs, starting from epoch 0
        if epoch % 5 == 0:
            mean_train_loss = np.mean(epoch_losses)
            print(f">>> Epoch {epoch} train loss: ", mean_train_loss)
            train_losses.append(mean_train_loss)
            epochs_count_train.append(epoch)

            # Validation loop
            model.eval()
            val_losses_epoch = []
            with torch.no_grad():
                for step in range(len(val_dataloader)):
                    sample = val_dataloader.dataset[step]
                    inputs = sample['data']
                    if dispersion_arg is not False:
                        disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
                        disp,_,_ = prepare_disp_for_NN(disp)
                        disp = disp.unsqueeze(0).to(device)
                    labels = sample['label']
                    inputs = inputs.unsqueeze(0)
                    inputs, labels = inputs.to(device), labels.to(device)
                    if dispersion_arg is not False:
                        outputs = model(disp)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.permute(1, 0).float())
                    val_losses_epoch.append(loss.item())

            mean_val_loss = np.mean(val_losses_epoch)
            val_losses.append(mean_val_loss)
            epochs_count_val.append(epoch)
            print(f">>> Epoch {epoch} validation loss: ", mean_val_loss)

            # Early stopping check
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_without_improvement = 0
                # Save the best model here if needed
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    return train_losses, val_losses, epochs_count_train, epochs_count_val, device, model



def learning_curves(epochs_count_train=None, train_losses=None, epochs_count_val=None, val_losses=None, od=None):
    '''
    Plot the learning curves for training and validation losses.

    :param epochs_count_train: epochs count for training; vector (training loss is calculated every epoch)
    :param train_losses: training losses; vector
    :param epochs_count_val: epochs count for validations; vector (validation loss is calculated every 5 epochs)
    :param val_losses:  validations losses; vector
    :param od: output direction; str

    :return: plot of learning curves for training and testing datasets; Loss=f(epoch)
    '''
    #verify if all parameters are well defined
    if od is None:
        raise ValueError("Output directory must be provided.")
    if epochs_count_train is None:
        raise ValueError("Training epochs count must be provided.")
    if train_losses is None:
        raise ValueError("Training losses must be provided.")
    if epochs_count_val is None:
        raise ValueError("Validation epochs count must be provided.")
    if val_losses is None:
        raise ValueError("Validation losses must be provided.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7.5, 5))  # Largeur de 7.5 pouces pour s'adapter à une colonne

    # plot the training and validation losses
    ax.plot(epochs_count_train, train_losses, label='Training Loss', linewidth=1.5)
    ax.plot(epochs_count_val, val_losses, label='Validation Loss', linewidth=1.5)

    # Add labels and title
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_yscale('log')
    ax.set_title('Learning Curves', fontsize=12)

    # Add a legend
    ax.legend(fontsize=10)

    # Optimize layout
    plt.tight_layout()

    # Save the figure to the specified directory and close the plot
    plt.savefig(f'figures/{od}/learning_curves.pdf', format='pdf', dpi=300)
    plt.close()