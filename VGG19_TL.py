import os
import glob
import h5py as h5
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from random import random
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import ViTModel
import torch.optim as optim
import argparse
import time
from Utilities import *
from data_augmentation import *
import numpy as np
from sklearn.model_selection import ParameterGrid
import timm
from PhaseShiftMethod import *
import torchvision.models as models
from NN_architectures import PretrainedVGG19
from Training_methods import training,learning_curves
from PytorchDataset import create_dataloaders
from Evaluation import evaluate,visualize_predictions

#define parser
parser = argparse.ArgumentParser(description='ResNet18 with Transfer Learning for MASW')

parser.add_argument('--dataset_name','-data', type=str,default='Dataset1Dsimple', required=False,
                    help='Name of the dataset to use, choose between \n Debug_simple_Dataset \n SimpleDataset \n IntermediateDataset')
parser.add_argument('--nepochs','-ne', type=int, default=1, required=False,help='number of epochs for training')
parser.add_argument('--lr','-lr', type=float, default=0.0005, required=False,help='learning rate')
parser.add_argument('--batch_size','-bs', type=int, default=32, required=False,help='batch size')
parser.add_argument('--output_dir','-od', type=str, default='VGG19_TransferLearning_debug', required=False,help='output directory for figures')
parser.add_argument('--data_augmentation','-aug', type=bool, default=False, required=False,help='data augmentation')
parser.add_argument('--optimizer','-opt', type=str, default='Adam', required=False,help='Optimization strategy to minimize the loss')
parser.add_argument('--GS','-GS', type=bool, default=False, required=False,help='Grid search')
parser.add_argument('--dispersion_image','-disp', type=bool, default=False, required=False, help='Use the dispersion image instead of shotgather')
parser.add_argument('--label_reduction','-label',type=float,default=1,required=False,help='label reduction factor, multiply the size of the vector by this factor')
parser.add_argument('--random_restart','-RR',type=bool,default=False,required=False,help='Random restart')
parser.add_argument('--alpha','-alpha',type=float,default=0.05,required=False,help='strength of L1 regularization for the loss')




args = parser.parse_args()

# Paths
data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
dataset_name = args.dataset_name
files_path = os.path.join(data_path, dataset_name)

train_folder = glob.glob(f'{files_path}/train/*')
validate_folder = glob.glob(f'{files_path}/validate/*')
test_folder = glob.glob(f'{files_path}/test/*')

setup_directories(name=args.output_dir)

# Verify the dataset
print('VERIFYING DATASET')
bad_files = verify([train_folder,validate_folder,test_folder])

#create pytorch dataloaders
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
    data_path=data_path,
    dataset_name=dataset_name,
    batch_size=args.batch_size,
    use_dispersion=args.dispersion_image,
    data_augmentation=args.data_augmentation
)

#%%
#Data processing
# Verify if data is normalized
print('check a random image to verify if data is normalized:')
print('min:',train_dataloader.dataset.data[0].min(),' max:', train_dataloader.dataset.data[0].max())

# Call the function with your dataloaders
train_dataloader, val_dataloader, test_dataloader, max_label = normalize_labels(
    train_dataloader, val_dataloader, test_dataloader
)

#check a random label to verify if data is normalized
print('check a random label to verify if data is normalized:')
print('min:',train_dataloader.dataset.labels[0].min(),' max:', train_dataloader.dataset.labels[0].max())
#check the shape of the image
print('check the shape of the image:',train_dataloader.dataset.data[0].shape)


plot_random_samples(train_dataloader=train_dataloader, num_samples=5,od=args.output_dir,max_label=max_label)

out_dim=len(train_dataloader.dataset.labels[0])
print('Output dimension: ',out_dim)


# Display the model architecture
print('MODEL ARCHITECTURE:')
model = PretrainedVGG19(out_dim=out_dim)
print(model)
#print the amount of unfrozen parameters:
unfrozen_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of unfrozen parameters:',unfrozen_parameters)

def random_restart_training(device=None, lr=args.lr, nepochs=1000,
             early_stopping_patience=2, alpha=args.alpha):
    print('\nTRAINING LOOP:')
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Device: ', device)
    model = PretrainedVGG19().to(device)

    # Load the acquisition parameters if the dispersion image is used
    if args.dispersion_image:
        fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name,max_c=max_label)

    # Initializations
    optimizer = {
        'Adam': optim.Adam(model.parameters(), lr=lr),
        'Nadam': optim.NAdam(model.parameters(), lr=lr),
        'RMSprop': optim.RMSprop(model.parameters(), lr=lr)
    }.get(args.optimizer, optim.Adam(model.parameters(), lr=lr))

    print(f'Optimizer: {optimizer} Learning rate: {lr}')

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None  # To save the best model

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
            if args.dispersion_image:
                disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
                disp, _, _ = prepare_disp_for_NN(disp)
                disp = disp.unsqueeze(0).to(device)
            labels = sample['label']
            inputs = inputs.unsqueeze(0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(disp if args.dispersion_image else inputs)
            criterion = JeffLoss(l1=alpha)
            loss = criterion(outputs.float(), labels.permute(1, 0).float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

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
                    if args.dispersion_image:
                        disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
                        disp, _, _ = prepare_disp_for_NN(disp)
                        disp = disp.unsqueeze(0).to(device)
                    labels = sample['label']
                    inputs = inputs.unsqueeze(0)
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(disp if args.dispersion_image else inputs)
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
                best_model_state = model.state_dict()  # Save the best model state
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}, restarting training...")
                    # Restart model and optimizer when early stopping patience is exceeded
                    model = PretrainedVGG19().to(device)  # Reinitialize model
                    optimizer = {
                        'Adam': optim.Adam(model.parameters(), lr=lr),
                        'Nadam': optim.NAdam(model.parameters(), lr=lr),
                        'RMSprop': optim.RMSprop(model.parameters(), lr=lr)
                    }.get(args.optimizer, optim.Adam(model.parameters(), lr=lr))
                    epochs_without_improvement = 0  # Reset the counter for early stopping

    # Load the best model state after training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print('Best loss on validation set after Early stopping:', best_val_loss)

    return train_losses, val_losses, epochs_count_train, epochs_count_val, device, model


#if GS = True
if args.GS == True:
    # Training
    # Définir la grille des paramètres
    param_grid = {
        'alpha': [0.02, 0.03],
        'beta': [0.05, 0.1, 0.15],
        'l1': [0.01, 0.05],
        'v_max': [0.1, 0.2, 0.3]
    }

    # Grid search
    best_params = None
    best_val_loss = float('inf')
    best_time = 0
    final_val_losses = []
    final_train_losses = []
    final_epochs_count_train = 0
    final_epochs_count_val = 0

    # init for total time
    t00 = time.time()

    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        alpha = params['alpha']
        beta = params['beta']
        l1param = params['l1']
        vmaxparam = params['v_max']

        # Appel à la fonction de formation
        t0 = time.time()
        train_losses, val_losses, epochs_count_train, epochs_count_val, device, model = training(
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args.lr,
            nepochs=args.nepochs,
        )
        training_time = time.time() - t0

        # Évaluer la perte de validation et mettre à jour les meilleurs paramètres si nécessaire
        final_val_loss = val_losses[-1] if val_losses else float('inf')
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            final_val_losses = val_losses
            final_train_losses = train_losses
            best_params = params
            best_time = training_time
            final_epochs_count_train = epochs_count_train
            final_epochs_count_val = epochs_count_val

    # total time
    total_time = time.time() - t00
    print(f"Best parameters: {best_params} with validation loss: {best_val_loss}")
    print('\nTraining time best model:', training_time)
    print('\nTotal Grid search time:', total_time)

else:
    #use default loss parameters
    # Training
    t0 = time.time()
    if args.random_restart:
        train_losses, val_losses, epochs_count_train, epochs_count_val, device, model = random_restart_training(
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args.lr,
            nepochs=args.nepochs,
            alpha=args.alpha
        )
    else:
        train_losses, val_losses, epochs_count_train, epochs_count_val, device, model = training(
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args.lr,
            nepochs=args.nepochs,
            alpha=args.alpha,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_label=max_label,
            model=model,
            dispersion_arg=args.dispersion_image
        )
    best_time = time.time() - t0
    best_params = {'alpha': 0.00, 'beta': 0.0, 'l1': args.alpha, 'v_max': 0.0}
    final_val_losses = val_losses
    final_train_losses = train_losses
    final_epochs_count_train = epochs_count_train
    final_epochs_count_val = epochs_count_val

# Display learning curves
learning_curves(final_epochs_count_train, final_train_losses, final_epochs_count_val, final_val_losses,args.output_dir)

#evaluate
all_images, all_predictions, all_labels, all_disp,mean_loss,shape1,shape2= evaluate(model, test_dataloader, device,dispersion_arg=args.dispersion_image,dataset_name=args.dataset_name,max_label=max_label)

# Display some predictions
visualize_predictions(all_predictions=all_predictions, all_labels=all_labels,all_disp=all_disp, shape1=shape1, shape2=shape2, test_dataloader=test_dataloader, num_samples=5, od=args.output_dir,max_label=max_label,dataset_name=args.dataset_name)

#save runs infos
#display on a signgle image all informations about the current model
main_path= os.path.abspath(__file__)
display_run_info(model=model,od=args.output_dir,args=args,metrics=mean_loss,training_time=best_time,main_path=main_path,best_params=best_params,nb_param=unfrozen_parameters)

#save the model parameters
torch.save(model.state_dict(), f'figures/{args.output_dir}/model.pth')