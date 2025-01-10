import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from PhaseShiftMethod import *
from NN_architectures import PretrainedViT
from Training_methods import training,learning_curves
from PytorchDataset import create_dataloaders
from Evaluation import evaluate,visualize_predictions

#define parser
parser = argparse.ArgumentParser(description='ViT with Transfer Learning for MASW')

parser.add_argument('--dataset_name','-data', type=str,default='Dataset1Dsimple', required=False,
                    help='Name of the dataset to use, choose between \n Debug_simple_Dataset \n SimpleDataset \n IntermediateDataset')
parser.add_argument('--nepochs','-ne', type=int, default=1, required=False,help='number of epochs for training')
parser.add_argument('--lr','-lr', type=float, default=0.0005, required=False,help='learning rate')
parser.add_argument('--batch_size','-bs', type=int, default=1, required=False,help='batch size')
parser.add_argument('--output_dir','-od', type=str, default='ViT_TransferLearning_debug', required=False,help='output directory for figures')
parser.add_argument('--data_augmentation','-aug', type=bool, default=False, required=False,help='data augmentation')
parser.add_argument('--decay','-dec', type=float, default=0, required=False,help='weight decay')
parser.add_argument('--GS','-GS', type=bool, default=False, required=False,help='Grid search')
parser.add_argument('--dispersion_image','-disp', type=bool, default=False, required=False, help='Use the dispersion image instead of shotgather')
parser.add_argument('--label_reduction','-label',type=float,default=1,required=False,help='label reduction factor, multiply the size of the vector by this factor')
parser.add_argument('--alpha','-alpha',type=float,default=0.00,required=False,help='strength of L1 regularization for the loss')
parser.add_argument('--blocky_inversion','-blocky',type=float,default=0,required=False,help='parameter to control the strength of blocky term in loss')



args = parser.parse_args()

print('Arguments:',args)

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

plot_random_samples(train_dataloader=train_dataloader, num_samples=5,od=args.output_dir,max_label=max_label)

out_dim=len(train_dataloader.dataset.labels[0])
print('Output dimension: ',out_dim)

# Display the model architecture
print('MODEL ARCHITECTURE:')
model = PretrainedViT(out_dim=out_dim)
print(model)
#print the amount of unfrozen parameters:
unfrozen_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of unfrozen parameters:',unfrozen_parameters)

#if GS = True
if args.GS == True:
    # Training
    # Définir la grille des paramètres tel que (1- alpha -beta toujours supérieur ou égal à 0.1):
    alphas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    betas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


    param_grid = {
        'alpha': alphas,
        'beta': betas
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
        if params['alpha'] + params['beta'] + 0.1 > 1:
            print('Parameters not valid, sum must be less than 1')
            #test anothers parameters directly
            continue
        alpha = params['alpha']
        beta = params['beta']

        # Appel à la fonction de formation
        t0 = time.time()
        train_losses, val_losses, epochs_count_train, epochs_count_val, device, model = training(
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args.lr,
            nepochs=args.nepochs,
            alpha=alpha,
            beta=beta
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
    best_params = { 'l1': args.alpha, 'beta': args.blocky_inversion}
    final_val_losses = val_losses
    final_train_losses = train_losses
    final_epochs_count_train = epochs_count_train
    final_epochs_count_val = epochs_count_val

# Display learning curves
learning_curves(final_epochs_count_train, final_train_losses, final_epochs_count_val, final_val_losses,args.output_dir)

all_images, all_predictions, all_labels, all_disp,mean_loss,shape1,shape2= evaluate(model, test_dataloader, device,dispersion_arg=args.dispersion_image,dataset_name=args.dataset_name,max_label=max_label)


#%%
# Display some predictions
visualize_predictions(all_predictions=all_predictions, all_labels=all_labels,all_disp=all_disp, shape1=shape1, shape2=shape2, test_dataloader=test_dataloader, num_samples=5, od=args.output_dir,max_label=max_label,dataset_name=args.dataset_name)

#save runs infos
#display on a signgle image all informations about the current model
main_path= os.path.abspath(__file__)
display_run_info(model=model,od=args.output_dir,args=args,metrics=mean_loss,training_time=best_time,main_path=main_path,best_params=best_params,nb_param=unfrozen_parameters)


#save the model parameters
torch.save(model.state_dict(), f'figures/{args.output_dir}/model.pth')