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
from PytorchDataset import create_dataloaders
from NN_architectures import PretrainedPVTv2, PretrainedPVTv2QLoRA
from Training_methods import training,learning_curves
from Evaluation import evaluate,visualize_predictions

print('GPU availables:',torch.cuda.device_count(),'\n')  # Devrait retourner le nombre de GPU disponibles
print('Torch version is :',torch.__version__,' and Cuda version is :',torch.version.cuda,'\n')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print('Using GPU:',os.environ["CUDA_VISIBLE_DEVICES"],'\n')

#define parser
parser = argparse.ArgumentParser(description='PVT with Transfer Learning for MASW')

parser.add_argument('--dataset_name','-data', type=str,default='Halton_debug', required=False,
                    help='Name of the dataset to use, choose between \n Debug_simple_Dataset \n SimpleDataset \n IntermediateDataset')
parser.add_argument('--nepochs','-ne', type=int, default=1, required=False,help='number of epochs for training')
parser.add_argument('--lr','-lr', type=float, default=0.0005, required=False,help='learning rate')
parser.add_argument('--batch_size','-bs', type=int, default=64, required=False,help='batch size')
parser.add_argument('--output_dir','-od', type=str, default='PVT_TransferLearning_debug', required=False,help='output directory for figures')
parser.add_argument('--data_augmentation','-aug', type=bool, default=False, required=False,help='data augmentation')
parser.add_argument('--optimizer','-opt', type=str, default='Adam', required=False,help='Optimization strategy to minimize the loss')
parser.add_argument('--GS','-GS', type=bool, default=False, required=False,help='Grid search')
parser.add_argument('--dispersion_image','-disp', type=bool, default=False, required=False, help='Use the dispersion image instead of shotgather')
parser.add_argument('--label_reduction','-label',type=float,default=1,required=False,help='label reduction factor, multiply the size of the vector by this factor')
parser.add_argument('--random_restart','-RR',type=bool,default=False,required=False,help='Random restart')
parser.add_argument('--alpha','-alpha',type=float,default=0.0,required=False,help='strength of L1 regularization for the loss')


args = parser.parse_args()

# Paths
#data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_Geoflow/GeoFlow/Tutorial/Datasets/'
data_path = '/home/rbertille/data/pycharm/ViT_project/pycharm_ViT/DatasetGeneration/Datasets/'
dataset_name = args.dataset_name
files_path = os.path.join(data_path, dataset_name)

train_folder = glob.glob(f'{files_path}/train/*')
validate_folder = glob.glob(f'{files_path}/validate/*')
test_folder = glob.glob(f'{files_path}/test/*')

setup_directories(name=args.output_dir)

# Verify the dataset
#print('VERIFYING DATASET')
#bad_files = verify([train_folder,validate_folder,test_folder])

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
train_dataloader, val_dataloader, test_dataloader, max_labelVP,max_labelVS = normalize_labels(
    train_dataloader, val_dataloader, test_dataloader
)

#check a random label to verify if VP is normalized:
print('check a random label to verify if VP is normalized:')
print('min:',train_dataloader.dataset.labelsVP[0].min(),' max:', train_dataloader.dataset.labelsVP[0].max())

#check a random label to verify if Vs is normalized
print('check a random label to verify if Vs is normalized:')
print('min:',train_dataloader.dataset.labelsVS[0].min(),' max:', train_dataloader.dataset.labelsVS[0].max())

plot_random_samples(train_dataloader=train_dataloader, num_samples=5,od=args.output_dir,max_labelVS=max_labelVS)


out_dim=len(train_dataloader.dataset.labelsVS[0])
print('Output dimension: ',out_dim)

# Display the model architecture
print('MODEL ARCHITECTURE:')
model = PretrainedPVTv2QLoRA(out_dim=out_dim)
print(model)
#print the amount of unfrozen parameters:
unfrozen_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of unfrozen parameters:',unfrozen_parameters)


t0 = time.time()
train_losses, val_losses, epochs_count_train, epochs_count_val, device, model = training(
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args.lr,
            nepochs=args.nepochs,
            alpha=args.alpha,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_labelVS=max_labelVS,
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

all_images, all_predictionsVS, all_labelsVS, all_disp,mean_loss,shape1,shape2= evaluate(model, test_dataloader, device,dispersion_arg=args.dispersion_image,dataset_name=args.dataset_name,max_labelVS=max_labelVS)


#%%
# Display some predictions
visualize_predictions(all_predictionsVS=all_predictionsVS, all_labelsVS=all_labelsVS,all_disp=all_disp, shape1=shape1, shape2=shape2, test_dataloader=test_dataloader, num_samples=5, od=args.output_dir,max_labelVS=max_labelVS,dataset_name=args.dataset_name)

#save runs infos
#display on a signgle image all informations about the current model
main_path= os.path.abspath(__file__)
display_run_info(model=model,od=args.output_dir,args=args,metrics=mean_loss,training_time=best_time,main_path=main_path,best_params=best_params,nb_param=unfrozen_parameters)

#save the model parameters
torch.save(model.state_dict(), f'figures/{args.output_dir}/model.pth')