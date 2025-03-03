#Here you can find the implementation of the traing methods used in this project

import torch
from PhaseShiftMethod import *
import torch.optim as optim
from Utilities import JeffLoss
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model


def apply_qlora(model):
    """Apply QLoRA to the model by quantizing layers and enabling LoRA on select layers."""
    config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.05,  # Dropout rate
        target_modules=["fc1", "fc2"],  # Targeted layers for LoRA
        bias="none",
    )
    model = get_peft_model(model, config)
    return model


def training(
        dataset_name='Halton_Dataset',
        device=None, lr=0.1, nepochs=1, early_stopping_patience=2,
        alpha=None, beta=None, model=None, dispersion_arg=False,
        train_dataloader=None, val_dataloader=None, max_labelVS=None):
    """Train a neural network model with QLoRA."""
    print('\nTRAINING LOOP:')
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        raise ValueError("Model must be provided.")
    if alpha is None or beta is None:
        alpha, beta = 0, 0
    if train_dataloader is None or val_dataloader is None:
        raise ValueError("Data loaders must be provided.")
    if max_labelVS is None:
        print('If max_labelVS is not defined, it will be set to 2000.')
        max_labelVS = 2000

    print('Device:', device)
    model = apply_qlora(model).to(device)  # Appliquer QLoRA sur le modèle
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f'Optimizer: {optimizer}, Learning rate: {lr}')
    print('Early stopping patience:', early_stopping_patience)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    epochs_count_train, epochs_count_val = [], []

    for epoch in range(nepochs):
        epoch_losses = []
        model.train()
        for step, sample in enumerate(train_dataloader):
            inputs = sample['data'].to(device)
            #print('inputs shape:', inputs.shape)
            labelsVS = sample['label_VS'].to(device)
            #print('labelsVS shape:', labelsVS.unsqueeze(2).shape)

            optimizer.zero_grad()

            # plot input for verification:
            image=inputs[0][0]
            image=image.cpu().detach().numpy()
            print('[training] verification: input shape:', image.shape)
            plt.imshow(image, aspect='auto', cmap='gray')
            plt.show()
            plt.close()


            outputs = model(inputs)
            #print('outputs shape:', outputs.shape)
            criterion = JeffLoss(l1=alpha, beta=beta)  # Vérifie la définition de cette fonction
            loss = criterion(outputs.float(), labelsVS.squeeze(2).float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % 5 == 0:
            mean_train_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch} train loss: {mean_train_loss}")
            train_losses.append(mean_train_loss)
            epochs_count_train.append(epoch)

            model.eval()
            val_losses_epoch = []
            with torch.no_grad():
                for step, sample in enumerate(val_dataloader):
                    inputs = sample['data'].to(device)
                    labelsVS = sample['label_VS'].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labelsVS.squeeze(2).float())
                    val_losses_epoch.append(loss.item())

            mean_val_loss = sum(val_losses_epoch) / len(val_losses_epoch)
            val_losses.append(mean_val_loss)
            epochs_count_val.append(epoch)
            #print(f"Epoch {epoch} validation loss: {mean_val_loss}")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_without_improvement = 0
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