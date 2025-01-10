from Utilities import JeffLoss, dispersion, prepare_disp_for_NN, acquisition_parameters
import torch
import numpy as np
import matplotlib.pyplot as plt



def evaluate(model=None, test_dataloader=None, device=None,l1param=None,dispersion_arg=False,dataset_name=None,max_label=None):

    #Verify parameters:
    if model is None:
        raise ValueError('model is None')
    if test_dataloader is None:
        raise ValueError('test_dataloader is None')
    if device is None:
        print('device is None, using default device')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if l1param is None:
        print('l1param is None, using default value of 0')
        l1param = 0
    if dispersion_arg is False:
        print('dispersion_arg is False, not using dispersion')
    if dataset_name is None:
        raise ValueError('dataset_name is None')
    if max_label is None:
        print('max_label is None, using default value of 6000')
        max_label = 6000
    print('\nEVALUATION:')
    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    #loss
    criterion = JeffLoss(l1=l1param)

    # Initializations
    all_images = []
    all_predictions = []
    all_labels = []
    all_disp = []

    if dispersion_arg is not False:
        fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name,max_c=max_label)

    # Loop through the test set
    for step in range(len(test_dataloader)):
        sample = test_dataloader.dataset[step]
        inputs = sample['data']
        if dispersion_arg is not False:
            disp = dispersion(inputs[0].T, dt, x, c, epsilon=1e-6, fmax=25).numpy().T
            disp,shape1,shape2 = prepare_disp_for_NN(disp)
            disp = disp.unsqueeze(0).to(device)
        labels = sample['label']
        #print('labels:',labels.shape)
        #inputs= torch.tensor(test_dataloader.dataset.data[step], dtype=torch.float32)
        #labels= torch.tensor(test_dataloader.dataset.labels[step], dtype=torch.float32)
        inputs= inputs.unsqueeze(0)
        inputs, labels = inputs.to(device), labels.to(device)  # Move to device
        all_images.append(inputs.cpu().numpy()) # Transfer images to CPU
        with torch.no_grad():  # No need to calculate gradients
            if dispersion_arg is not False:
                outputs = model(disp)
            else:
                outputs = model(inputs)  # Make predictions
            #calculate error:
            loss = criterion(outputs.float(), labels.permute(1,0).float())
            #print('test loss:',loss.item())
        all_predictions.append(outputs.detach().cpu().numpy())  # Transfer predictions to CPU
        all_labels.append(labels.cpu().numpy().T)  # Transfer labels to CPU
        if dispersion_arg is not False:
            all_disp.append(disp.cpu().numpy())

    # Concatenate all images, predictions, and labels
    all_images = np.concatenate(all_images, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    #mean loss:
    loss_np = loss.cpu().detach().numpy()
    mean_loss = np.mean(loss_np)

    # Display the mean squared error
    print("Loss on Test Set:", mean_loss)

    if dispersion_arg is not False:
        return all_images, all_predictions, all_labels, all_disp, mean_loss, shape1, shape2
    else:
        return all_images, all_predictions, all_labels, None,mean_loss,None,None


def visualize_predictions(all_predictions=None, all_labels=None,all_disp=None, shape1=None, shape2=None, test_dataloader=None, num_samples=5, od=None,max_label=None,dataset_name=None):
    #verify the input parameters:
    if all_disp is None and shape1 is None and shape2 is None:
        print('all_disp, shape1, shape2 are None SO dispersion is not used')
        dispersion_arg = False
    else:
        dispersion_arg = True
    if all_predictions is None:
        raise ValueError('all_predictions is None')
    if all_labels is None:
        raise ValueError('all_labels is None')
    if test_dataloader is None:
        raise ValueError('test_dataloader is None')
    if od is None:
        raise ValueError('od is None')
    if max_label is None:
        print('max_label is None, using default value of 6000')
        max_label = 6000
    if dataset_name is None:
        raise ValueError('dataset_name is None')


    # Calculer Vs_min et Vs_max
    Vs_min = min([np.min(all_labels)* max_label, np.min(all_predictions)* max_label])
    Vs_max = max([np.max(all_labels)* max_label, np.max(all_predictions)* max_label])

    if dispersion_arg is not False:
        all_dispersion = all_disp
        # Paramètres de dispersion
        fmax, dt, src_pos, rec_pos, dg, off0, off1, ng, offmin, offmax, x, c = acquisition_parameters(dataset_name,max_c=max_label)

        if num_samples > len(all_predictions):
            num_samples = len(all_predictions)

        fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(7.5, 3.75 * num_samples))  # Taille ajustée pour rapport

        if num_samples == 1:
            print('You must display at least 2 samples')
        else:
            for i in range(num_samples):
                # Données et étiquettes pour l'échantillon i
                sample = test_dataloader.dataset[i]
                data = sample['data']
                label = sample['label']

                prediction = all_predictions[i]
                disp = all_dispersion[i]

                # Conversion du label en unités réelles
                dz = 0.25
                depth_vector = np.arange(label.shape[0]) * dz

                # Afficher l'image dans la première colonne
                axs[i, 0].imshow(disp[0][0][:shape1, :shape2], aspect='auto')
                xticks_positions = np.linspace(0, disp.shape[3] - 1, 5).astype(int)
                xticks_labels = np.round(np.linspace(np.min(c), np.max(c), 5)).astype(int)
                axs[i, 0].set_xticks(xticks_positions)
                axs[i, 0].set_xticklabels(xticks_labels)
                axs[i, 0].set_title(f'Dispersion image {i}', fontsize=10)  # Taille du titre
                axs[i, 0].set_xlabel('phase velocity (m/s)', fontsize=9)  # Taille de l'étiquette
                axs[i, 0].set_ylabel('frequency (Hz)', fontsize=9)

                # Afficher le label dans la deuxième colonne
                axs[i, 1].plot(label * max_label, depth_vector, label='Label', linewidth=1)
                axs[i, 1].plot(prediction.reshape(-1) * max_label, depth_vector, label='Prédiction', linewidth=1)
                axs[i, 1].invert_yaxis()
                axs[i, 1].set_xlim(Vs_min, Vs_max)
                axs[i, 1].set_xlabel('Vs (m/s)', fontsize=10)
                axs[i, 1].set_ylabel('Depth (m)', fontsize=10)
                axs[i, 1].set_title(f'Vs Depth {i}', fontsize=11)
                axs[i, 1].legend(fontsize=10)  # Taille de la légende

        plt.tight_layout()
        plt.savefig(f'figures/{od}/predictionsVSlabels.pdf', format='pdf', dpi=300)  # Format PDF et 300 dpi
        plt.close()

    else:
        if num_samples > len(all_predictions):
            num_samples = len(all_predictions)

        fig, axs = plt.subplots(nrows=num_samples, ncols=2, figsize=(7.5, 3.75 * num_samples))  # Taille ajustée

        if num_samples == 1:
            print('You must display at least 2 samples')
        else:
            for i in range(num_samples):
                # Données et étiquettes pour l'échantillon i
                sample = test_dataloader.dataset[i]
                data = sample['data']
                label = sample['label']

                prediction = all_predictions[i]

                # Conversion en unités réelles
                time_vector = np.linspace(0, 1.5, data.shape[0])
                nb_traces = 96 if dataset_name == 'Dataset1Dsimple' or dataset_name == 'Dataset1Duge_96tr' else 48
                dz = 0.25
                depth_vector = np.arange(label.shape[0]) * dz

                # Afficher l'image dans la première colonne
                axs[i, 0].imshow(data[0], aspect='auto', cmap='gray',
                                 extent=[0, nb_traces, time_vector[-1], time_vector[0]])
                axs[i, 0].set_title(f'Shot Gather {i}', fontsize=10)
                axs[i, 0].set_xlabel('Distance (m)', fontsize=10)
                axs[i, 0].set_ylabel('Time (sample)', fontsize=10)

                # Afficher le label dans la deuxième colonne
                axs[i, 1].plot(label * max_label, depth_vector, label='Label', linewidth=1)
                axs[i, 1].plot(prediction.reshape(-1) * max_label, depth_vector, label='Prédiction', linewidth=1)
                axs[i, 1].invert_yaxis()
                axs[i, 1].set_xlim(Vs_min, Vs_max)
                axs[i, 1].set_xlabel('Vs (m/s)', fontsize=10)
                axs[i, 1].set_ylabel('Depth (m)', fontsize=10)
                axs[i, 1].set_title(f'Vs Depth {i}', fontsize=11)
                axs[i, 1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f'figures/{od}/predictionsVSlabels.pdf', format='pdf', dpi=300)  # Format PDF et 300 dpi
        plt.close()