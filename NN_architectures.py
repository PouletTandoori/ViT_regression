# Here you can find the implementation of all the neural networks architectures tested, using Pytorch
import torch
import torch.nn as nn
from torchvision import models
import timm
from transformers import ViTModel

class PretrainedResNet18(nn.Module):
    def __init__(self, out_dim=200):
        super(PretrainedResNet18, self).__init__()

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Sortie : (64, 750, 96)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (64, 375, 48)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Sortie : (128, 375, 48)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (128, 187, 24)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Sortie : (256, 187, 24)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (256, 93, 12)

            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),  # Sortie : (3, 93, 12)
            nn.BatchNorm2d(3),
            nn.ReLU(),

            nn.Upsample(size=(224, 224), mode="bilinear")  # Sortie : (3, 224, 224)
        )

        # Charger le modèle pré-entraîné ResNet18
        self.resnet18 = models.resnet18(pretrained=True)
        emb_dim = self.resnet18.fc.in_features  # Taille des embeddings du modèle pré-entraîné

        # Geler les poids du modèle ResNet18
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Dégeler les poids de la dernière couche
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

        # Ajouter une tête personnalisée pour ta tâche
        self.head = nn.Sequential(
            nn.LayerNorm(1000),  # Changer la normalisation pour s'adapter à la sortie de ResNet18
            nn.Linear(1000, 256),  # Couche linéaire intermédiaire
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_dim)  # Couche de sortie
        )

    def forward(self, img):
        if img.shape[-2:] == (750, 96):
            img = self.conv_adapter(img)

        # Obtenir les embeddings à partir de ResNet18
        x = self.resnet18(img)

        # Utiliser les embeddings pour faire des prédictions
        output = self.head(x)

        return output

class PretrainedVGG19(nn.Module):
    def __init__(self, out_dim=200):
        super(PretrainedVGG19, self).__init__()

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Sortie : (64, 750, 96)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (64, 375, 48)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Sortie : (128, 375, 48)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (128, 187, 24)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Sortie : (256, 187, 24)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (256, 93, 12)

            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),  # Sortie : (3, 93, 12)
            nn.BatchNorm2d(3),
            nn.ReLU(),

            nn.Upsample(size=(224, 224), mode="bilinear")  # Sortie : (3, 224, 224)
        )

        # Charger le modèle pré-entraîné ResNet18
        self.VGG19 = models.vgg19(pretrained=True)

        # Geler les poids du modèle pré-entraîné
        for param in self.VGG19.parameters():
            param.requires_grad = False

        # Dégeler les poids de la dernière couche
        for param in self.VGG19.classifier[6].parameters():
            param.requires_grad = True

        # Ajouter une tête personnalisée pour ta tâche
        self.head = nn.Sequential(
            nn.LayerNorm(1000),  # Changer la normalisation pour s'adapter à la sortie de VGG19
            nn.Linear(1000, 256),  # Couche linéaire intermédiaire
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_dim)  # Couche de sortie
        )

    def forward(self, x):
        if x.shape[-2:] == (750, 96):  # Si l'entrée est (3, 750, 96)
            x = self.conv_adapter(x)  # Appliquer l'adaptation des dimensions
        x = self.VGG19(x)
        x = self.head(x)
        return x

class PretrainedPVTv2(nn.Module):
    def __init__(self, out_dim=200, pretrained_model_name='pvt_v2_b2'):
        super(PretrainedPVTv2, self).__init__()

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Sortie : (64, 750, 96)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (64, 375, 48)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Sortie : (128, 375, 48)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (128, 187, 24)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Sortie : (256, 187, 24)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (256, 93, 12)

            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),  # Sortie : (3, 93, 12)
            nn.BatchNorm2d(3),
            nn.ReLU(),

            nn.Upsample(size=(224, 224), mode="bilinear")  # Sortie : (3, 224, 224)
        )

        # Charger le modèle pré-entraîné PVTv2
        self.pvt = timm.create_model(pretrained_model_name, pretrained=True)
        emb_dim = self.pvt.num_features  # Taille des embeddings du modèle pré-entraîné

        # Geler les poids du modèle PVTv2
        for param in self.pvt.parameters():
            param.requires_grad = False

        # Dégeler les poids du dernier stage (dernier bloc du dernier stage)
        for param in self.pvt.stages[-1].blocks[-1].parameters():
            param.requires_grad = True

        # Ajouter une opération de global pooling pour obtenir [batch_size, emb_dim, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Ajouter une tête personnalisée pour ta tâche
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),  # Normalisation
            nn.Linear(emb_dim, 256),  # Couche linéaire intermédiaire
            nn.ReLU(),  # Activation non-linéaire
            nn.Dropout(0.5),  # Dropout pour la régularisation
            nn.Linear(256, out_dim)  # Couche de sortie
        )

    def forward(self, img):
        if img.shape[-2:] == (750, 96):
            img = self.conv_adapter(img)

        # Obtenir les embeddings à partir de PVTv2
        x = self.pvt.forward_features(img)

        # Réduire les dimensions spatiales avec un global pooling
        x = self.global_pool(x)

        # Aplatir la sortie de la forme [batch_size, emb_dim, height, width] à [batch_size, emb_dim]
        x = torch.flatten(x, start_dim=1)  # Aplatir les dimensions height et width

        # Utiliser les embeddings pour faire des prédictions
        output = self.head(x)

        return output

class PretrainedViT(nn.Module):
    def __init__(self, out_dim=200, pretrained_model_name='google/vit-base-patch32-224-in21k'):
        super(PretrainedViT, self).__init__()

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Sortie : (64, 750, 96)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (64, 375, 48)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Sortie : (128, 375, 48)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (128, 187, 24)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Sortie : (256, 187, 24)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Sortie : (256, 93, 12)

            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),  # Sortie : (3, 93, 12)
            nn.BatchNorm2d(3),
            nn.ReLU(),

            nn.Upsample(size=(224, 224), mode="bilinear")  # Sortie : (3, 224, 224)
        )

        # Load the pretrained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        emb_dim = self.vit.config.hidden_size  # Taille des embeddings du modèle pré-entraîné

        # Freeze the weights of the ViT model
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze the weights of the last layer of the ViT model
        for param in self.vit.encoder.layer[-1].parameters():
            param.requires_grad = True


        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),  # Normalization
            nn.Linear(emb_dim, 256),  # Intermediar linear layer
            nn.ReLU(),  # Activation non-linéaire
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, out_dim)  # Output layer
        )

    def forward(self, img):
        if img.shape[-2:] == (750, 96):
            img = self.conv_adapter(img)

        # Obtain embeddings
        outputs = self.vit(pixel_values=img)

        # Extract the embedding of the [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Use the embedding to make regression predictions
        output = self.head(cls_embedding)

        return output