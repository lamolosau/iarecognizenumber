import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Couche de convolution 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Couche de convolution 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Couche de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Couches fully connected (après aplatissage)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes pour les chiffres de 0 à 9

    def forward(self, x):
        # Propagation avant dans le réseau
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # Aplatir les données pour les couches fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prépare les données d'entraînement (MNIST dataset par exemple)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Obtiens le chemin absolu du dossier courant
current_directory = os.path.dirname(os.path.abspath(__file__))

# Utilise ce chemin absolu pour le dossier 'data'
data_directory = os.path.join(current_directory, 'data')

# Assure-toi que le dossier 'data' existe, sinon crée-le
os.makedirs(data_directory, exist_ok=True)

# Utilise le chemin absolu pour le téléchargement des données MNIST
train_dataset = MNIST(root=data_directory, train=True, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialise le modèle, la fonction de coût et l'optimiseur
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Utilise un planificateur de taux d'apprentissage (learning rate scheduler)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

validation_dataset = MNIST(root=data_directory, train=False, transform=transform, download=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=False)

# Augmente le nombre d'époques
for epoch in range(30):
    # Entraînement
    model.train()
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Évaluation sur l'ensemble de validation
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for val_data in validation_loader:
            val_inputs, val_labels = val_data
            val_outputs = model(val_inputs)
            _, predicted = torch.max(val_outputs, 1)
            total_samples += val_labels.size(0)
            total_correct += (predicted == val_labels).sum().item()

    accuracy = total_correct / total_samples
    print(f"Epoch {epoch+1}/{30}, Accuracy on Validation Set: {accuracy * 100:.2f}%")

    # Ajuste le taux d'apprentissage avec le scheduler
    scheduler.step()

# Une fois le modèle entraîné, tu peux l'utiliser pour faire des prédictions
# Charger une image de test
image_path = "img/six.jpg"
# Convertir en chemin absolu si nécessaire
image_path = os.path.abspath(image_path)

# Vérifier si le fichier existe
if os.path.exists(image_path):
    # Charger l'image
    image = Image.open(image_path)
    # Appliquer des transformations pour prétraiter l'image
    preprocess = transforms.Compose([
        transforms.Grayscale(),  # Convertir l'image en niveaux de gris
        transforms.Resize((28, 28)),  # Redimensionner l'image à la taille d'entrée attendue par le modèle
        transforms.ToTensor(),  # Convertir l'image en un tensor
        transforms.Normalize((0.5,), (0.5,))  # Normaliser les valeurs de pixel
    ])
    # Appliquer les transformations à l'image
    preprocessed_image = preprocess(image)
    # Continuer avec le reste du code...
else:
    print(f"Le fichier {image_path} n'existe pas.")

# Appliquer les transformations à l'image
preprocessed_image = preprocess(image)

# Ajouter une dimension pour représenter le batch (dimension de lot)
preprocessed_image = preprocessed_image.unsqueeze(0)

# Faire la prédiction
prediction = model(preprocessed_image)

result_tensor = prediction

# Appliquer la fonction softmax
probabilities = F.softmax(result_tensor, dim=1)

# Trouver la classe prédite
predicted_class = torch.argmax(probabilities, dim=1).item()

print("Probabilités:", probabilities)
print("Classe prédite:", predicted_class)