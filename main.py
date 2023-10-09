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

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

current_directory = os.path.dirname(os.path.abspath(__file__))

data_directory = os.path.join(current_directory, 'data')

os.makedirs(data_directory, exist_ok=True)

train_dataset = MNIST(root=data_directory, train=True, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

validation_dataset = MNIST(root=data_directory, train=False, transform=transform, download=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=False)


for epoch in range(30):

    model.train()
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

    scheduler.step()

image_path = "img/six.jpg"

image_path = os.path.abspath(image_path)

if os.path.exists(image_path):

    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    preprocessed_image = preprocess(image)

else:
    print(f"Le fichier {image_path} n'existe pas.")

preprocessed_image = preprocess(image)

preprocessed_image = preprocessed_image.unsqueeze(0)

prediction = model(preprocessed_image)

result_tensor = prediction

probabilities = F.softmax(result_tensor, dim=1)

predicted_class = torch.argmax(probabilities, dim=1).item()

print("Probabilités:", probabilities)
print("Classe prédite:", predicted_class)