import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

os.makedirs("../results", exist_ok=True)


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root='../MNIST-full/train', transform=transform)
test_dataset = datasets.ImageFolder(root='../MNIST-full/test', transform=transform)


val_size = 10000
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CNN Model
class CNNModel(nn.Module):
    def __init__(self, kernel_size=3, num_conv_layers=2, num_classes=10):
        super(CNNModel, self).__init__()
        layers = []
        in_channels = 1
        out_channels = 32

        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels *= 2

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        # Automatically calculate output feature dimension
        with torch.no_grad():
            sample = torch.zeros(1, 1, 28, 28)
            sample_out = self.conv(sample)
            flat_dim = sample_out.view(1, -1).size(1)

        self.fc = nn.Linear(flat_dim, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Training + Validation
def train_and_validate(model, lr, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# Hyperparameter tuning
kernel_sizes = [3, 5, 7]
conv_layers = [2, 3]
learning_rates = [0.001, 0.0005]

best_val_acc = 0
best_params = None
best_model_state = None

for k in kernel_sizes:
    for n in conv_layers:
        for lr in learning_rates:
            print(f"\nTraining CNN with kernel={k}, conv_layers={n}, lr={lr}")
            model = CNNModel(kernel_size=k, num_conv_layers=n).to(device)
            train_losses, val_losses, train_accs, val_accs = train_and_validate(model, lr)

            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title(f'Loss (kernel={k}, layers={n}, lr={lr})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"../results/cnn_loss_k{k}_l{n}_lr{lr}.png")
            plt.close()

            plt.figure()
            plt.plot(train_accs, label='Train Acc')
            plt.plot(val_accs, label='Val Acc')
            plt.title(f'Accuracy (kernel={k}, layers={n}, lr={lr})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f"../results/cnn_acc_k{k}_l{n}_lr{lr}.png")
            plt.close()

            final_val_acc = val_accs[-1]
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_params = (k, n, lr)
                best_model_state = model.state_dict()


print(f"\nBest CNN configuration: kernel={best_params[0]}, conv_layers={best_params[1]}, lr={best_params[2]}")
best_model = CNNModel(kernel_size=best_params[0], num_conv_layers=best_params[1]).to(device)
best_model.load_state_dict(best_model_state)
best_model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")


with open("../results/cnn_results.txt", "w") as f:
    f.write(f"Best CNN params: kernel={best_params[0]}, conv_layers={best_params[1]}, lr={best_params[2]}\n")
    f.write(f"Validation Accuracy: {best_val_acc}\n")
    f.write(f"Test Accuracy: {test_acc}\n")

print("Training completed and results saved in '../results' folder.")
