import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, X):
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        return X


def data_preparation(data_dir, batch_size=128, val_size=10000, num_workers=4):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    train_size = len(full_train) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def evaluate(model, val_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        val_loss, val_correct, val_total = 0.0, 0, 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)  # flatten
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    return val_loss, val_acc


def train_and_validate(model, train_loader, val_loader, device, lr, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)  # flatten
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


def plot(train_losses, val_losses, train_accs, val_accs, hidden, lr, out_dir):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss (hidden={hidden}, lr={lr})')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy (hidden={hidden}, lr={lr})')
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(out_dir, f"mlp_hidden{hidden}_lr{int(-math.log10(lr)) if lr<1 else int(lr)}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Combined plots per hidden size
def plot_combined_hidden(all_results, out_dir):
    hidden_sizes = sorted({r['hidden_size'] for r in all_results})
    for hidden in hidden_sizes:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        for r in sorted([x for x in all_results if x['hidden_size'] == hidden], key=lambda d: d['lr']):
            plt.plot(range(1, len(r['train_losses']) + 1), r['val_losses'], marker='o', label=f"val lr={r['lr']}")
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Val Loss (hidden={hidden})'); plt.grid(True); plt.legend()

        plt.subplot(1, 2, 2)
        for r in sorted([x for x in all_results if x['hidden_size'] == hidden], key=lambda d: d['lr']):
            plt.plot(range(1, len(r['val_accs']) + 1), r['val_accs'], marker='o', label=f"val lr={r['lr']}")
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'Val Accuracy (hidden={hidden})'); plt.grid(True); plt.legend()

        out_path = os.path.join(out_dir, f"mlp_combined_hidden{hidden}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def main():
    out_dir = '../results/mlp'
    os.makedirs(out_dir, exist_ok=True)

    data_dir = '../../MNIST-full'

    param_grid = {
        'hidden_size': [64, 128, 256],
        'lr': [1e-4, 1e-3, 1e-2, 1e-1],
    }

    batch_size = 128
    num_epochs = 10
    num_workers = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = \
        data_preparation(data_dir, batch_size=batch_size, val_size=10000, num_workers=num_workers)

    all_results = []
    best_val_acc = -1.0
    best_params = None
    best_model_state = None

    for hidden in param_grid['hidden_size']:
        for lr in param_grid['lr']:
            print(f"\nTraining MLP with hidden={hidden}, lr={lr}")
            model = MLP(hidden_size=hidden)
            train_losses, val_losses, train_accs, val_accs = train_and_validate(
                model, train_loader, val_loader, device, lr, epochs=num_epochs)

            final_val_acc = val_accs[-1]
            res = {
                'hidden_size': hidden,
                'lr': lr,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'val_acc': final_val_acc,
            }
            all_results.append(res)

            plot(train_losses, val_losses, train_accs, val_accs, hidden, lr, out_dir)

            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_params = {'hidden_size': hidden, 'lr': lr}
                best_model_state = model.state_dict()
    
    
    # final evaluation on test set with best model
    print(f"\nBest MLP configuration: hidden_size={best_params['hidden_size']}, lr={best_params['lr']}")
    best_model = MLP(hidden_size=best_params['hidden_size'])
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    _, test_acc = evaluate(best_model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    with open("../results/mlp/mlp_results.txt", "w") as f:
        f.write(f"Best MLP params: hidden_size={best_params['hidden_size']}, lr={best_params['lr']}\n")
        f.write(f"Validation Accuracy: {best_val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    print("Training completed and results saved in '../results/cnn' folder.")
    
    # Combined plots
    plot_combined_hidden(all_results, out_dir)


if __name__ == '__main__':
    main()
