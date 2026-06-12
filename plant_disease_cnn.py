import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths
DATA_DIR = Path("D:/PlantDiseaseDetector/data/plantvillage dataset/color")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

# Split into train/test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Apply test transform to test set
test_dataset.dataset.transform = test_transform

# DataLoaders
# Change batch size
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Info
print(f"Total images: {len(full_dataset)}")
print(f"Train size: {train_size}")
print(f"Test size: {test_size}")
print(f"Number of classes: {len(full_dataset.classes)}")
print(f"Classes: {full_dataset.classes[:5]}...")
# Baseline Model - 2 Linear Layers
class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layer_stack(x)

# input_size = 3 channels * 128 * 128
baseline_model = BaselineModel(input_size=3*64*64, 
                                hidden_size=512, 
                                output_size=38).to(device)
print(baseline_model)
# CNN Model
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return self.classifier(x)

cnn_model = PlantDiseaseCNN(num_classes=38).to(device)
print(cnn_model)
# Loss function and optimizers
loss_fn = nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

# Training function
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_acc = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (y_pred.argmax(1) == y).float().mean().item()
    
    return total_loss/len(dataloader), total_acc/len(dataloader)

# Testing function
def test(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_acc += (y_pred.argmax(1) == y).float().mean().item()
    
    return total_loss/len(dataloader), total_acc/len(dataloader)

print("Functions ready!")
# Train both models and compare
EPOCHS = 5

results = {"baseline_train_loss": [], "baseline_test_loss": [],
           "baseline_train_acc": [], "baseline_test_acc": [],
           "cnn_train_loss": [], "cnn_test_loss": [],
           "cnn_train_acc": [], "cnn_test_acc": []}

print("\n--- Training Baseline Model ---")
for epoch in range(EPOCHS):
    train_loss, train_acc = train(baseline_model, train_loader, baseline_optimizer, loss_fn, device)
    test_loss, test_acc = test(baseline_model, test_loader, loss_fn, device)
    results["baseline_train_loss"].append(train_loss)
    results["baseline_test_loss"].append(test_loss)
    results["baseline_train_acc"].append(train_acc)
    results["baseline_test_acc"].append(test_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

print("\n--- Training CNN Model ---")
for epoch in range(EPOCHS):
    train_loss, train_acc = train(cnn_model, train_loader, cnn_optimizer, loss_fn, device)
    test_loss, test_acc = test(cnn_model, test_loader, loss_fn, device)
    results["cnn_train_loss"].append(train_loss)
    results["cnn_test_loss"].append(test_loss)
    results["cnn_train_acc"].append(train_acc)
    results["cnn_test_acc"].append(test_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    # Plot loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(results["baseline_test_acc"], label="Baseline")
plt.plot(results["cnn_test_acc"], label="CNN")
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results["baseline_test_loss"], label="Baseline")
plt.plot(results["cnn_test_loss"], label="CNN")
plt.title("Test Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("D:/project/model_comparison.png")
plt.show()
print("Plot saved!")

# Save best model
MODEL_PATH = Path("D:/project/models")
MODEL_PATH.mkdir(exist_ok=True)
torch.save(cnn_model.state_dict(), MODEL_PATH / "plant_disease_cnn.pth")
print(f"Model saved to {MODEL_PATH / 'plant_disease_cnn.pth'}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds, all_labels = [], []
cnn_model.eval()
with torch.inference_mode():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        preds = cnn_model(X).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=False, fmt="d", 
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes)
plt.title("Confusion Matrix - Plant Disease CNN")
plt.savefig("D:/project/confusion_matrix.png")
plt.show()
print("Confusion matrix saved!")