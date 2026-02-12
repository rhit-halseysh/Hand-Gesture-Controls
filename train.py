"""
Training script for ResNet-50 gesture recognition.
Usage: python train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from src.resnet_model import GestureResNet, save_model


# ========== TRAINING PARAMETERS ==========
DATA_DIR = 'dataset'               # Path to dataset with train/ and val/ folders
EPOCHS = 30                         # Number of training epochs
BATCH_SIZE = 32                     # Batch size for training
LEARNING_RATE = 0.01              # Learning rate
# ======================================================================


def get_data_loaders(data_dir, batch_size=32):
    """Create train and validation data loaders."""
    # Transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def train_model(data_dir, epochs, batch_size, learning_rate):
    """
    Train the ResNet model.
    
    Args:
        data_dir: Path to dataset with train/ and val/ folders
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(os.cpu_count())
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size)
    print(f"Classes: {classes}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Val images: {len(val_loader.dataset)}")
    
    # Create model
    model = GestureResNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_acc = 0
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*50)
    print("Training started")
    print("="*50)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train - Loss: {train_loss:.3f}, Acc: {train_acc:.1f}%")
        print(f"  Val   - Loss: {val_loss:.3f}, Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'models/best_model.pth', classes)
            print(f"New best model! ({val_acc:.1f}%)")
    
    print("\n" + "="*50)
    print(f"Training finished...\n Best validation accuracy: {best_val_acc:.1f}%")
    print("="*50)


if __name__ == '__main__':
    train_model(DATA_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE)
