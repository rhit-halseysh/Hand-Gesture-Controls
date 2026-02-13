import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the gesture dataset into train/valid/test sets
    
    Args:
        source_dir: Root/gestures/ directory containing class folders
        output_dir: Output directory for split datasets
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Splitting dataset from {source_dir}")
    print(f"Train: {train_ratio*100:.0f}%, Valid: {valid_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    print('='*60)
    
    total_images = 0
    class_stats = {}
    
    # Process each class folder
    for class_folder in sorted(source_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_folder.glob(ext)))
        
        if len(image_files) == 0:
            print(f"  Warning: No images found in {class_name}")
            continue
        
        # Shuffle
        image_files = list(image_files)
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n_images = len(image_files)
        n_train = int(n_images * train_ratio)
        n_valid = int(n_images * valid_ratio)
        
        train_files = image_files[:n_train]
        valid_files = image_files[n_train:n_train + n_valid]
        test_files = image_files[n_train + n_valid:]
        
        # Copy files to appropriate directories
        for split, files in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
            split_dir = output_path / split
            for i, img_file in enumerate(files):
                # Create filename: ClassLetter_originalname.ext
                new_name = f"{class_name}_{img_file.name}"
                dest_path = split_dir / new_name
                shutil.copy2(img_file, dest_path)
        
        class_stats[class_name] = {
            'total': n_images,
            'train': len(train_files),
            'valid': len(valid_files),
            'test': len(test_files)
        }
        
        total_images += n_images
        
        print(f"  Total: {n_images} images")
        print(f"  Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    
    print(f"\n{'='*60}")
    print("DATASET SPLIT SUMMARY")
    print('='*60)
    print(f"Total classes: {len(class_stats)}")
    print(f"Total images: {total_images}")
    
    total_train = sum(s['train'] for s in class_stats.values())
    total_valid = sum(s['valid'] for s in class_stats.values())
    total_test = sum(s['test'] for s in class_stats.values())
    
    print(f"\nTrain set: {total_train} images ({100*total_train/total_images:.1f}%)")
    print(f"Valid set: {total_valid} images ({100*total_valid/total_images:.1f}%)")
    print(f"Test set: {total_test} images ({100*total_test/total_images:.1f}%)")
    
    print(f"\nDataset saved to: {output_dir}")
    print('='*60)
    
    return class_stats


# Custom Dataset class for ASL images
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                self.images.append(filename)
                label = filename[0].upper()
                self.labels.append(label)
        
        # Create label to index mapping
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Print class distribution
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Found {len(self.images)} images with {len(self.unique_labels)} classes")
        print(f"Classes: {self.unique_labels}")
        print(f"Class distribution: {dict(sorted(label_counts.items()))}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.label_to_idx[self.labels[idx]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_model(num_classes, model_name='resnet50', pretrained=True, freeze_layers=True):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.fc.in_features
        
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.fc.in_features
        
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.fc.in_features
        
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.classifier[1].in_features
        
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.features[-2:].parameters():
                param.requires_grad = True
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    epoch_loss = train_loss / train_total
    epoch_acc = 100 * train_correct / train_total
    
    return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            valid_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
    
    epoch_loss = valid_loss / valid_total
    epoch_acc = 100 * valid_correct / valid_total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device, use_amp=True):
    best_val_acc = 0
    patience = 7
    patience_counter = 0
    
    os.makedirs('models', exist_ok=True)
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    print("\n" + "="*50)
    print("Training started")
    print("="*50)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train - Loss: {train_loss:.3f}, Acc: {train_acc:.1f}%")
        print(f"  Val   - Loss: {val_loss:.3f}, Acc: {val_acc:.1f}%")
        if scheduler is not None:
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'models/asl_best_model.pth')
            print(f"  New best model! ({val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    print("\n" + "="*50)
    print(f"Training completed. Best validation accuracy: {best_val_acc:.1f}%")
    print("="*50)
    
    return model


def test_model(model, test_loader, device, idx_to_label):
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    print("\n" + "="*50)
    print("Testing model...")
    print("="*50)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    print("-" * 50)
    class_correct = {}
    class_total = {}
    
    for pred, label in zip(all_predictions, all_labels):
        label_name = idx_to_label[label]
        if label_name not in class_correct:
            class_correct[label_name] = 0
            class_total[label_name] = 0
        
        class_total[label_name] += 1
        if pred == label:
            class_correct[label_name] += 1
    
    for label_name in sorted(class_correct.keys()):
        acc = 100 * class_correct[label_name] / class_total[label_name]
        print(f"  {label_name}: {acc:.1f}% ({class_correct[label_name]}/{class_total[label_name]})")
    
    print("="*50)
    
    return accuracy, all_predictions, all_labels


def main():
    # ============ CONFIGURATION ============
    # Source dataset location
    SOURCE_GESTURES_DIR = 'Root/gestures'  # Update this path
    OUTPUT_SPLIT_DIR = 'asl_split'  # Where to save train/valid/test splits
    
    # Split ratios
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Model selection: 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
    MODEL_NAME = 'resnet18'
    FREEZE_LAYERS = False
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    INITIAL_LR = 0.001
    WEIGHT_DECAY = 0.0001
    USE_AMP = True
    
    MODEL_SAVE_PATH = 'asl_resnet18_model.pth'
    # =======================================
    
    # Step 1: Split the dataset (only run once)
    if not os.path.exists(OUTPUT_SPLIT_DIR):
        print("Splitting dataset...")
        split_dataset(SOURCE_GESTURES_DIR, OUTPUT_SPLIT_DIR, 
                     TRAIN_RATIO, VALID_RATIO, TEST_RATIO)
    else:
        print(f"Using existing split at {OUTPUT_SPLIT_DIR}")
        print("Delete this directory to re-split the dataset")
    
    # Define paths to split datasets
    TRAIN_DIR = os.path.join(OUTPUT_SPLIT_DIR, 'train')
    VALID_DIR = os.path.join(OUTPUT_SPLIT_DIR, 'valid')
    TEST_DIR = os.path.join(OUTPUT_SPLIT_DIR, 'test')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(os.cpu_count())
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    print(f"Number of CPU threads: {os.cpu_count()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Freeze base layers: {FREEZE_LAYERS}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print('='*60)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ASLDataset(TRAIN_DIR, transform=train_transform)
    valid_dataset = ASLDataset(VALID_DIR, transform=test_transform)
    test_dataset = ASLDataset(TEST_DIR, transform=test_transform)
    
    print(f"\nTrain images: {len(train_dataset)}")
    print(f"Val images: {len(valid_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Create model
    num_classes = len(train_dataset.unique_labels)
    print(f"\nCreating {MODEL_NAME} model with {num_classes} classes...")
    model = create_model(num_classes, model_name=MODEL_NAME, pretrained=True, 
                        freeze_layers=FREEZE_LAYERS).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for different parts
    if FREEZE_LAYERS:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    else:
        # Lower learning rate for pretrained layers
        pretrained_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': INITIAL_LR * 0.1},
            {'params': new_params, 'lr': INITIAL_LR}
        ], weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                      patience=3)
    
    # Train model
    model = train_model(model, train_loader, valid_loader, criterion, optimizer, 
                       scheduler, NUM_EPOCHS, device, use_amp=USE_AMP)
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load('models/asl_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model
    test_accuracy, predictions, labels = test_model(model, test_loader, device, train_dataset.idx_to_label)
    
    # Save final model with all metadata
    print(f"\nSaving final model to {MODEL_SAVE_PATH}...")
    torch.save({
        'model_name': MODEL_NAME,
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
        'test_accuracy': test_accuracy,
        'classes': train_dataset.unique_labels
    }, MODEL_SAVE_PATH)
    print("Model saved successfully!")
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")


if __name__ == '__main__':
    main()