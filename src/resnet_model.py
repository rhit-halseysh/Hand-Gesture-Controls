"""
ResNet-18 model for gesture recognition.
"""
import torch
import torch.nn as nn
from torchvision import models


class GestureResNet(nn.Module):
    """ResNet-18 for gesture recognition using transfer learning."""
    
    def __init__(self, num_classes=7):
        super().__init__()
        # Load pretrained ResNet-18
        self.model = models.resnet18(weights='DEFAULT')
        
        # Replace final layer for our gesture classes
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)


def save_model(model, filepath, class_names):
    """Save model and class names."""
    torch.save({
        'model_state': model.state_dict(),
        'classes': class_names
    }, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath, num_classes):
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model = GestureResNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint['classes']
