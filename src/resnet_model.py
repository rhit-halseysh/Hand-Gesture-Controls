"""
ResNet model for gesture recognition.
"""
import torch
import torch.nn as nn
from torchvision import models


class GestureResNet(nn.Module):
    def __init__(self, num_classes=7, sequential_fc=False):
        super().__init__()
        self.model = models.resnet18(weights='DEFAULT')

        if sequential_fc:
            # Matches ASL model: Sequential(Dropout, Linear)
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            # Matches gesture model: plain Linear
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


def save_model(model, filepath, class_names):
    """Save model and class names."""
    torch.save({
        'model_state': model.state_dict(),
        'classes': class_names
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, num_classes):
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location='cpu')

    # Resolve label mapping
    if 'idx_to_label' in checkpoint:
        idx_to_label = checkpoint['idx_to_label']
    elif 'classes' in checkpoint:
        idx_to_label = {i: label for i, label in enumerate(checkpoint['classes'])}
    else:
        raise KeyError("Checkpoint missing label mapping")

    num_classes = checkpoint.get('num_classes') or len(idx_to_label)

    # Resolve state dict
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model_state')
    if state_dict is None:
        raise KeyError("Checkpoint missing model state — expected 'model_state_dict' or 'model_state'")

    # Fix missing model. prefix
    first_key = next(iter(state_dict))
    if not first_key.startswith('model.'):
        state_dict = {f'model.{k}': v for k, v in state_dict.items()}

    # Detect whether fc is plain Linear or Sequential(dropout, Linear)
    uses_sequential_fc = 'model.fc.1.weight' in state_dict

    model = GestureResNet(num_classes=num_classes, sequential_fc=uses_sequential_fc)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, idx_to_label