"""
Gesture recognizer using trained ResNet-18 model.
"""
import torch
from torchvision import transforms
import cv2

from src.resnet_model import load_model


class ResNetRecognizer:
    """Recognize gestures using trained ResNet-18 model."""
    
    def __init__(self, model_path='models/resnet18.pth'):
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model file
        """
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        self.classes = checkpoint['classes']
        
        # Load model and move to device
        self.model, _ = load_model(model_path, len(self.classes))
        self.model = self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded with classes: {self.classes}")
        print(f"Running on: {self.device}")
    
    def recognize(self, image):
        """
        Recognize gesture from image.
        
        Args:
            image: OpenCV image (BGR format)
        
        Returns:
            (gesture_name, confidence) - e.g. ('peace', 0.95)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess and predict
        img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, predicted = probs.max(0)
        
        gesture_name = self.classes[predicted.item()]
        return gesture_name, confidence.item()
