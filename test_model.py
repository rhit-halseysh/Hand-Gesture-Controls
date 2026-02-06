"""
Test trained ResNet model with webcam.
Usage: python test_model.py
"""
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.resnet_recognizer import ResNetRecognizer


def main():
    """Run webcam test with gesture recognition."""
    # Load model
    print("Loading model...")
    model_path = 'models/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    recognizer = ResNetRecognizer(model_path)
    print("Model loaded successfully!")
    print(f"Recognized gestures: {recognizer.classes}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nWebcam opened. Press 'q' to quit.")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Mirror the frame for better UX
        frame = cv2.flip(frame, 1)
        
        # Recognize gesture
        gesture, confidence = recognizer.recognize(frame)
        
        # Prepare display
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Display gesture name
        text = f"Gesture: {gesture}"
        cv2.putText(frame, text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display confidence
        conf_text = f"Confidence: {confidence:.2%}"
        cv2.putText(frame, conf_text, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (w - 200, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Gesture Recognition Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest completed.")


if __name__ == '__main__':
    main()
