"""
Main application for hand gesture tracking.
Integrates hand tracking, gesture recognition, and action handling.
"""

import cv2
import time
import sys
from typing import Optional
from src.hand_tracker import HandTracker
from src.resnet_recognizer import ResNetRecognizer
from src.action_handler import ActionHandler
from src.ui import GestureTrackerUI


class GestureTrackerApp:
    """Main application orchestrating gesture tracking."""
    
    def __init__(self, camera_id: int = 0, debug: bool = True):
        """
        Initialize the gesture tracker application.
        
        Args:
            camera_id: Camera device ID (default 0 for primary camera).
            debug: If True, prints debug information.
        """
        self.debug = debug
        self.camera_id = camera_id
        
        # Components
        self.hand_tracker: Optional[HandTracker] = None
        self.recognizer: Optional[ResNetRecognizer] = None
        self.action_handler: Optional[ActionHandler] = None
        self.ui: Optional[GestureTrackerUI] = None
        
        # State
        self.is_running = False
        self.is_tracking = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_times = []
        self.current_gesture = "no_gesture"  # Changed to string
        
        if self.debug:
            print("[GestureTrackerApp] Initializing...")
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all application components."""
        try:
            # Initialize hand tracker (for landmarks/drawing only)
            self.hand_tracker = HandTracker(debug=self.debug, use_gesture_recognition=False)
            print("✓ Hand tracker initialized")
            
            # Initialize gesture recognizer (exactly like test_model.py)
            model_path = 'models/best_model.pth'
            self.recognizer = ResNetRecognizer(model_path)
            print("✓ Gesture recognizer initialized")
            
            # Initialize action handler
            self.action_handler = ActionHandler(debug=self.debug)
            print("✓ Action handler initialized")
            
            # Initialize UI
            self.ui = GestureTrackerUI(debug=self.debug)
            self.ui.set_toggle_callback(self._on_toggle)
            print("✓ UI initialized")
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("✓ Camera initialized")
            print("[GestureTrackerApp] All components initialized successfully")
            
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            self.cleanup()
            raise
    
    def _on_toggle(self, should_track: bool):
        """
        Callback when toggle button is pressed.
        
        Args:
            should_track: True to start tracking, False to stop.
        """
        self.is_tracking = should_track
        if self.debug:
            print(f"[GestureTrackerApp] Tracking toggled: {should_track}")
    
    def run(self):
        """Run the main application loop."""
        self.is_running = True
        print("[GestureTrackerApp] Starting main loop...")
        
        try:
            while self.is_running:
                # Process UI events
                if not self.ui.process_events():
                    break
                
                # Get frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    if self.debug:
                        print("✗ Failed to read from camera")
                    continue
                
                frame_start_time = time.time()
                
                # Process frame only if tracking is enabled
                if self.is_tracking:
                    # Mirror the frame (exactly like test_model.py)
                    frame = cv2.flip(frame, 1)
                    
                    # Recognize gesture on full frame (exactly like test_model.py)
                    gesture, gesture_confidence = self.recognizer.recognize(frame)
                    
                    # Run hand tracker for landmark drawing only
                    try:
                        frame, hand_data_list = self.hand_tracker.process_frame(frame)
                    except Exception as e:
                        if self.debug:
                            print(f"✗ Error processing frame: {e}")
                        hand_data_list = None
                    
                    # Update UI hand count
                    hand_count = len(hand_data_list) if hand_data_list else 0
                    self.ui.update_hands_count(hand_count)
                    
                    # Update UI with gesture and confidence
                    self.ui.update_gesture(gesture, gesture_confidence)
                    
                    # Only trigger actions with high confidence and accepted gestures
                    if gesture_confidence > 0.7 and gesture in self.action_handler.gesture_actions:
                        self.current_gesture = gesture
                        self.action_handler.handle_gesture(gesture)
                        
                        if self.debug:
                            print(f"Gesture: {gesture} (confidence: {gesture_confidence:.3f})")
                    else:
                        self.current_gesture = gesture
                else:
                    self.ui.update_gesture("--", 0.0)
                
                # Calculate and display FPS
                frame_time = time.time() - frame_start_time
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                self.ui.update_fps(fps)
                
                # Display frame
                cv2.imshow("Hand Gesture Tracker", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[GestureTrackerApp] 'q' pressed, exiting...")
                    break
                elif key == ord(' '):
                    self.is_tracking = not self.is_tracking
                    if self.debug:
                        print(f"[GestureTrackerApp] Space pressed, tracking: {self.is_tracking}")
        
        except KeyboardInterrupt:
            print("\n[GestureTrackerApp] Keyboard interrupt received")
        except Exception as e:
            print(f"✗ Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("[GestureTrackerApp] Cleaning up...")
        
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            print("✓ Camera released")
        
        if self.hand_tracker:
            self.hand_tracker.release()
            print("✓ Hand tracker released")
        
        if self.ui:
            self.ui.close()
            print("✓ UI closed")
        
        cv2.destroyAllWindows()
        print("[GestureTrackerApp] Cleanup complete")


def main():
    """Main entry point."""
    print("=" * 50)
    print("Hand Gesture Tracker")
    print("=" * 50)
    print("Controls:")
    print("  - Click 'Start Tracking' button to begin")
    print("  - Press SPACE to toggle tracking on/off")
    print("  - Press 'q' to quit")
    print("=" * 50)
    
    app = GestureTrackerApp(camera_id=0, debug=True)
    app.run()


if __name__ == "__main__":
    main()
