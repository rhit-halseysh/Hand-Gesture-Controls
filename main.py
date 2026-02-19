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
        self.grecognizer: Optional[ResNetRecognizer] = None
        self.aslrecognizer: Optional[ResNetRecognizer] = None
        self.recognizer: Optional[ResNetRecognizer] = None
        self.action_handler: Optional[ActionHandler] = None
        self.ui: Optional[GestureTrackerUI] = None
        
        # State
        self.is_running = False
        self.is_tracking = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_times = []
        self.current_gesture = "no_gesture"  # Changed to string
        self.frame_count = 0
        self.last_gesture = "no_gesture"
        self.last_gesture_confidence = 0.0
        self.last_hand_data = None
        self.mode = 'gesture'  # 'gesture' or 'asl'
        
        # if self.debug:
        #     print("[GestureTrackerApp] Initializing...")
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all application components."""
        try:
            # Initialize hand tracker (for landmarks/drawing only)
            self.hand_tracker = HandTracker(debug=self.debug, use_gesture_recognition=False)
            print("✓ Hand tracker initialized")
            
            # Initialize gesture recognizer (exactly like test_model.py)
            gmodel_path = 'models/resnet18.pth'
            self.grecognizer = ResNetRecognizer(gmodel_path)
            print("✓ Gesture recognizer initialized")
            
            # Initialize ASL recognizer
            aslmodel_path = 'models/asl_resnet18_model.pth'
            self.aslrecognizer = ResNetRecognizer(aslmodel_path)
            print("✓ ASL recognizer initialized")

            self.recognizer = self.grecognizer
            
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
            
            # Set camera properties (lower resolution for better performance)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
            
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
        # if self.debug:
        #     print(f"[GestureTrackerApp] Tracking toggled: {should_track}")
    
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
                    # if self.debug:
                    #     print("✗ Failed to read from camera")
                    continue
                
                self.frame_count += 1
                frame_start_time = time.time()
                
                # Process frame only if tracking is enabled
                if self.is_tracking:
                    # Mirror the frame (exactly like test_model.py)
                    frame = cv2.flip(frame, 1)
                    
                    # Run gesture recognition only every 2 frames (MAJOR performance boost)
                    if self.frame_count % 2 == 0:
                        gesture, gesture_confidence = self.recognizer.recognize(frame)
                        self.last_gesture = gesture
                        self.last_gesture_confidence = gesture_confidence
                    else:
                        # Reuse last result
                        gesture = self.last_gesture
                        gesture_confidence = self.last_gesture_confidence
                    
                    # Run hand tracker every frame when landmarks enabled, otherwise every 2 frames
                    # This prevents flickering while maintaining performance when landmarks are off
                    should_run_tracker = self.hand_tracker.draw_landmarks or (self.frame_count % 2 == 0)
                    
                    if should_run_tracker:
                        try:
                            frame, hand_data_list = self.hand_tracker.process_frame(frame)
                        except Exception as e:
                            hand_data_list = None
                    else:
                        # Reuse last hand data when skipping frames
                        hand_data_list = self.last_hand_data
                    
                    # Cache hand data for next frame
                    if hand_data_list is not None:
                        self.last_hand_data = hand_data_list
                        self.last_hand_data = hand_data_list
                    
                    # Update UI hand count (only every 6 frames to reduce overhead)
                    if self.frame_count % 6 == 0 and hand_data_list:
                        hand_count = len(hand_data_list)
                        self.ui.update_hands_count(hand_count)
                    
                    # Update mouse position based on hand tracking (only when we have data)
                    if hand_data_list and len(hand_data_list) > 0:
                        first_hand = hand_data_list[0]
                        palm_position = first_hand.get('palm_position')
                        if palm_position:
                            h, w, _ = frame.shape
                            self.action_handler.move_mouse_to_hand(palm_position, w, h)
                    
                    # Update UI with gesture and confidence (only every 6 frames)
                    if self.frame_count % 6 == 0:
                        self.ui.update_gesture(gesture, gesture_confidence)
                    
                    # Only trigger actions with high confidence and accepted gestures
                    if gesture_confidence > 0.7 and gesture in self.action_handler.gesture_actions:
                        self.current_gesture = gesture
                        self.action_handler.handle_gesture(gesture)
                    else:
                        self.current_gesture = gesture
                else:
                    # When not tracking, update UI more frequently for responsiveness
                    if self.frame_count % 2 == 0:
                        self.ui.update_gesture("--", 0.0)
                        self.ui.update_hands_count(0)
                
                # Calculate and display FPS
                frame_time = time.time() - frame_start_time
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                # Update FPS more frequently when not tracking for better responsiveness
                if self.is_tracking:
                    # When tracking, update every 5 frames (performance)
                    if self.frame_count % 5 == 0:
                        self.ui.update_fps(fps)
                else:
                    # When not tracking, update every 2 frames (more responsive)
                    if self.frame_count % 2 == 0:
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
                    # if self.debug:
                    #     print(f"[GestureTrackerApp] Space pressed, tracking: {self.is_tracking}")
                elif key == ord('m'):
                    mouse_enabled = self.action_handler.toggle_mouse_control()
                    self.ui.update_mouse_control_status(mouse_enabled)
                    # if self.debug:
                    #     print(f"[GestureTrackerApp] 'm' pressed, mouse control: {mouse_enabled}")
                elif key == ord('i'):
                    landmarks_enabled = self.hand_tracker.toggle_landmarks()
                    self.ui.update_landmarks_status(landmarks_enabled)
                    # if self.debug:
                    #     print(f"[GestureTrackerApp] 'i' pressed, landmarks: {landmarks_enabled}")
                elif key == ord('g'):
                    if self.mode == 'gesture':
                        self.mode = 'asl'
                        self.recognizer = self.aslrecognizer
                        print("[GestureTrackerApp] Switched to ASL mode")
                        self.ui.update_mode_status(self.mode)
                    else:
                        self.mode = 'gesture'
                        self.recognizer = self.grecognizer
                        print("[GestureTrackerApp] Switched to Gesture mode")
                        self.ui.update_mode_status(self.mode)
        
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
    print("  - Press 'm' to toggle mouse control on/off")
    print("  - Press 'i' to toggle landmarks on/off")
    print("  - Press 'q' to quit")
    print("=" * 50)
    
    app = GestureTrackerApp(camera_id=0, debug=True)
    app.run()


if __name__ == "__main__":
    main()
