"""
Hand tracking module using OpenCV and MediaPipe Tasks.
Detects hand landmarks and provides hand position data with custom gesture recognition.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, List, Tuple
import os
import urllib.request
from collections import deque, Counter
from src.resnet_recognizer import ResNetRecognizer


class HandTracker:
    """Tracks hand positions and landmarks using MediaPipe Tasks with custom gesture recognition."""
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"
    GESTURE_MODEL_PATH = 'models/best_model.pth'
    
    def __init__(self, debug: bool = False, use_gesture_recognition: bool = True):
        """
        Initialize hand tracker.
        
        Args:
            debug: If True, prints debug information.
            use_gesture_recognition: If True, load custom gesture recognition model.
        """
        self.debug = debug
        self.detector = None
        self.gesture_recognizer = None
        self.use_gesture_recognition = use_gesture_recognition
        self.draw_landmarks = False  # Landmark drawing can be toggled
        
        # Gesture smoothing buffer (prevents flickering)
        self.gesture_buffer_size = 15  # Number of frames to average over
        self.gesture_history = deque(maxlen=self.gesture_buffer_size)
        self.stable_gesture = 'no_gesture'
        self.stable_confidence = 0.0
        self.min_agreement = 0.5  # 50% of buffer must agree
        
        # Only accept gestures that have actions bound
        self.accepted_gestures = {
            'peace', 'peace_inverted', 'point', 'like', 'dislike',
            'call', 'mute', 'stop', 'one', 'two_up', 'ok',
            'palm', 'fist', 'hand_heart', 'hand_heart2', 'rock'
        }
        
        # Try to load the models
        self._setup_model()
        if self.use_gesture_recognition:
            self._setup_gesture_model()
        
        # if self.debug:
        #     print("[HandTracker] Initialized")
    
    def _setup_model(self):
        """Setup hand detection model, downloading if necessary."""
        # Check if model file exists, if not download it
        if not os.path.exists(self.MODEL_PATH):
            # if self.debug:
            #     print(f"[HandTracker] Downloading model from {self.MODEL_URL}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            # if self.debug:
            #     print("[HandTracker] Model downloaded successfully")
        
        # Create hand landmarker with Tasks API
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        # if self.debug:
        #     print("[HandTracker] Using MediaPipe Tasks API")
    
    def _setup_gesture_model(self):
        """Setup custom gesture recognition model."""
        try:
            if os.path.exists(self.GESTURE_MODEL_PATH):
                self.gesture_recognizer = ResNetRecognizer(self.GESTURE_MODEL_PATH)
                # if self.debug:
                #     print(f"[HandTracker] Custom gesture model loaded from {self.GESTURE_MODEL_PATH}")
            else:
                # if self.debug:
                #     print(f"[HandTracker] Warning: Gesture model not found at {self.GESTURE_MODEL_PATH}")
                self.use_gesture_recognition = False
        except Exception as e:
            # if self.debug:
            #     print(f"[HandTracker] Error loading gesture model: {e}")
            self.use_gesture_recognition = False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[dict]]]:
        """
        Process a frame and detect hands.
        
        Args:
            frame: Input frame from camera (BGR format).
            
        Returns:
            Tuple of (annotated_frame, hand_data)
            hand_data contains landmarks and hand info, or None if no hands detected.
        """
        if frame is None or frame.size == 0:
            return frame, None
        
        try:
            annotated_frame = frame.copy()
            hand_data_list = []
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with Tasks API
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(image)
            
            if results.hand_landmarks and results.handedness:
                h, w, _ = frame.shape
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    hand_label = handedness[0].category_name
                    hand_confidence = handedness[0].score
                    
                    landmark_list = []
                    for landmark in hand_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        z = landmark.z
                        landmark_list.append((x, y, z))
                    
                    hand_data = {
                        'label': hand_label,
                        'confidence': hand_confidence,
                        'landmarks': landmark_list,
                        'palm_position': self._get_palm_position(landmark_list),
                        'fingers_up': self._get_fingers_up(landmark_list)
                    }
                    
                    # Add gesture recognition if available
                    if self.use_gesture_recognition and self.gesture_recognizer is not None:
                        gesture_result = self._recognize_gesture(frame, landmark_list)
                        if gesture_result:
                            # Add raw result to smoothing buffer
                            self.gesture_history.append(gesture_result)
                            
                            # Get stabilized gesture from buffer
                            stable_gesture, stable_confidence = self._get_stable_gesture()
                            hand_data['gesture'] = stable_gesture
                            hand_data['gesture_confidence'] = stable_confidence
                    hand_data_list.append(hand_data)
                    
                    if self.draw_landmarks:
                        self._draw_landmarks(annotated_frame, landmark_list)
            
            return annotated_frame, hand_data_list if hand_data_list else None
        
        except Exception as e:
            # if self.debug:
            #     print(f"[HandTracker] Fatal error in process_frame: {e}")
            return frame, None
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[int, int, float]]):
        """Draw hand landmarks on frame."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                start_pos = (landmarks[start][0], landmarks[start][1])
                end_pos = (landmarks[end][0], landmarks[end][1])
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
        
        for x, y, _ in landmarks:
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    
    def _get_palm_position(self, landmarks: List[Tuple[int, int, float]]) -> Tuple[int, int]:
        """
        Calculate palm center from landmarks.
        
        Args:
            landmarks: List of (x, y, z) landmark positions.
            
        Returns:
            Tuple of (palm_x, palm_y).
        """
        if len(landmarks) < 14:
            return (0, 0)
        
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        
        palm_x = (wrist[0] + middle_mcp[0] + ring_mcp[0]) // 3
        palm_y = (wrist[1] + middle_mcp[1] + ring_mcp[1]) // 3
        
        return (palm_x, palm_y)
    
    def _get_fingers_up(self, landmarks: List[Tuple[int, int, float]]) -> List[int]:
        """
        Determine which fingers are up (extended).
        
        Args:
            landmarks: List of (x, y, z) landmark positions.
            
        Returns:
            List of 5 binary values (0 or 1) for thumb, index, middle, ring, pinky.
        """
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]
        
        fingers_up = []
        
        # Thumb
        if landmarks[4][0] < landmarks[3][0]:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(tip_ids, pip_ids):
            if landmarks[tip_id][1] < landmarks[pip_id][1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        return fingers_up
    
    def _get_stable_gesture(self) -> Tuple[str, float]:
        """
        Get stabilized gesture from the history buffer.
        Uses majority voting over recent frames to prevent flickering.
        
        Returns:
            Tuple of (gesture_name, average_confidence).
        """
        if not self.gesture_history:
            return self.stable_gesture, self.stable_confidence
        
        # Count gesture occurrences in buffer
        gesture_names = [g[0] for g in self.gesture_history]
        counts = Counter(gesture_names)
        most_common_gesture, most_common_count = counts.most_common(1)[0]
        
        # Only switch gesture if it has enough agreement in the buffer
        agreement_ratio = most_common_count / len(self.gesture_history)
        
        if agreement_ratio >= self.min_agreement:
            # Calculate average confidence for the winning gesture
            matching_confidences = [g[1] for g in self.gesture_history if g[0] == most_common_gesture]
            avg_confidence = sum(matching_confidences) / len(matching_confidences)
            
            self.stable_gesture = most_common_gesture
            self.stable_confidence = avg_confidence
        
        return self.stable_gesture, self.stable_confidence
    
    def _recognize_gesture(self, frame: np.ndarray, landmarks: List[Tuple[int, int, float]]) -> Optional[Tuple[str, float]]:
        """
        Recognize gesture using custom trained model.
        Uses the full frame (matching test_model.py behavior).
        
        Args:
            frame: Original frame.
            landmarks: Hand landmarks.
            
        Returns:
            Tuple of (gesture_name, confidence) or None if recognition fails.
        """
        try:
            gesture_name, confidence = self.gesture_recognizer.recognize(frame)
            # Only accept gestures that have actions bound
            if gesture_name not in self.accepted_gestures:
                return 'no_gesture', confidence
            if self.debug:
                print(f"[HandTracker] Recognized gesture: {gesture_name} (confidence: {confidence:.3f})")
            return gesture_name, confidence
        except Exception as e:
            if self.debug:
                print(f"[HandTracker] Error in gesture recognition: {e}")
        return None
    
    def _extract_hand_region(self, frame: np.ndarray, landmarks: List[Tuple[int, int, float]]) -> Optional[np.ndarray]:
        """
        Extract hand region from frame based on landmarks.
        Uses a square crop with generous padding to better match training data.
        
        Args:
            frame: Original frame.
            landmarks: Hand landmarks.
            
        Returns:
            Cropped hand region or None if extraction fails.
        """
        if len(landmarks) == 0:
            return None
        
        # Get bounding box from landmarks
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Make the crop square (use the larger dimension)
        box_w = max_x - min_x
        box_h = max_y - min_y
        box_size = max(box_w, box_h)
        
        # Center the square on the hand
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # Add generous padding (50% of box size on each side)
        half_size = int(box_size * 0.75)
        
        h, w = frame.shape[:2]
        min_x = max(0, center_x - half_size)
        min_y = max(0, center_y - half_size)
        max_x = min(w, center_x + half_size)
        max_y = min(h, center_y + half_size)
        
        # Extract region
        hand_region = frame[min_y:max_y, min_x:max_x]
        
        # Ensure minimum size
        if hand_region.shape[0] < 50 or hand_region.shape[1] < 50:
            return None
            
        return hand_region
    
    def toggle_landmarks(self) -> bool:
        """
        Toggle landmark drawing on/off.
        
        Returns:
            bool: Current state of landmark drawing (True = on, False = off).
        """
        self.draw_landmarks = not self.draw_landmarks
        return self.draw_landmarks
    
    def release(self):
        """Release resources."""
        if self.detector:
            self.detector.close()
        if self.debug:
            print("[HandTracker] Released resources")
