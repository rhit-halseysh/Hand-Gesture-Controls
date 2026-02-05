"""
Hand tracking module using OpenCV and MediaPipe Tasks.
Detects hand landmarks and provides hand position data.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, List, Tuple
import os
import urllib.request


class HandTracker:
    """Tracks hand positions and landmarks using MediaPipe Tasks."""
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"
    
    def __init__(self, debug: bool = False):
        """
        Initialize hand tracker.
        
        Args:
            debug: If True, prints debug information.
        """
        self.debug = debug
        self.detector = None
        
        # Try to load the model, download if necessary
        self._setup_model()
        
        if self.debug:
            print("[HandTracker] Initialized")
    
    def _setup_model(self):
        """Setup hand detection model, downloading if necessary."""
        # Check if model file exists, if not download it
        if not os.path.exists(self.MODEL_PATH):
            if self.debug:
                print(f"[HandTracker] Downloading model from {self.MODEL_URL}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            if self.debug:
                print("[HandTracker] Model downloaded successfully")
        
        # Create hand landmarker with Tasks API
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        if self.debug:
            print("[HandTracker] Using MediaPipe Tasks API")
    
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
                    hand_data_list.append(hand_data)
                    
                    if self.debug:
                        print(f"[HandTracker] Detected {hand_label} hand - Confidence: {hand_confidence:.2f}")
                    
                    self._draw_landmarks(annotated_frame, landmark_list)
            
            return annotated_frame, hand_data_list if hand_data_list else None
        
        except Exception as e:
            if self.debug:
                print(f"[HandTracker] Fatal error in process_frame: {e}")
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
    
    def release(self):
        """Release resources."""
        if self.debug:
            print("[HandTracker] Released resources")
