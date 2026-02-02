"""
Gesture recognition module.
Recognizes hand gestures from landmark data.
"""

from typing import Optional, List
from enum import Enum


class Gesture(Enum):
    """Enumeration of recognized gestures."""
    FIST = "fist"
    OPEN_PALM = "open_palm"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    POINTING = "pointing"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    UNKNOWN = "unknown"


class GestureRecognizer:
    """Recognizes gestures from hand landmarks."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize gesture recognizer.
        
        Args:
            debug: If True, prints debug information.
        """
        self.debug = debug
        if self.debug:
            print("[GestureRecognizer] Initialized")
    
    def recognize(self, hand_data: dict) -> Gesture:
        """
        Recognize a gesture from hand data.
        
        Args:
            hand_data: Dictionary containing 'landmarks' and 'fingers_up'.
            
        Returns:
            Recognized Gesture enum.
        """
        if not hand_data:
            return Gesture.UNKNOWN
        
        fingers_up = hand_data.get('fingers_up', [])
        landmarks = hand_data.get('landmarks', [])
        
        if not fingers_up or not landmarks:
            return Gesture.UNKNOWN
        
        # Count fingers up
        num_fingers_up = sum(fingers_up)
        
        # Recognize gestures
        gesture = self._classify_gesture(fingers_up, landmarks, num_fingers_up)
        
        if self.debug:
            print(f"[GestureRecognizer] Recognized gesture: {gesture.value} (fingers: {num_fingers_up})")
        
        return gesture
    
    def _classify_gesture(self, fingers_up: List[int], landmarks: List[tuple], num_fingers_up: int) -> Gesture:
        """
        Classify gesture based on finger positions.
        
        Args:
            fingers_up: List of 5 binary values indicating which fingers are up.
            landmarks: List of (x, y, z) landmark positions.
            num_fingers_up: Total number of fingers up.
            
        Returns:
            Recognized Gesture.
        """
        # Fist - no fingers up
        if num_fingers_up == 0:
            return Gesture.FIST
        
        # Open palm - all fingers up
        if num_fingers_up == 5:
            return Gesture.OPEN_PALM
        
        # Peace sign - index and middle up, others down
        if fingers_up == [0, 1, 1, 0, 0]:
            return Gesture.PEACE
        
        # Pointing - only index up
        if fingers_up == [0, 1, 0, 0, 0]:
            return Gesture.POINTING
        
        # Thumbs up - only thumb up
        if fingers_up == [1, 0, 0, 0, 0]:
            return Gesture.THUMBS_UP
        
        # Scroll gestures based on hand movement (placeholder for now)
        # These will be detected by motion tracking in the action handler
        
        return Gesture.UNKNOWN
    
    def get_all_gestures(self) -> List[str]:
        """Return list of all recognized gesture names."""
        return [gesture.value for gesture in Gesture]
