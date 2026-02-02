"""
Action handler module.
Executes actions based on recognized gestures.
"""

import pyautogui
import time
from typing import Callable, Dict
from .gesture_recognizer import Gesture
from collections import deque
import numpy as np


class ActionHandler:
    """Handles gesture-to-action mapping and execution."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize action handler.
        
        Args:
            debug: If True, prints debug information.
        """
        self.debug = debug
        self.gesture_actions: Dict[Gesture, Callable] = {}
        self.palm_position_history = deque(maxlen=10)
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.1  # Seconds between scrolls
        
        # Register default actions
        self._register_default_actions()
        
        if self.debug:
            print("[ActionHandler] Initialized")
    
    def _register_default_actions(self):
        """Register default gesture-to-action mappings."""
        self.register_gesture_action(Gesture.SCROLL_UP, self._scroll_up)
        self.register_gesture_action(Gesture.SCROLL_DOWN, self._scroll_down)
        self.register_gesture_action(Gesture.OPEN_PALM, self._open_palm_action)
        self.register_gesture_action(Gesture.POINTING, self._pointing_action)
        
        if self.debug:
            print("[ActionHandler] Registered default actions")
    
    def register_gesture_action(self, gesture: Gesture, action: Callable):
        """
        Register a callback for a gesture.
        
        Args:
            gesture: Gesture enum.
            action: Callable that executes the action.
        """
        self.gesture_actions[gesture] = action
        if self.debug:
            print(f"[ActionHandler] Registered action for {gesture.value}")
    
    def handle_gesture(self, gesture: Gesture, hand_data: dict = None):
        """
        Execute action for a recognized gesture.
        
        Args:
            gesture: Recognized gesture.
            hand_data: Hand data dictionary (optional, for context).
        """
        if gesture in self.gesture_actions:
            if self.debug:
                print(f"[ActionHandler] Executing action for {gesture.value}")
            self.gesture_actions[gesture](hand_data)
        else:
            if self.debug:
                print(f"[ActionHandler] No action registered for {gesture.value}")
    
    def update_palm_position(self, palm_position: tuple):
        """
        Update palm position history for motion tracking.
        
        Args:
            palm_position: Tuple of (x, y) coordinates.
        """
        self.palm_position_history.append(palm_position)
    
    def detect_vertical_motion(self) -> str:
        """
        Detect vertical hand motion from position history.
        
        Returns:
            'up', 'down', or 'none'
        """
        if len(self.palm_position_history) < 3:
            return 'none'
        
        positions = list(self.palm_position_history)
        recent_y = positions[-1][1]
        previous_y = positions[0][1]
        
        diff = previous_y - recent_y  # Negative = moving up, positive = moving down
        
        # Threshold for motion detection (pixels)
        threshold = 20
        
        if diff > threshold:
            return 'up'
        elif diff < -threshold:
            return 'down'
        else:
            return 'none'
    
    # Default action implementations
    
    def _scroll_up(self, hand_data: dict = None):
        """Scroll up action."""
        current_time = time.time()
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return
        
        self.last_scroll_time = current_time
        if self.debug:
            print("[ActionHandler] Scrolling UP")
        pyautogui.scroll(3)  # Scroll up 3 clicks
    
    def _scroll_down(self, hand_data: dict = None):
        """Scroll down action."""
        current_time = time.time()
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return
        
        self.last_scroll_time = current_time
        if self.debug:
            print("[ActionHandler] Scrolling DOWN")
        pyautogui.scroll(-3)  # Scroll down 3 clicks
    
    def _open_palm_action(self, hand_data: dict = None):
        """Open palm action (placeholder)."""
        if self.debug:
            print("[ActionHandler] Open palm detected")
    
    def _pointing_action(self, hand_data: dict = None):
        """Pointing action (placeholder)."""
        if self.debug:
            print("[ActionHandler] Pointing detected")
    
    def move_mouse_to_hand(self, palm_position: tuple, frame_width: int, frame_height: int):
        """
        Move mouse to follow hand position (optional feature).
        
        Args:
            palm_position: Tuple of (x, y) in frame coordinates.
            frame_width: Width of the camera frame.
            frame_height: Height of the camera frame.
        """
        screen_width, screen_height = pyautogui.size()
        
        # Map frame coordinates to screen coordinates
        screen_x = int((palm_position[0] / frame_width) * screen_width)
        screen_y = int((palm_position[1] / frame_height) * screen_height)
        
        # Move mouse (with bounds checking)
        screen_x = max(0, min(screen_x, screen_width - 1))
        screen_y = max(0, min(screen_y, screen_height - 1))
        
        pyautogui.moveTo(screen_x, screen_y, duration=0.05)
        
        if self.debug:
            print(f"[ActionHandler] Mouse moved to ({screen_x}, {screen_y})")
