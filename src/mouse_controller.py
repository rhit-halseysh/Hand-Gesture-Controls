"""
Mouse controller module.
Maps hand position to mouse movement with smoothing.
"""

import pyautogui
import numpy as np
from collections import deque
from typing import Tuple, Optional


class MouseController:
    """Controls mouse movement based on hand tracking."""
    
    def __init__(self, debug: bool = False, smoothing_frames: int = 5, sensitivity: float = 1.5):
        """
        Initialize mouse controller.
        
        Args:
            debug: If True, prints debug information.
            smoothing_frames: Number of frames to average for smooth movement.
            sensitivity: Mouse movement sensitivity multiplier.
        """
        self.debug = debug
        self.smoothing_frames = smoothing_frames
        self.sensitivity = sensitivity
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Position smoothing buffer
        self.position_history = deque(maxlen=smoothing_frames)
        
        # Dead zone to ignore micro-movements
        self.dead_zone_threshold = 5  # pixels
        
        # Frame dimensions (will be set when first frame is processed)
        self.frame_width = None
        self.frame_height = None
        
        # Movement boundaries (to avoid edge jitter and corners)
        self.boundary_margin = 0.15  # 15% margin on each side to stay away from corners
        
        # Screen safety margin to avoid PyAutoGUI fail-safe corners
        self.screen_margin = 50  # Keep mouse 50px away from screen edges
        
        # Disable PyAutoGUI failsafe to prevent corner interruptions
        # (We handle safety with our own boundary checks)
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0  # No pause between commands for smooth movement
        
        # Drag state
        self.is_dragging = False
        
        if self.debug:
            print(f"[MouseController] Initialized")
            print(f"[MouseController] Screen size: {self.screen_width}x{self.screen_height}")
            print(f"[MouseController] Smoothing: {smoothing_frames} frames")
            print(f"[MouseController] Sensitivity: {sensitivity}x")
    
    def set_frame_dimensions(self, width: int, height: int):
        """
        Set the camera frame dimensions for coordinate mapping.
        
        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.frame_width = width
        self.frame_height = height
        
        # Calculate usable area (excluding margins)
        self.min_x = int(width * self.boundary_margin)
        self.max_x = int(width * (1 - self.boundary_margin))
        self.min_y = int(height * self.boundary_margin)
        self.max_y = int(height * (1 - self.boundary_margin))
        
        if self.debug:
            print(f"[MouseController] Frame dimensions set: {width}x{height}")
            print(f"[MouseController] Active area: ({self.min_x}, {self.min_y}) to ({self.max_x}, {self.max_y})")
    
    def update_mouse_position(self, hand_x: int, hand_y: int):
        """
        Update mouse position based on hand coordinates.
        
        Args:
            hand_x: X coordinate of hand position in frame.
            hand_y: Y coordinate of hand position in frame.
        """
        if self.frame_width is None or self.frame_height is None:
            if self.debug:
                print("[MouseController] Warning: Frame dimensions not set")
            return
        
        # Clamp hand position to active area
        clamped_x = max(self.min_x, min(self.max_x, hand_x))
        clamped_y = max(self.min_y, min(self.max_y, hand_y))
        
        # Normalize to 0-1 range within active area
        normalized_x = (clamped_x - self.min_x) / (self.max_x - self.min_x)
        normalized_y = (clamped_y - self.min_y) / (self.max_y - self.min_y)
        
        # Map to screen coordinates
        screen_x = int(normalized_x * self.screen_width)
        screen_y = int(normalized_y * self.screen_height)
        
        # Add to smoothing buffer
        self.position_history.append((screen_x, screen_y))
        
        # Calculate smoothed position using weighted moving average
        # (recent positions have more weight to reduce lag)
        if len(self.position_history) >= 3:
            positions = list(self.position_history)
            weights = np.linspace(0.5, 1.5, len(positions))  # Recent frames weighted more
            weights = weights / weights.sum()  # Normalize
            
            avg_x = int(sum(pos[0] * w for pos, w in zip(positions, weights)))
            avg_y = int(sum(pos[1] * w for pos, w in zip(positions, weights)))
            
            # Get current mouse position
            current_x, current_y = pyautogui.position()
            
            # Calculate distance to target
            dist_x = avg_x - current_x
            dist_y = avg_y - current_y
            
            # Apply dead zone to ignore micro-movements
            if abs(dist_x) < self.dead_zone_threshold and abs(dist_y) < self.dead_zone_threshold:
                return
            
            # Move directly to smoothed position (no momentum/spring effect)
            new_x = avg_x
            new_y = avg_y
            
            # Clamp to safe screen bounds (away from corners to avoid fail-safe)
            new_x = max(self.screen_margin, min(self.screen_width - self.screen_margin - 1, new_x))
            new_y = max(self.screen_margin, min(self.screen_height - self.screen_margin - 1, new_y))
            
            pyautogui.moveTo(new_x, new_y, duration=0)
            
            if self.debug and len(self.position_history) % 30 == 0:
                print(f"[MouseController] Hand: ({hand_x}, {hand_y}) -> Screen: ({new_x}, {new_y})")
    
    def reset_smoothing(self):
        """Clear the position history buffer."""
        self.position_history.clear()
        if self.debug:
            print("[MouseController] Smoothing buffer reset")
    
    def set_sensitivity(self, sensitivity: float):
        """
        Set mouse movement sensitivity.
        
        Args:
            sensitivity: Sensitivity multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower).
        """
        self.sensitivity = max(0.1, min(5.0, sensitivity))
        if self.debug:
            print(f"[MouseController] Sensitivity set to {self.sensitivity}x")
    
    def set_smoothing(self, frames: int):
        """
        Set smoothing buffer size.
        
        Args:
            frames: Number of frames to average (1 = no smoothing, higher = smoother).
        """
        self.smoothing_frames = max(1, min(30, frames))
        self.position_history = deque(maxlen=self.smoothing_frames)
        if self.debug:
            print(f"[MouseController] Smoothing set to {self.smoothing_frames} frames")
    
    def left_click(self):
        """Perform a left mouse click."""
        pyautogui.click()
        if self.debug:
            print("[MouseController] Left click")
    
    def right_click(self):
        """Perform a right mouse click."""
        pyautogui.rightClick()
        if self.debug:
            print("[MouseController] Right click")
    
    def start_drag(self):
        """Start click-and-hold for dragging."""
        if not self.is_dragging:
            pyautogui.mouseDown()
            self.is_dragging = True
            if self.debug:
                print("[MouseController] Drag started")
    
    def stop_drag(self):
        """Release mouse button to stop dragging."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
            if self.debug:
                print("[MouseController] Drag stopped")
    
    def is_drag_active(self) -> bool:
        """Check if currently dragging."""
        return self.is_dragging
