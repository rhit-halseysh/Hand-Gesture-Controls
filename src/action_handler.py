"""
Action handler module.
Executes actions based on recognized gestures.
"""

import pyautogui
import time
from typing import Callable, Dict
from collections import deque
import numpy as np
from src.mouse_controller import MouseController


class ActionHandler:
    """Handles gesture-to-action mapping and execution."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize action handler.
        
        Args:
            debug: If True, prints debug information.
        """
        self.debug = debug
        self.gesture_actions: Dict[str, Callable] = {}  # Changed to string keys
        self.palm_position_history = deque(maxlen=10)
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.1  # Seconds between scrolls
        self.last_gesture_time = 0
        self.gesture_cooldown = 2.0  # Seconds between gesture commands
        self.click_cooldown = 0.5  # Shorter cooldown for click gestures
        
        # Mouse control
        self.mouse_controller = MouseController(debug=debug, smoothing_frames=8, sensitivity=0.35)
        self.mouse_control_enabled = False
        
        # Click gestures that need shorter cooldown
        self.click_gestures = {'grabbing', 'little_finger', 'point'}
        
        # Register gesture mappings for YouTube Shorts
        self._register_youtube_actions()
        
        # if self.debug:
        #     print("[ActionHandler] Initialized")
    
    def _register_youtube_actions(self):
        """Register gesture-to-YouTube action mappings."""
        # Navigation gestures
        self.register_gesture_action('peace', self._next_short)  # Peace -> Next short
        self.register_gesture_action('peace_inverted', self._previous_short)  # Inverted peace -> Previous short
        self.register_gesture_action('palm', self._previous_short)  # Palm -> Previous short
        
        # Interaction gestures
        self.register_gesture_action('like', self._like_video)  # Like gesture -> Like video
        self.register_gesture_action('dislike', self._dislike_video)  # Dislike gesture -> Dislike video
        self.register_gesture_action('thumbs_up', self._like_video)  # Thumbs up -> Like video (alternative)
        
        # Comment actions
        self.register_gesture_action('call', self._open_comments)  # Call gesture -> Open comments
        self.register_gesture_action('mute', self._close_comments)  # Mute gesture -> Close comments
        self.register_gesture_action('stop', self._close_comments)  # Stop gesture -> Close comments
        
        # Number gestures for navigation
        # self.register_gesture_action('one', self._select_comment_box)  # One finger -> Select comment box
        self.register_gesture_action('two_up', self._post_comment)  # Two fingers up -> Post comment
        
        # Special gestures
        self.register_gesture_action('ok', self._remove_like_dislike)  # OK gesture -> Remove like/dislike
        self.register_gesture_action('fist', self._pause_play)  # Open palm -> Pause/Play
        # self.register_gesture_action('fist', self._take_screenshot)  # Fist -> Take screenshot
        
        # Heart gestures for liking
        self.register_gesture_action('hand_heart', self._like_video)  # Heart -> Like video
        self.register_gesture_action('hand_heart2', self._like_video)  # Heart variant -> Like video
        
        # Rock/metal gesture for special actions
        self.register_gesture_action('rock', self._fullscreen_toggle)  # Rock gesture -> Fullscreen
        
        # Mouse click gestures (work when mouse control is enabled)
        self.register_gesture_action('one', self._left_click)  # Grabbing -> Left click
        self.register_gesture_action('little_finger', self._right_click)  # Little finger -> Right click
        self.register_gesture_action('point', self._toggle_drag)  # Point -> Toggle drag
        
        if self.debug:
            print(f"[ActionHandler] Registered {len(self.gesture_actions)} gesture actions")
    
    def register_gesture_action(self, gesture: str, action: Callable):
        """
        Register a callback for a gesture.
        
        Args:
            gesture: Gesture name string.
            action: Callable that executes the action.
        """
        self.gesture_actions[gesture] = action
        if self.debug:
            print(f"[ActionHandler] Registered action for {gesture}")
    
    def handle_gesture(self, gesture: str, hand_data: dict = None):
        """
        Execute action for a recognized gesture.
        
        Args:
            gesture: Recognized gesture name string.
            hand_data: Hand data dictionary (optional, for context).
        """
        current_time = time.time()
        
        # Use shorter cooldown for click gestures
        cooldown = self.click_cooldown if gesture in self.click_gestures else self.gesture_cooldown
        
        if current_time - self.last_gesture_time < cooldown:
            if self.debug:
                print(f"[ActionHandler] Gesture on cooldown - skipping {gesture}")
            return
        
        if gesture in self.gesture_actions:
            self.last_gesture_time = current_time
            if self.debug:
                print(f"[ActionHandler] Executing action for {gesture}")
            self.gesture_actions[gesture](hand_data)
        else:
            if self.debug:
                print(f"[ActionHandler] No action registered for {gesture}")
    
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
    
    # YouTube Shorts action implementations (matching content.js)
    
    def _next_short(self, hand_data: dict = None):
        """Next short - Ctrl+Alt+Y."""
        if self.debug:
            print("[ActionHandler] Next short gesture - Ctrl+Alt+Y")
        pyautogui.hotkey('ctrl', 'alt', 'y')
    
    def _previous_short(self, hand_data: dict = None):
        """Previous short - Ctrl+Alt+T."""
        if self.debug:
            print("[ActionHandler] Previous short gesture - Ctrl+Alt+T")
        pyautogui.hotkey('ctrl', 'alt', 't')
    
    def _like_video(self, hand_data: dict = None):
        """Like video - Ctrl+Alt+L."""
        if self.debug:
            print("[ActionHandler] Like video gesture - Ctrl+Alt+L")
        pyautogui.hotkey('ctrl', 'alt', 'l')
    
    def _dislike_video(self, hand_data: dict = None):
        """Dislike video - Ctrl+Alt+D."""
        if self.debug:
            print("[ActionHandler] Dislike video gesture - Ctrl+Alt+D")
        pyautogui.hotkey('ctrl', 'alt', 'd')
    
    def _remove_like_dislike(self, hand_data: dict = None):
        """Remove like/dislike - Ctrl+Alt+R."""
        if self.debug:
            print("[ActionHandler] Remove like/dislike gesture - Ctrl+Alt+R")
        pyautogui.hotkey('ctrl', 'alt', 'r')
    
    def _open_comments(self, hand_data: dict = None):
        """Open comments - Ctrl+Alt+C."""
        if self.debug:
            print("[ActionHandler] Open comments gesture - Ctrl+Alt+C")
        pyautogui.hotkey('ctrl', 'alt', 'c')
    
    def _close_comments(self, hand_data: dict = None):
        """Close comments - Ctrl+Alt+X."""
        if self.debug:
            print("[ActionHandler] Close comments gesture - Ctrl+Alt+X")
        pyautogui.hotkey('ctrl', 'alt', 'x')
    
    def _select_comment_box(self, hand_data: dict = None):
        """Select comment box - Ctrl+Alt+B."""
        if self.debug:
            print("[ActionHandler] Select comment box gesture - Ctrl+Alt+B")
        pyautogui.hotkey('ctrl', 'alt', 'b')
    
    def _post_comment(self, hand_data: dict = None):
        """Post comment - Ctrl+Alt+V."""
        if self.debug:
            print("[ActionHandler] Post comment gesture - Ctrl+Alt+V")
        pyautogui.hotkey('ctrl', 'alt', 'v')
    
    def _pause_play(self, hand_data: dict = None):
        """Pause/Play - Space bar."""
        if self.debug:
            print("[ActionHandler] Pause/Play gesture - Space")
        pyautogui.press('space')
    
    def _take_screenshot(self, hand_data: dict = None):
        """Take screenshot action."""
        if self.debug:
            print("[ActionHandler] Take screenshot gesture")
        pyautogui.hotkey('win', 'shift', 's')
    
    def _fullscreen_toggle(self, hand_data: dict = None):
        """Toggle fullscreen - F key."""
        if self.debug:
            print("[ActionHandler] Fullscreen toggle gesture - F")
        pyautogui.press('f')
    
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
    
    def move_mouse_to_hand(self, palm_position: tuple, frame_width: int, frame_height: int):
        """
        Move mouse to follow hand position with smoothing.
        
        Args:
            palm_position: Tuple of (x, y) in frame coordinates.
            frame_width: Width of the camera frame.
            frame_height: Height of the camera frame.
        """
        if not self.mouse_control_enabled:
            return
        
        # Set frame dimensions if not already set
        if self.mouse_controller.frame_width is None:
            self.mouse_controller.set_frame_dimensions(frame_width, frame_height)
        
        # Update mouse position
        self.mouse_controller.update_mouse_position(palm_position[0], palm_position[1])
    
    def enable_mouse_control(self):
        """Enable mouse control."""
        self.mouse_control_enabled = True
        if self.debug:
            print("[ActionHandler] Mouse control ENABLED")
    
    def disable_mouse_control(self):
        """Disable mouse control."""
        self.mouse_control_enabled = False
        self.mouse_controller.reset_smoothing()
        if self.debug:
            print("[ActionHandler] Mouse control DISABLED")
    
    def toggle_mouse_control(self):
        """Toggle mouse control on/off."""
        if self.mouse_control_enabled:
            self.disable_mouse_control()
        else:
            self.enable_mouse_control()
        return self.mouse_control_enabled
    
    # Mouse click actions
    def _left_click(self, hand_data: dict = None):
        """Perform left click (only when mouse control is enabled)."""
        if not self.mouse_control_enabled:
            if self.debug:
                print("[ActionHandler] Left click ignored - mouse control disabled")
            return
        
        self.mouse_controller.left_click()
    
    def _right_click(self, hand_data: dict = None):
        """Perform right click (only when mouse control is enabled)."""
        if not self.mouse_control_enabled:
            if self.debug:
                print("[ActionHandler] Right click ignored - mouse control disabled")
            return
        
        self.mouse_controller.right_click()
    
    def _toggle_drag(self, hand_data: dict = None):
        """Toggle click-and-hold drag (only when mouse control is enabled)."""
        if not self.mouse_control_enabled:
            if self.debug:
                print("[ActionHandler] Drag toggle ignored - mouse control disabled")
            return
        
        if self.mouse_controller.is_drag_active():
            self.mouse_controller.stop_drag()
        else:
            self.mouse_controller.start_drag()
