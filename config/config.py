"""
Configuration file for Hand Gesture Tracker.
Adjust these settings to customize gesture tracking behavior.
"""

# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Hand tracking settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Gesture recognition settings
GESTURE_SMOOTHING = 5  # Number of frames to smooth gesture detection
MOTION_DETECTION_THRESHOLD = 20  # Pixels

# Action settings
SCROLL_COOLDOWN = 0.1  # Seconds between scroll actions
SCROLL_AMOUNT = 3  # Number of scroll clicks per action

# UI settings
UI_WIDTH = 400
UI_HEIGHT = 300
DEBUG_MODE = True  # Set to False to reduce console output

# Gesture Actions
GESTURE_ACTIONS = {
    'scroll_up': 'scroll',
    'scroll_down': 'scroll',
    'open_palm': 'ready',
    'pointing': 'point',
    'peace': 'peace',
    'thumbs_up': 'thumbs_up',
    'fist': 'fist'
}
