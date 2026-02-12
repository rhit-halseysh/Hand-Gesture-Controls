"""
UI module for gesture tracker control.
Provides a simple interface with toggle switch and debug display.
"""

import tkinter as tk
from tkinter import ttk, Label
import threading
from typing import Callable, Optional


class GestureTrackerUI:
    """Simple UI for controlling gesture tracker."""
    
    def __init__(self, width: int = 400, height: int = 300, debug: bool = False):
        """
        Initialize UI.
        
        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            debug: If True, shows debug information.
        """
        self.debug = debug
        self.window = tk.Tk()
        self.window.title("Hand Gesture Tracker")
        self.window.geometry(f"{width}x{height}")
        self.window.resizable(False, False)
        
        self.is_tracking = False
        self.on_toggle_callback: Optional[Callable[[bool], None]] = None
        
        # Create UI elements
        self._create_widgets()
        
        if self.debug:
            print("[GestureTrackerUI] Initialized")
    
    def _create_widgets(self):
        """Create UI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Hand Gesture Tracker",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Status: STOPPED",
            font=("Arial", 12),
            foreground="red"
        )
        self.status_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Toggle button
        self.toggle_button = ttk.Button(
            main_frame,
            text="Start Tracking",
            command=self._toggle_tracking
        )
        self.toggle_button.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Separator
        separator = ttk.Separator(main_frame, orient="horizontal")
        separator.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Info section
        info_label = ttk.Label(
            main_frame,
            text="Current Gesture:",
            font=("Arial", 10, "bold")
        )
        info_label.grid(row=4, column=0, sticky=tk.W, pady=5)
        
        self.gesture_label = ttk.Label(
            main_frame,
            text="None",
            font=("Arial", 10),
            foreground="blue"
        )
        self.gesture_label.grid(row=4, column=1, sticky=tk.E, pady=5)
        
        # Confidence level info
        confidence_label = ttk.Label(
            main_frame,
            text="Confidence:",
            font=("Arial", 10, "bold")
        )
        confidence_label.grid(row=5, column=0, sticky=tk.W, pady=5)
        
        self.confidence_label = ttk.Label(
            main_frame,
            text="0%",
            font=("Arial", 10),
            foreground="orange"
        )
        self.confidence_label.grid(row=5, column=1, sticky=tk.E, pady=5)
        
        # Confidence level info
        confidence_label = ttk.Label(
            main_frame,
            text="Confidence:",
            font=("Arial", 10, "bold")
        )
        confidence_label.grid(row=5, column=0, sticky=tk.W, pady=5)
        
        self.confidence_label = ttk.Label(
            main_frame,
            text="0%",
            font=("Arial", 10),
            foreground="orange"
        )
        self.confidence_label.grid(row=5, column=1, sticky=tk.E, pady=5)
        
        # Frame rate info
        fps_label = ttk.Label(
            main_frame,
            text="FPS:",
            font=("Arial", 10, "bold")
        )
        fps_label.grid(row=6, column=0, sticky=tk.W, pady=5)
        
        self.fps_label = ttk.Label(
            main_frame,
            text="0",
            font=("Arial", 10),
            foreground="green"
        )
        self.fps_label.grid(row=6, column=1, sticky=tk.E, pady=5)
        
        # Hand count info
        hands_label = ttk.Label(
            main_frame,
            text="Hands Detected:",
            font=("Arial", 10, "bold")
        )
        hands_label.grid(row=7, column=0, sticky=tk.W, pady=5)
        
        self.hands_label = ttk.Label(
            main_frame,
            text="0",
            font=("Arial", 10),
            foreground="purple"
        )
        self.hands_label.grid(row=7, column=1, sticky=tk.E, pady=5)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="5")
        instructions_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        instructions_text = """
• Click "Start Tracking" to begin
• Use hand gestures to scroll
• Fist = No action
• Open Palm = Ready
• Motion detection for scrolling
        """
        
        instructions_label = ttk.Label(
            instructions_frame,
            text=instructions_text,
            font=("Arial", 9),
            justify=tk.LEFT
        )
        instructions_label.pack(side=tk.LEFT)
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
    
    def _toggle_tracking(self):
        """Handle toggle button click."""
        self.is_tracking = not self.is_tracking
        
        if self.is_tracking:
            self.status_label.config(text="Status: RUNNING", foreground="green")
            self.toggle_button.config(text="Stop Tracking")
            if self.debug:
                print("[GestureTrackerUI] Tracking started")
        else:
            self.status_label.config(text="Status: STOPPED", foreground="red")
            self.toggle_button.config(text="Start Tracking")
            if self.debug:
                print("[GestureTrackerUI] Tracking stopped")
        
        if self.on_toggle_callback:
            self.on_toggle_callback(self.is_tracking)
    
    def set_toggle_callback(self, callback: Callable[[bool], None]):
        """
        Set callback for toggle button.
        
        Args:
            callback: Function called with bool (True = start, False = stop).
        """
        self.on_toggle_callback = callback
    
    def update_gesture(self, gesture_name: str, confidence: float = 0.0):
        """Update displayed gesture and confidence."""
        self.gesture_label.config(text=gesture_name)
        
        # Update confidence with color coding
        confidence_percent = confidence * 100
        confidence_text = f"{confidence_percent:.1f}%"
        
        # Color code confidence level
        if confidence >= 0.8:
            color = "green"  # High confidence
        elif confidence >= 0.6:
            color = "orange"  # Medium confidence  
        else:
            color = "red"  # Low confidence
            
        self.confidence_label.config(text=confidence_text, foreground=color)
    
    def update_fps(self, fps: float):
        """Update displayed FPS."""
        self.fps_label.config(text=f"{fps:.1f}")
    
    def update_hands_count(self, count: int):
        """Update displayed hand count."""
        self.hands_label.config(text=str(count))
    
    def is_active(self) -> bool:
        """Check if tracking is active."""
        return self.is_tracking
    
    def process_events(self):
        """Process pending UI events (non-blocking)."""
        try:
            self.window.update()
        except tk.TclError:
            return False
        return True
    
    def run(self):
        """Run the UI event loop (blocking)."""
        self.window.mainloop()
    
    def close(self):
        """Close the UI."""
        try:
            self.window.quit()
            self.window.destroy()
        except tk.TclError:
            pass
        if self.debug:
            print("[GestureTrackerUI] Closed")
