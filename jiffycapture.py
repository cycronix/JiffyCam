"""
jiffycapture.py: Video capture functionality for JiffyCam

This module provides the core video capture functionality for the JiffyCam application.
"""

import time
import os
import threading
import gc
from queue import Queue
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import cv2

from jiffyput import jiffyput
from jiffyconfig import JiffyConfig


class VideoCapture:
    def __init__(self):
        """Initialize video capture functionality."""
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.capture_thread: Optional[threading.Thread] = None
        self.config_manager = JiffyConfig()
        self.config = self.config_manager.config
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        self.last_error = None
        self.error_event = threading.Event()  # Add error event
        # Add status tracking variables
        self.current_fps = 0
        self.current_width = 0
        self.current_height = 0
        self.last_save_time = 0  # Track last save time
        self.save_status = ""  # Track save status message
        self.skip_first_save = True  # Flag to skip the first save
        self.image_just_saved = False  # Flag to track when an image was just saved
        self.image_saved_time = 0  # Track when the image was saved
        
    def handle_error(self, error_msg: str):
        """Handle errors by setting error message and stopping capture"""
        self.last_error = error_msg
        self.error_event.set()
        self.stop_event.set()

    def send_frame(self, cam_name: str, frame, ftime: float, session: str):
        """Send frame to server and optionally save to disk."""
        try:
            # Call jiffyput function
            result = jiffyput(cam_name, frame, ftime, session, self.config.get('data_dir', 'JiffyData'))
            # Update class attributes
            self.image_just_saved = True
            self.image_saved_time = self.last_save_time = time.time()
            self.save_status = f"Frame saved: {datetime.fromtimestamp(ftime).strftime('%Y-%m-%d %H:%M:%S')}"

            if result is None:
                # If jiffyput returned None, there was an error
                self.handle_error("Error in jiffyput processing")
                return None
                
        except Exception as e:
            self.handle_error(f"Error sending frame: {str(e)}")
            return None

        return result

    def capture_video_loop(self, source, cam_name: str, cam_width: int, cam_height: int, save_interval: int, session: str):
        """Initialize and run video capture loop."""
        
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.handle_error(f"Error: Could not open video source: {source}")
            return

        # Configure camera
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cam_width != 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        if cam_height != 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        save_frame = False  # Initialize as False to skip first frame
        last_fps_time = time.time()
        self.last_save_time = 0  # Initialize last save time
        frames_this_second = 0
        current_fps = 0
        consecutive_failures = 0  # Track consecutive read failures
        frame_count_since_start = 0  # Counter for frames since starting capture

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                consecutive_failures = 0  # Reset failure counter on success
                current_time = time.time()
                frame_count_since_start += 1  # Increment frame counter

                # Regular interval saving for subsequent frames
                if save_interval > 0 and (current_time - self.last_save_time) >= save_interval:
                    save_frame = True
                    self.last_save_time = current_time

                # Update FPS calculation
                frames_this_second += 1
                if current_time - last_fps_time >= 1.0:
                    current_fps = frames_this_second
                    frames_this_second = 0
                    last_fps_time = current_time

                if save_frame:
                    frame = self.send_frame(cam_name, frame, time.time(), session)
                if self.error_event.is_set():  # Check if error occurred in send_frame
                    break
                save_frame = False
                self.frame_count += 1

                # Update frame queue
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, current_fps, width, height))
                else:
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                        self.frame_queue.put((frame, current_fps, width, height))
                    except:
                        pass

                time.sleep(0.01)  # Small delay to prevent overwhelming the system
            else:
                consecutive_failures += 1
                if consecutive_failures >= 10:  # Stop after 10 consecutive failures
                    self.handle_error("Video capture failed repeatedly - stopping capture")
                    break
                self.last_error = "Video capture read failed"
                time.sleep(1)

        cap.release()

    def start_capture(self, resolution_str=None, session=None, cam_device=None, cam_name=None, 
                      save_interval=None, device_aliases=None, selected_device_alias=None):
        """Start video capture with the given parameters.
        
        Args:
            resolution_str: Resolution string in format "widthxheight" or a key from RESOLUTIONS
            session: Session name
            cam_device: Camera device identifier
            cam_name: Camera name
            save_interval: Interval between saved frames
            device_aliases: Dictionary of device aliases
            selected_device_alias: Currently selected device alias
        """
        # Set up configuration with provided or default values
        if cam_device is None:
            cam_device = self.config.get('cam_device', '0')
        if session is None:
            session = self.config.get('session', 'Default')
        if cam_name is None:
            cam_name = self.config.get('cam_name', 'cam0')
        if save_interval is None:
            save_interval = int(self.config.get('save_interval', 60))
        if device_aliases is None:
            device_aliases = self.config.get('device_aliases', {'USB0': '0', 'USB1': '1', 'Default': '0'})
        
        # Process resolution
        if resolution_str is None:
            resolution_str = self.config.get('resolution', '1920x1080')
            
        # Create configuration dictionary
        config = {
            'cam_device': selected_device_alias or cam_device,
            'cam_name': cam_name,
            'save_interval': save_interval,
            'device_aliases': device_aliases
        }
        
        # Preserve data_dir if it exists in the current config
        if hasattr(self, 'config') and isinstance(self.config, dict) and 'data_dir' in self.config:
            config['data_dir'] = self.config['data_dir']
        
        # Add resolution to config
        config['resolution'] = resolution_str
        
        # Save configuration
        self.config = config
        self.config_manager.save_config(self.config)
        
        # Parse resolution string for capture thread
        try:
            width, height = map(int, resolution_str.split('x'))
        except (ValueError, AttributeError, IndexError):
            # Default to 1080p if parsing fails
            width, height = 1920, 1080
        
        # Start capture thread
        self.start_capture_thread(
            cam_device=cam_device,
            width=width,
            height=height,
            session=session,
            cam_name=cam_name,
            save_interval=save_interval
        )

    def stop_capture(self):
        """Stop the capture thread."""
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass

        try:
            # Release PyTorch resources explicitly
            if hasattr(self, 'model') and self.model is not None:
                self.model = None
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Warning during cleanup: {e}")
            # Continue shutdown despite errors

    def start_capture_thread(self, cam_device, width, height, session, cam_name, save_interval):
        """Start video capture in a separate thread."""
        if self.capture_thread and self.capture_thread.is_alive():
            return

        self.stop_event.clear()
        self.error_event.clear()  # Clear error event
        self.frame_count = 0
        self.last_error = None
        self.current_fps = 0
        self.current_width = 0
        self.current_height = 0
        self.last_save_time = 0
        self.save_status = ""
        self.skip_first_save = True  # Reset the skip flag when starting capture
        self.image_just_saved = False  # Reset the image_just_saved flag
        self.image_saved_time = 0  # Reset the image_saved_time variable
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self.capture_video_loop,
            args=(
                cam_device,
                cam_name,
                width,
                height,
                int(save_interval),
                session
            ),
            daemon=True
        )
        self.capture_thread.start()

    def get_frame(self) -> Tuple:
        """Get the latest frame from the queue.
        
        Returns:
            Tuple of (frame, fps, width, height) or None if queue is empty
        """
        if not self.frame_queue.empty():
            try:
                return self.frame_queue.get_nowait()
            except:
                pass
        return None, 0, 0, 0

    def is_capturing(self) -> bool:
        """Check if capture is currently running.
        
        Returns:
            bool: True if capture thread is running, False otherwise
        """
        return self.capture_thread is not None and self.capture_thread.is_alive() 