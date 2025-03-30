"""
vidcap_streamlit: Streamlit-based video capture utility

A modern version of vidcap that uses Streamlit for the UI to capture video 
from a camera and send it to a CloudTurbine (CT) server.
"""

import cv2
import time
import sys
import urllib3
import yaml
import os
import streamlit as st
import threading
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from queue import Queue
from datetime import datetime, timedelta, time as datetime_time
import glob
from collections import OrderedDict
import traceback

from jiffydetect import detect
from jiffyput import jiffyput  # Import the jiffyput module
from jiffyget import jiffyget as external_jiffyget, get_timestamp_range as external_get_timestamp_range  # Import the jiffyget functions

# Initialize HTTP pool manager
http = urllib3.PoolManager()

# Add a representer for OrderedDict to maintain order in YAML
yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))

# Add a constructor to load mappings as OrderedDict
yaml.add_constructor(yaml.resolver.Resolver.DEFAULT_MAPPING_TAG, 
                    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

# Configuration flags
SHOW_RESOLUTION_SETTING = False  # Set to True to show the Resolution setting in the sidebar

# Common camera resolutions
RESOLUTIONS = {
    "4K (3840x2160)": (3840, 2160),
    "1080p (1920x1080)": (1920, 1080),
    "720p (1280x720)": (1280, 720),
    "480p (854x480)": (854, 480),
    "360p (640x360)": (640, 360),
    "Default (0x0)": (0, 0)
}

# Add this right after your imports, before any functions are defined

# Initialize session state if running for the first time
if 'st' in globals() and hasattr(st, 'session_state'):
    # Initialize slider-related state variables

    if 'slider_currently_being_dragged' not in st.session_state:
        st.session_state.slider_currently_being_dragged = False

class VideoCapture:
    def __init__(self):
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.capture_thread: Optional[threading.Thread] = None
        self.yaml_file = 'jiffycam.yaml'
        self.config = self.load_config()
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
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file if it exists."""
        default_config = {
            'cam_device': '0',
            'session': 'Default',  # Default session value, but not saved to YAML anymore
            'cam_name': 'cam0',
            'resolution': '1920x1080',  # Combined resolution field
            'save_interval': 60,  # Changed to integer default
            'data_dir': 'JiffyData',  # Default data directory
            'device_aliases': OrderedDict([   # Use OrderedDict for default device aliases
                ('USB0', '0'),
                ('USB1', '1'),
                ('Default', '0')
            ])
        }
        
        if os.path.exists(self.yaml_file):
            try:
                with open(self.yaml_file, 'r') as file:
                    # Use safe_load which will now use our custom constructor for mappings
                    config = yaml.safe_load(file)
                    
                    # Remove debug output
                    # print(f"Loaded config from {self.yaml_file}: {config}")
                    
                    if config:
                        # Ensure save_interval is an integer
                        if 'save_interval' in config:
                            config['save_interval'] = int(config['save_interval'])
                        
                        # Ensure device_aliases exists and is an OrderedDict
                        if 'device_aliases' not in config:
                            config['device_aliases'] = default_config['device_aliases']
                        
                        # Handle legacy config with separate width and height
                        if 'cam_width' in config and 'cam_height' in config and 'resolution' not in config:
                            config['resolution'] = f"{config['cam_width']}x{config['cam_height']}"
                            # Remove old fields
                            config.pop('cam_width', None)
                            config.pop('cam_height', None)
                            
                        # Handle legacy cam_path key
                        if 'cam_path' in config and 'cam_device' not in config:
                            config['cam_device'] = config.pop('cam_path')
                        
                        # Remove debug output
                        # print(f"Final config: {config}")
                            
                        return config
            except Exception as e:
                self.last_error = f"Error loading configuration: {str(e)}"
                # Remove debug output
                # print(f"Error loading configuration: {str(e)}")
        return default_config

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to YAML file."""
        try:
            # Create a copy to avoid modifying the original
            config = config.copy()
            
            # Ensure data_dir is preserved if it exists in the current config
            if 'data_dir' not in config and hasattr(self, 'config') and isinstance(self.config, dict) and 'data_dir' in self.config:
                config['data_dir'] = self.config['data_dir']
            
            # Convert device_aliases to OrderedDict to preserve order
            if 'device_aliases' in config and isinstance(config['device_aliases'], dict):
                config['device_aliases'] = OrderedDict(config['device_aliases'])
            
            with open(self.yaml_file, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            self.last_error = f"Failed to save configuration: {str(e)}"

    def handle_error(self, error_msg: str):
        """Handle errors by setting error message and stopping capture"""
        self.last_error = error_msg
        self.error_event.set()
        self.stop_event.set()

    def send_frame(self, cam_name: str, frame, ftime: float, save_frame: bool, session: str):
        """Send frame to server and optionally save to disk."""
        try:
            # Create state dictionary to pass to jiffyput
            state = {
                'image_just_saved': self.image_just_saved,
                'image_saved_time': self.image_saved_time,
                'last_save_time': self.last_save_time,
                'save_status': self.save_status,
                'last_error': self.last_error
            }
            
            # Call jiffyput function
            result = jiffyput(cam_name, frame, ftime, save_frame, session, self.config, state)
            
            # Update class attributes from state dictionary
            if result is not None:
                self.image_just_saved = state.get('image_just_saved', self.image_just_saved)
                self.image_saved_time = state.get('image_saved_time', self.image_saved_time)
                self.last_save_time = state.get('last_save_time', self.last_save_time)
                self.save_status = state.get('save_status', self.save_status)
                
                # If there was an error, handle it
                if 'last_error' in state and state['last_error'] is not None:
                    self.handle_error(state['last_error'])
                    return None
            else:
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
                if( save_interval > 0 and (current_time - self.last_save_time) >= save_interval):
                    save_frame = True
                    self.last_save_time = current_time

                # Update FPS calculation
                frames_this_second += 1
                if current_time - last_fps_time >= 1.0:
                    current_fps = frames_this_second
                    frames_this_second = 0
                    last_fps_time = current_time

                frame = self.send_frame(cam_name, frame, time.time(), save_frame, session)
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

    def start_capture(self, resolution_str=None):
        """Start video capture with the given resolution"""
        # Initialize state variables
        # st.session_state.rt_capture = True  # This is now handled in toggle_rt_capture
        #print(f"Starting capture: {st.session_state.rt_capture}")
        st.session_state.rt_active = True
        st.session_state.browsing_saved_images = False
        
        # Reset date to current date
        current_time = datetime.now()
        st.session_state.date = current_time.date()
        st.session_state.browsing_date = current_time.date()
        
        # Set the hour, minute, second values to current time
        st.session_state.hour = current_time.hour
        st.session_state.minute = current_time.minute
        st.session_state.second = current_time.second
        
        # Update time_slider to current time
        st.session_state.time_slider = datetime_time(
            current_time.hour,
            current_time.minute,
            current_time.second
        )
        
        # Get configuration values
        cam_device = st.session_state.cam_device
        session = st.session_state.session
        cam_name = st.session_state.cam_name
        save_interval = st.session_state.save_interval
        device_aliases = st.session_state.device_aliases
        
        # Get the selected device alias (key) instead of the device path (value)
        selected_device_alias = st.session_state.selected_device_alias
        
        # Process resolution
        if resolution_str is None:
            resolution = st.session_state.resolution
            # Check if resolution is a key in RESOLUTIONS dictionary
            if resolution in RESOLUTIONS:
                width, height = RESOLUTIONS[resolution]
                resolution_str = f"{width}x{height}"
            else:
                # Assume resolution is already in the format "widthxheight"
                resolution_str = resolution
                
                # Validate format
                if not isinstance(resolution_str, str) or 'x' not in resolution_str:
                    resolution_str = "1920x1080"  # Default if invalid
        
        # Create configuration dictionary
        config = {
            'cam_device': selected_device_alias,  # Store the alias key instead of the device path
            'cam_name': cam_name,
            'save_interval': save_interval,
            'device_aliases': device_aliases
        }
        
        # Preserve data_dir if it exists in the current config
        if hasattr(self, 'config') and isinstance(self.config, dict) and 'data_dir' in self.config:
            config['data_dir'] = self.config['data_dir']
        
        # Only update resolution in config if the resolution setting is shown
        if SHOW_RESOLUTION_SETTING:
            config['resolution'] = resolution_str
        else:
            # Keep the existing resolution from the config
            config['resolution'] = self.config.get('resolution', resolution_str)
        
        # Save configuration
        st.session_state.video_capture.config = config
        st.session_state.video_capture.save_config(st.session_state.video_capture.config)
        
        # Parse resolution string for capture thread
        try:
            width, height = map(int, resolution_str.split('x'))
        except (ValueError, AttributeError, IndexError):
            # Default to 1080p if parsing fails
            width, height = 1920, 1080
        
        # Start capture thread
        st.session_state.video_capture.start_capture_thread(
            cam_device=cam_device,  # Use the actual device path for capture
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
        
        # Update the rt_capture state to match the actual capture state
        #print(f"Stopping capture: {st.session_state.rt_capture}")
        #if 'rt_capture' in st.session_state:
        #    st.session_state.rt_capture = False

        try:
            # Release PyTorch resources explicitly
            if hasattr(self, 'model') and self.model is not None:
                self.model = None
                
            # Force garbage collection
            import gc
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

def jiffyget(hour: int, minute: int, second: int, cam_name: str, direction: str = "down"):
    """Find the closest image to the given time.
    
    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second (0-59)
        cam_name: Camera name
        direction: Direction to search ("up" or "down")
        
    Returns:
        Tuple of (image_path, timestamp) or None if no image found
    """
    # Safely get session with a fallback to 'Default'
    session = st.session_state.get('session', 'Default')
    
    # Get the data directory directly from the VideoCapture config
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')
    
    # Use browsing_date from session state instead of current date
    browse_date = st.session_state.browsing_date
    
    # Call the external jiffyget function
    return external_jiffyget(hour, minute, second, cam_name, session, data_dir, browse_date, direction)


def get_timestamp_range(cam_name: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the oldest and newest timestamps available for the camera.
    
    Args:
        cam_name: Camera name
        
    Returns:
        Tuple of (oldest_timestamp, newest_timestamp) as datetime objects, or (None, None) if no images
    """
    # Safely get session with a fallback to 'Default'
    session = st.session_state.get('session', 'Default')
    
    # Get the data directory directly from the VideoCapture config
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')
    
    # Call the external get_timestamp_range function
    return external_get_timestamp_range(cam_name, session, data_dir)

def main():
    # Add these at the very beginning of main(), before any other code

    if 'slider_currently_being_dragged' not in st.session_state:
        st.session_state.slider_currently_being_dragged = False

    
    st.set_page_config(
        page_title="JiffyCam",
        page_icon="🎥",
        layout="wide"
    )
    
    # Function to format time in 12-hour format with AM/PM
    def format_time_12h(hour, minute, second):
        """Format time in 12-hour format with AM/PM indicator."""
        period = "AM" if hour < 12 else "PM"
        hour_12 = hour % 12
        if hour_12 == 0:
            hour_12 = 12
        return f"{hour_12}:{minute:02d}:{second:02d} {period}"
    
    # Function to toggle real-time capture - MOVED HERE before it's referenced in sidebar
    def toggle_rt_capture():
        """Toggle real-time capture on/off"""
        # Toggle the rt_capture state
        st.session_state.rt_capture = not st.session_state.rt_capture
        
        if st.session_state.rt_capture:     
            # When starting capture, also exit playback mode
            st.session_state.in_playback_mode = False
            # Get resolution
            resolution = st.session_state.resolution
            
            # Check if resolution is a key in RESOLUTIONS dictionary
            if resolution in RESOLUTIONS:
                width, height = RESOLUTIONS[resolution]
                resolution_str = f"{width}x{height}"
            else:
                # Assume resolution is already in the format "widthxheight"
                resolution_str = resolution
                
                # Validate format
                if not isinstance(resolution_str, str) or 'x' not in resolution_str:
                    resolution_str = "1920x1080"  # Default if invalid
            
            # Reset browsing state
            st.session_state.browsing_saved_images = False
            st.session_state.date = datetime.now().date()

            # Set the time slider to 0
            st.session_state.time_slider = 0
            
            # Start capture with the resolution string
            st.session_state.video_capture.start_capture(resolution_str)
        else:
            st.session_state.video_capture.stop_capture()
    
    # Function to synchronize time slider with current time values
    def sync_time_slider():
        """Synchronize the time slider with the current hour, minute, and second values."""
        # This function is disabled to prevent the time slider from syncing to actual time
        pass  # Do nothing to prevent syncing
    
    # Call the sync function at startup
    # sync_time_slider()  # Commented out to prevent automatic syncing to actual time
    
    # Initialize VideoCapture instance
    if 'video_capture' not in st.session_state:
        st.session_state.video_capture = VideoCapture()
    
    # Initialize in_playback_mode first, before any other code tries to use it
    if 'in_playback_mode' not in st.session_state:
        st.session_state.in_playback_mode = False
    
    # Initialize session state from config
    if 'cam_name' not in st.session_state:
        st.session_state.cam_name = st.session_state.video_capture.config.get('cam_name', 'cam0')
    if 'previous_cam_name' not in st.session_state:
        st.session_state.previous_cam_name = st.session_state.cam_name
    if 'device_aliases' not in st.session_state:
        st.session_state.device_aliases = st.session_state.video_capture.config.get('device_aliases', {'USB0': '0', 'USB1': '1', 'Default': '0'})
    
    # Always ensure data_dir is set in session state
    # This is critical as it's used in multiple places
    st.session_state.data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')
    
    # Ensure the data directory exists
    os.makedirs(st.session_state.data_dir, exist_ok=True)
    
    # If data_dir is not in the config but we're using a custom value, add it to the config
    if 'data_dir' not in st.session_state.video_capture.config and st.session_state.data_dir != 'JiffyData':
        config = st.session_state.video_capture.config.copy()
        config['data_dir'] = st.session_state.data_dir
        st.session_state.video_capture.config = config
        st.session_state.video_capture.save_config(config)
    
    # Initialize all session state variables with defaults if they don't exist
    if 'cam_device' not in st.session_state:
        # Get the cam_device from config - this might be an alias key now
        config_cam_device = st.session_state.video_capture.config.get('cam_device', '0')
        
        # Check if the cam_device is an alias key in device_aliases
        device_aliases = st.session_state.video_capture.config.get('device_aliases', {'USB0': '0', 'USB1': '1', 'Default': '0'})
        if config_cam_device in device_aliases:
            # If it's an alias key, use its value as the actual device path
            st.session_state.cam_device = device_aliases[config_cam_device]
        else:
            # If it's not an alias key, use it directly (backward compatibility)
            st.session_state.cam_device = config_cam_device
    
    # Add device_aliases to session state
    if 'device_aliases' not in st.session_state:
        st.session_state.device_aliases = st.session_state.video_capture.config.get('device_aliases', {'USB0': '0', 'USB1': '1', 'Default': '0'})
    
    # Add selected device alias to session state
    if 'selected_device_alias' not in st.session_state:
        # First check if cam_device in config is already an alias key
        config_cam_device = st.session_state.video_capture.config.get('cam_device', '0')
        if config_cam_device in st.session_state.device_aliases:
            # If it's an alias key, use it directly
            found_alias = config_cam_device
        else:
            # Otherwise, find the alias that corresponds to the current cam_device
            current_path = st.session_state.cam_device
            found_alias = None
            for alias, path in st.session_state.device_aliases.items():
                if path == current_path:
                    found_alias = alias
                    break
            
            # If no matching alias found, use the first one or 'Default'
            if not found_alias:
                if 'Default' in st.session_state.device_aliases:
                    found_alias = 'Default'
                elif st.session_state.device_aliases:
                    found_alias = list(st.session_state.device_aliases.keys())[0]
                else:
                    found_alias = 'Default'
                    st.session_state.device_aliases = {'Default': '0'}
        
        st.session_state.selected_device_alias = found_alias
    
    # Set session based on the selected device alias
    if 'session' not in st.session_state or st.session_state.session == 'Default':
        # Always set session to match the selected device alias
        # Note: 'session' is no longer saved to YAML as it's automatically derived from selected_device_alias
        st.session_state.session = st.session_state.selected_device_alias

    # Add a variable to track the previous camera name for change detection
    if 'previous_cam_name' not in st.session_state:
        st.session_state.previous_cam_name = st.session_state.cam_name

    # Store last frame to keep it visible when stopped
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
        
    # Add a flag to track when we're browsing saved images
    if 'browsing_saved_images' not in st.session_state:
        st.session_state.browsing_saved_images = False
        
    # Initialize resolution from config
    if 'resolution' not in st.session_state:
        # Ensure resolution is treated as a string before splitting
        config_resolution = st.session_state.video_capture.config.get('resolution', '1920x1080')
        if not isinstance(config_resolution, str):
            config_resolution = str(config_resolution)
        
        # Check if the resolution is already in the format "widthxheight"
        if 'x' in config_resolution:
            # Try to find a matching resolution in the RESOLUTIONS dictionary
            resolution_found = False
            for res_name, (width, height) in RESOLUTIONS.items():
                if f"{width}x{height}" == config_resolution:
                    st.session_state.resolution = res_name
                    resolution_found = True
                    break
            
            # If no matching resolution found in the dictionary, use the string directly
            if not resolution_found:
                # Store the resolution string directly
                st.session_state.resolution = config_resolution
        else:
            # Default to 1080p if no valid format
            st.session_state.resolution = "1080p (1920x1080)"
    
    # Initialize save_interval from config
    if 'save_interval' not in st.session_state:
        st.session_state.save_interval = int(st.session_state.video_capture.config.get('save_interval', 60))
    
    # Get current time for initializing time values
    current_time = datetime.now()
    
    # Initialize session state for time values if not present
    if 'hour' not in st.session_state:
        st.session_state.hour = current_time.hour
    if 'minute' not in st.session_state:
        st.session_state.minute = current_time.minute
    if 'second' not in st.session_state:
        st.session_state.second = current_time.second
    # Add back the date initialization
    if 'date' not in st.session_state:
        st.session_state.date = current_time.date()
    
    # Initialize additional session state variables for browsing
    if 'browsing_hour' not in st.session_state:
        st.session_state.browsing_hour = st.session_state.hour
    if 'browsing_minute' not in st.session_state:
        st.session_state.browsing_minute = st.session_state.minute
    if 'browsing_second' not in st.session_state:
        st.session_state.browsing_second = st.session_state.second
    if 'browsing_date' not in st.session_state:
        st.session_state.browsing_date = st.session_state.date
    if 'time_display_html' not in st.session_state:
        st.session_state.time_display_html = None
    if 'actual_timestamp' not in st.session_state:
        st.session_state.actual_timestamp = None
    if 'oldest_timestamp' not in st.session_state:
        st.session_state.oldest_timestamp = None
    if 'newest_timestamp' not in st.session_state:
        st.session_state.newest_timestamp = None
    
    # Add a toggle state for the RT button
    if 'rt_active' not in st.session_state:
        st.session_state.rt_active = False
    
    # Add rt_capture state variable
    if 'rt_capture' not in st.session_state:
        st.session_state.rt_capture = False
    #print(f"add to session state rt_capture: {st.session_state.rt_capture}")

    # Add a flag to indicate that we need to display the most recent image on startup
    if 'need_to_display_recent' not in st.session_state:
        st.session_state.need_to_display_recent = True
    
    
    # Create sidebar for settings
    with st.sidebar:
        # Add the title at the top of the sidebar
        st.title("JiffyCam")
        
        st.header("Settings")
        
        # Function to get available camera names from JiffyData
        def get_camera_names():
            """Get list of available camera names from JiffyData directory."""
            camera_names = set()
            session = st.session_state.get('session', 'Default')  # Changed from camera_group
            
            # Default camera name if none found
            default_names = ['cam0']
            
            # Get the data directory directly from the VideoCapture config
            data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')
            
            # Check if data directory and the selected session exist
            group_path = os.path.join(data_dir, session)  # Changed from camera_group
            if not os.path.exists(group_path):
                return default_names + ['Add New Camera']
            
            # Walk through the directory structure to find all jpg files
            for root, dirs, files in os.walk(group_path):
                for file in files:
                    if file.endswith('.jpg'):
                        # Get the relative path from the group directory
                        rel_path = os.path.relpath(root, group_path)
                        if rel_path == '.':  # Files directly in the group directory
                            camera_name = os.path.splitext(file)[0]
                        else:
                            # Extract the parent directory (timestamp directories are the leaf nodes)
                            parent_dir = os.path.dirname(rel_path)
                            if parent_dir == '':  # Direct child of group directory
                                camera_name = os.path.splitext(file)[0]
                            else:
                                # Combine parent directory with filename without extension
                                camera_name = os.path.join(parent_dir, os.path.splitext(file)[0])
                        
                        # Add to our set of camera names
                        camera_names.add(camera_name)
            
            # Convert set to list and sort
            result = sorted(list(camera_names))
            
            # If no camera names found, use default
            if not result:
                result = default_names
                
            # Add default camera name if not in the list
            if 'cam0' not in result:
                result.insert(0, 'cam0')
            else:
                # Move cam0 to the beginning
                result.remove('cam0')
                result.insert(0, 'cam0')
            
            # Add option to create new camera
            result.append('Add New Camera')
            
            return result
        
        # Function to handle device alias selection
        def on_device_alias_change():
            """Update cam_device when device alias is changed"""
            selected_alias = st.session_state.selected_device_alias
            if selected_alias in st.session_state.device_aliases:
                st.session_state.cam_device = st.session_state.device_aliases[selected_alias]
                # Automatically set the session to match the selected device alias
                st.session_state.session = selected_alias
                
                # Reset timestamp range to force recalculation for the new device/session
                st.session_state.oldest_timestamp = None
                st.session_state.newest_timestamp = None
                
                # Set flag to display most recent image
                st.session_state.need_to_display_recent = True
        
        # Replace text input with selectbox for Device
        device_aliases = list(st.session_state.device_aliases.keys())
        
        selected_alias = st.selectbox(
            "Device",
            options=device_aliases,
            key='selected_device_alias',
            on_change=on_device_alias_change,
            help="Select camera device"
        )
        
        # Session is now automatically set based on the selected device alias
        # No need for a separate Session selectbox
        
        # Create a list of options that includes both RESOLUTIONS keys and any custom resolution
        resolution_options = list(RESOLUTIONS.keys())
        
        # If the current resolution is not in RESOLUTIONS, add it to the options
        if st.session_state.resolution not in RESOLUTIONS:
            if st.session_state.resolution not in resolution_options:
                resolution_options.append(st.session_state.resolution)
        
        # Only show the Resolution selectbox if the flag is True
        if SHOW_RESOLUTION_SETTING:
            st.selectbox(
                "Resolution",
                resolution_options,
                key="resolution",
                help="Select camera resolution"
            )
        else:
            # Instead of showing a hidden selectbox, we'll completely remove it
            # The session state for resolution will still be maintained
            # We don't need any placeholder here
            pass

        st.number_input(
            "Save Interval (seconds)",
            key='save_interval',
            min_value=0,
            help="Interval between saved frames (0 to disable periodic saves)"
        )
        
        # Add the toggle capture button to the sidebar
        # Check if capture is currently running to set the initial button state
        is_capturing = st.session_state.video_capture.capture_thread and st.session_state.video_capture.capture_thread.is_alive()
        st.session_state.rt_active = is_capturing
        
        # Create a toggle button that changes text based on state
        button_text = "Stop Capture" if is_capturing else "Start Capture"
        button_color = "secondary" if is_capturing else "primary"
        
        # Add a separator before the capture button
        st.markdown("---")
        
        # Create the button with the appropriate text and styling
        st.button(button_text, key="rt_toggle", on_click=toggle_rt_capture, 
                  help="Toggle real-time capture", type=button_color)
        
        if is_capturing:
            st.success("Capture is active and running")
        
        # Move status and error placeholders to sidebar
        st.header("Status")
        status_placeholder = st.empty()
        error_placeholder = st.empty()

    # CSS for better styling
    st.markdown("""
    <style>
    /* Reduce spacing in Streamlit containers but keep enough to prevent clipping */
    .block-container {
        padding-top: 0.7rem !important;
        padding-bottom: 0.7rem !important;
    }
    
    /* Remove extra padding from main elements but keep minimal spacing */
    div[data-testid="stVerticalBlock"] > div {
        padding-top: 1px !important;
        padding-bottom: 1px !important;
    }
    
    /* Reduce spacing between elements but maintain enough to prevent clipping */
    div[data-testid="element-container"] {
        margin-top: 1px !important;
        margin-bottom: 1px !important;
    }
    
    /* Compact time display */
    .time-display {
        font-size: 1.6rem;
        font-weight: 400;
        text-align: center;
        font-family: "Source Sans Pro", sans-serif;
        background-color: transparent;
        color: #333333;
        border-radius: 5px;
        padding: 2px;
        margin: 2px 0;
    }
    
    /* Make time display lighter in dark mode */
    @media (prefers-color-scheme: dark) {
        .time-display {
            color: #e0e0e0 !important;
        }
    }
    
    /* Streamlit-specific dark mode detection */
    [data-testid="stAppViewContainer"][data-theme="dark"] .time-display {
        color: #e0e0e0 !important;
    }
    
    .time-label {
        text-align: center;
        font-weight: normal;
        margin-bottom: 2px;
        margin-top: 0;
        font-size: 0.8rem;
        line-height: 1;
    }
    
    .time-button {
        text-align: center;
        font-size: 1.2rem;
    }
    
    .time-separator {
        font-size: 1.6rem;
        font-weight: bold;
        text-align: center;
        margin-top: 12px;
    }
    
    /* Make video container more compact but prevent clipping */
    div[data-testid="stImage"] {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    
    /* Reduce spacing in sidebar */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }
    
    /* Reduce spacing for headers */
    h1, h2, h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create time display and progress containers - MOVED ABOVE video placeholder
    time_container = st.container()
    with time_container:   
        
        def update_image_display(direction=None):
            """Update the image display based on the current date and time."""
            # Ensure necessary session state variables exist         
            if 'slider_currently_being_dragged' not in st.session_state:
                st.session_state.slider_currently_being_dragged = False
            
            # Always set playback mode to True when browsing images
            st.session_state.in_playback_mode = True
            
            # Find the closest image to the requested time using the wrapper function
            closest_image = jiffyget(
                st.session_state.hour,
                st.session_state.minute,
                st.session_state.second,
                st.session_state.cam_name,
                direction
            )
            
            if closest_image:
                frame, timestamp = closest_image
                try:
                    #frame = cv2.imread(image_path)
                    if frame is not None:
                        # Store the frame in session state and timestamp
                        st.session_state.last_frame = frame.copy()
                        st.session_state.actual_timestamp = timestamp
                        
                        # Update the display
                        video_placeholder.image(frame, channels="BGR", use_container_width=True)
                        
                        # Update time values with the timestamp
                        st.session_state.hour = timestamp.hour
                        st.session_state.minute = timestamp.minute
                        st.session_state.second = timestamp.second
                        
                        # Update the time display
                        time_display.markdown(
                            f'<div class="time-display">{format_time_12h(timestamp.hour, timestamp.minute, timestamp.second)}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Only update the slider position programmatically if this wasn't triggered by a manual slider adjustment
                        if not st.session_state.slider_currently_being_dragged:
                            st.session_state.time_slider = datetime_time(
                                timestamp.hour,
                                timestamp.minute,
                                timestamp.second
                            )
                        
                        # Update status with timestamp info
                        status_placeholder.text(f"Displaying image from: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        return True
                        
                except Exception as e:
                    status_placeholder.text(f"Error loading image: {str(e)}")
                    # Keep showing the last frame if there's an error
                    if st.session_state.last_frame is not None:
                        video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
            else:
                # No image found for the selected time
                status_placeholder.text(f"No images found for the selected time")
                # Keep showing the last frame
                if st.session_state.last_frame is not None:
                    video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
            
            # Stay in playback mode regardless of result
            st.session_state.in_playback_mode = True
            return False

        # Create placeholder for video - MOVED INSIDE time_container
        video_placeholder = st.empty()

        # Create a more compact container for time controls
        st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
        
        # Create a container with fixed width for time controls to ensure proper centering
        st.markdown("<div class='time-controls-container' style='max-width: 600px; margin: 0 auto;'>", unsafe_allow_html=True)
        
        # Function to handle date change
        def on_date_change():
            # Update browsing date
            st.session_state.browsing_date = st.session_state.date
            
            # Set flag to indicate we're in playback mode instead of stopping capture
            st.session_state.in_playback_mode = True
                        
            # Update the time slider to match the current time values
            if 'time_slider' in st.session_state:
                st.session_state.time_slider = datetime_time(
                    st.session_state.hour,
                    st.session_state.minute,
                    st.session_state.second
                )
            
            # Find and display closest image with current time and new date
            update_image_display(direction="down")

        # Function to toggle real-time capture
        def toggle_rt_capture():
            """Toggle real-time capture on/off"""
            # Toggle the rt_capture state
            #print(f"Toggling rt_capture: {st.session_state.rt_capture}")
            st.session_state.rt_capture = not st.session_state.rt_capture
            
            if st.session_state.rt_capture:     
                # When starting capture, also exit playback mode
                st.session_state.in_playback_mode = False
                # Get resolution
                resolution = st.session_state.resolution
                
                # Check if resolution is a key in RESOLUTIONS dictionary
                if resolution in RESOLUTIONS:
                    width, height = RESOLUTIONS[resolution]
                    resolution_str = f"{width}x{height}"
                else:
                    # Assume resolution is already in the format "widthxheight"
                    resolution_str = resolution
                    
                    # Validate format
                    if not isinstance(resolution_str, str) or 'x' not in resolution_str:
                        resolution_str = "1920x1080"  # Default if invalid
                
                # Reset browsing state
                st.session_state.browsing_saved_images = False
                st.session_state.date = datetime.now().date()
                #print(f"Setting time_slider to 0: {st.session_state.time_slider}")
                #traceback.print_stack()

                # Set the time slider to 0
                st.session_state.time_slider = 0
                
                # Start capture with the resolution string
                st.session_state.video_capture.start_capture(resolution_str)
            else:
                st.session_state.video_capture.stop_capture()

        # Date picker in its own row at the top
        st.markdown('<div class="date-picker-container">', unsafe_allow_html=True)

        # Get min and max dates from timestamp range if available
        min_date = None
        max_date = None
        
        # Update timestamp range if not already set
        if st.session_state.oldest_timestamp is None or st.session_state.newest_timestamp is None:
            oldest, newest = get_timestamp_range(st.session_state.cam_name)
            if oldest and newest:
                st.session_state.oldest_timestamp = oldest
                st.session_state.newest_timestamp = newest
        
        # Set min and max dates from session state
        if st.session_state.oldest_timestamp:
            min_date = st.session_state.oldest_timestamp.date()
        if st.session_state.newest_timestamp:
            max_date = st.session_state.newest_timestamp.date()
            
        # Ensure max_date includes the current date
        today = datetime.now().date()
        if max_date is None or max_date < today:
            max_date = today

        # Add the date picker with min and max dates
        st.date_input(
            "Date",  # Provide a proper label
            key="date",
            on_change=on_date_change,
            help="Select date to view images from",
            label_visibility="collapsed",  # Collapse the label to add it manually below
            min_value=min_date,
            max_value=max_date
        )

        # Close the container
        st.markdown('</div>', unsafe_allow_html=True)

        # Add a separator line after the date picker
       #st.markdown("<hr style='margin-top: 2px; margin-bottom: 0px; padding: 0;'>", unsafe_allow_html=True)

        # Create a more compact layout for time controls with equal column widths
        time_cols = st.columns([3, 1, 1, 0.2, 1])
        
        # Function to handle time changes
        def on_time_change():
            """Handle time changes and update the image display."""
            # Stop capture if running
            if st.session_state.video_capture.capture_thread and st.session_state.video_capture.capture_thread.is_alive():
                st.session_state.video_capture.stop_capture()
            
            # Update the image display with the new time
            update_image_display(direction="down")
        
        # Function to handle previous image button
        def on_prev_button():
            """Handle previous image button click."""
            
            # Rest of the function remains the same
            st.session_state.in_playback_mode = True
            st.session_state.slider_currently_being_dragged = False
            
            # Decrement time logic...
            current_second = st.session_state.second
            if current_second == 0:
                # Handle rollover
                current_minute = st.session_state.minute
                if current_minute == 0:
                    st.session_state.hour = (st.session_state.hour - 1) % 24
                    st.session_state.minute = 59
                else:
                    st.session_state.minute = current_minute - 1
                st.session_state.second = 59
            else:
                st.session_state.second = current_second - 1
            
            # Update the time slider and display
            if 'time_slider' in st.session_state:
                st.session_state.time_slider = datetime_time(
                    st.session_state.hour,
                    st.session_state.minute,
                    st.session_state.second
                )
            
            # Find and display the previous image
            update_image_display(direction="down")
            st.session_state.slider_currently_being_dragged = True   # mjm

        # Function to handle next image button
        def on_next_button():
            """Handle next image button click."""
            
            # Rest of the function remains the same
            st.session_state.in_playback_mode = True
            st.session_state.slider_currently_being_dragged = False
            
            # Increment time logic
            current_second = st.session_state.second
            if current_second == 59:
                # Handle rollover
                current_minute = st.session_state.minute
                if current_minute == 59:
                    st.session_state.hour = (st.session_state.hour + 1) % 24
                    st.session_state.minute = 0
                else:
                    st.session_state.minute = current_minute + 1
                st.session_state.second = 0
            else:
                st.session_state.second = current_second + 1
            
            # Update the time slider and display
            if 'time_slider' in st.session_state:
                st.session_state.time_slider = datetime_time(
                    st.session_state.hour,
                    st.session_state.minute,
                    st.session_state.second
                )
            
            # Find and display the next image
            update_image_display(direction="up")
            st.session_state.slider_currently_being_dragged = True   # mjm

        # Time display (combined H:M:S format)
        with time_cols[0]:
            time_display = st.empty()
            time_display.markdown(f'<div class="time-display">{format_time_12h(st.session_state.hour, st.session_state.minute, st.session_state.second)}</div>', unsafe_allow_html=True)
            #st.markdown('<div class="time-label">Time (H:M:S)</div>', unsafe_allow_html=True)
            
        # Prev button
        with time_cols[1]:
            st.button(" ◀ ", key="prev_button", on_click=on_prev_button, help="Previous image", use_container_width=True)
            
        # Next button
        with time_cols[2]:
            st.button(" ▶ ", key="next_button", on_click=on_next_button, help="Next image", use_container_width=True)
            
        # Vertical separator
        with time_cols[3]:
            st.markdown('<div style="width:1px; background-color:#555555; height:32px; margin:0 auto;"></div>', unsafe_allow_html=True)
            
        # Record button - in the last column - always shows Live Display button
        with time_cols[4]:
            # Check if capture is currently running
            is_capturing = st.session_state.video_capture.capture_thread and st.session_state.video_capture.capture_thread.is_alive()
            
            # Initialize the button state variables if they don't exist
            if 'live_button_clicked' not in st.session_state:
                st.session_state.live_button_clicked = False
            
            # Determine if we're in playback mode
            in_playback = st.session_state.in_playback_mode
            
            # Set button text based on current state - use pause icon instead of text
            button_text = "Live" if in_playback else "⏸"
            button_help = "Return to live view" if in_playback else "Pause on current frame"
            
            # Define helper function for button action
            def toggle_live_pause():
                # If in playback mode, go to live mode
                if st.session_state.in_playback_mode:
                    st.session_state.in_playback_mode = False
                    st.session_state.live_button_clicked = True  # Track that button was clicked
                    
                    # Update browsing date and time to current
                    current_time = datetime.now()
                    st.session_state.browsing_date = current_time.date()
                    st.session_state.hour = current_time.hour
                    st.session_state.minute = current_time.minute
                    st.session_state.second = current_time.second
                else:
                    # If in live mode, pause by entering playback mode
                    # The current frame is already stored in st.session_state.last_frame
                    st.session_state.in_playback_mode = True
                    # Store the current time
                    current_time = datetime.now()
                    st.session_state.actual_timestamp = current_time
            
            if not is_capturing:
                # Capture not running - show disabled button
                st.button("Live", key="live_btn", use_container_width=True, 
                          help="Start capture to enable live display", disabled=True)
            else:
                # Display the button with text based on current state
                st.button(button_text, key="live_btn", use_container_width=True,
                          help=button_help, on_click=toggle_live_pause)

        
        # Close the time controls container
        st.markdown("</div>", unsafe_allow_html=True)

   

    # Add functions to convert time to seconds and vice versa
    def time_to_seconds(hours, minutes, seconds):
        """Convert hours, minutes, seconds to total seconds."""
        return hours * 3600 + minutes * 60 + seconds
        
    def seconds_to_time(total_seconds):
        """Convert total seconds to hours, minutes, seconds."""
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return hours, minutes, seconds
        
    # Function to handle time slider change
    def on_time_slider_change():
        """Handle time slider change."""
        # Ensure necessary session state variables exist
        if 'slider_currently_being_dragged' not in st.session_state:
            st.session_state.slider_currently_being_dragged = False
        
        # Get the time object from the slider
        time_obj = st.session_state.time_slider
        
        # Extract hours, minutes, seconds
        hours = time_obj.hour
        minutes = time_obj.minute
        seconds = time_obj.second
        
        # Only update if the values have actually changed
        if (hours != st.session_state.hour or 
            minutes != st.session_state.minute or 
            seconds != st.session_state.second):
            
            # Update the time state variables
            st.session_state.hour = hours
            st.session_state.minute = minutes
            st.session_state.second = seconds
            
            # Mark that a manual adjustment is currently active (prevents immediate jumps)
            st.session_state.slider_currently_being_dragged = True
            
            # Set flag to indicate we're in playback mode instead of stopping capture
            st.session_state.in_playback_mode = True
            
            # Update the image display with the new time
            update_image_display(direction="down")

    # Add CSS for the time slider
    st.markdown("""
    <style>
    /* Style for time slider */
    div[data-testid="stSlider"] {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        margin-top: -10px !important;
        margin-bottom: 0px !important;
    }
    
    /* Center the slider label */
    div[data-testid="stSlider"] > label {
        text-align: center !important;
        width: 100% !important;
        display: block !important;
        font-weight: bold !important;
        margin-bottom: 0px !important;
        height: 0px !important;  /* Reduce height to minimum */
        overflow: hidden !important;  /* Hide any overflow */
    }
    
    /* Style for slider track */
    div[data-testid="stSlider"] > div > div > div {
        background-color: #555555 !important;
    }
    
    /* Style for slider thumb */
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: #0f0f0f !important;
        border-color: #ffffff !important;
    }
    
    /* Hide the min/max labels on the slider */
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] {
        display: none !important;
    }
    
    /* Reduce spacing for markdown elements */
    div[data-testid="stMarkdown"] {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    
    /* Tighten spacing for horizontal rules */
    hr {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        padding: 0 !important;
    }
    
    /* Remove space above slider container */
    div[data-testid="stSlider"] > div {
        padding-top: 0px !important;
        margin-top: -5px !important;
    }
    
    /* Target the element container for the slider */
    div[data-testid="element-container"]:has(div[data-testid="stSlider"]) {
        margin-top: -5px !important;
        padding-top: 0px !important;
    }

    /* Style the centered date picker */
    div[data-testid="column"]:nth-child(2):has(div[data-testid="stDateInput"]) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-bottom: -10px !important; /* Reduce space between date picker and slider */
    }

    div[data-testid="stDateInput"] {
        margin-bottom: 0px !important;
        padding-bottom: 1px !important;
        padding-top: 1px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        margin-top: -5px !important; /* Move the date picker up to align with time buttons */
    }

    div[data-testid="stDateInput"] > div {
        margin-bottom: 0px !important;
        display: flex !important;
        align-items: center !important;
        height: 32px !important; /* Match the height of the time buttons */
    }

    div[data-testid="stDateInput"] input {
        padding: 2px 8px !important;
        height: 32px !important;
        font-size: 14px !important;
        background-color: #2e2e2e !important; /* Dark background for dark mode */
        color: #ffffff !important; /* White text for contrast */
        border: 1px solid #555555 !important; /* Darker border */
        border-radius: 5px !important;
        text-align: center !important;
        font-weight: bold !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }

    /* Style the date picker icon for dark mode */
    div[data-testid="stDateInput"] svg {
        fill: #ffffff !important;
    }

    /* Center the date picker container */
    div.row-widget.stHorizontal:has(div[data-testid="stDateInput"]) {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }

    /* Ensure the middle column with date picker is properly centered */
    div.row-widget.stHorizontal:has(div[data-testid="stDateInput"]) > div:nth-child(2) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize the time_slider with a datetime.time object if not already set
    if 'time_slider' not in st.session_state or not st.session_state.slider_currently_being_dragged:
        st.session_state.time_slider = datetime_time(
            st.session_state.hour,
            st.session_state.minute,
            st.session_state.second
        )
    
    # Format the date for display in the slider label
    # Ensure date is a datetime.date object before calling strftime
    if isinstance(st.session_state.date, str):
        # If it's a string, try to parse it to a date object
        try:
            date_obj = datetime.strptime(st.session_state.date, "%Y-%m-%d").date()
            date_str = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            # If parsing fails, use today's date
            date_str = datetime.now().date().strftime("%Y-%m-%d")
    else:
        # It's already a date object
        date_str = st.session_state.date.strftime("%Y-%m-%d")
    
    # Function to generate timeline bar with marks for available captures
    def generate_timeline_bar():
        """Generate a timeline bar with marks for available capture times."""
        # Get all available timestamps for the current camera and date
        # Safely get session with a fallback to 'Default'
        session = st.session_state.get('session', 'Default')  # Changed from camera_group
        
        # Get the data directory directly from the VideoCapture config
        data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')
        
        # Construct the base directory path
        base_dir = os.path.join(data_dir, session, os.path.dirname(st.session_state.cam_name))  # Changed from camera_group
        
        if not os.path.exists(base_dir):
            return None  # No data available
            
        # Get all timestamp directories
        timestamp_dirs = glob.glob(os.path.join(base_dir, "*"))
        
        if not timestamp_dirs:
            return None  # No data available
            
        # Convert directory names to timestamps and filter by current date
        timestamps = []
        current_date = st.session_state.date
        
        for dir_path in timestamp_dirs:
            try:
                dir_name = os.path.basename(dir_path)
                timestamp_ms = int(dir_name)
                dt = datetime.fromtimestamp(timestamp_ms / 1000)
                
                # Only include timestamps from the current date
                if dt.date() == current_date:
                    # Calculate position as percentage of day (0-100%)
                    seconds_in_day = 24 * 60 * 60
                    seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
                    position = (seconds_since_midnight / seconds_in_day) * 100
                    timestamps.append((dt, position))
            except ValueError:
                continue
                
        if not timestamps:
            return None  # No data available for current date
            
        # Build the complete HTML for the timeline bar in one string
        html = """
        <style>
        .timeline-container {
            width: 100%;
            padding: 0;
            margin: 0;
            margin-top: -5px;
            margin-bottom: 2px;
        }
        .timeline-bar {
            position: relative;
            width: 100%;
            height: 8px;
            background-color: #333333;
            border-radius: 2px;
        }
        .timeline-mark {
            position: absolute;
            width: 2px;
            height: 8px;
            background-color: #ffffff;
            opacity: 0.7;
        }
        </style>
        <div class="timeline-container">
            <div class="timeline-bar">
        """
        
        # Add marks for each timestamp
        for dt, position in timestamps:
            html += f'<div class="timeline-mark" style="left: {position}%;"></div>'
        
        # Close the HTML
        html += """
            </div>
        </div>
        """
        
        return html
    
    # Display the timeline bar above the time slider
    timeline_html = generate_timeline_bar()
    if timeline_html:
        st.markdown(timeline_html, unsafe_allow_html=True)
    else:
        # Display an empty timeline bar as a placeholder
        empty_timeline_html = """
        <style>
        .timeline-container {
            width: 100%;
            padding: 0;
            margin: 0;
            margin-top: -5px;
            margin-bottom: 2px;
        }
        .timeline-bar {
            position: relative;
            width: 100%;
            height: 8px;
            background-color: #333333;
            border-radius: 2px;
        }
        </style>
        <div class="timeline-container">
            <div class="timeline-bar"></div>
        </div>
        """
        st.markdown(empty_timeline_html, unsafe_allow_html=True)
    
    # Create the time slider with time objects
    st.slider(
        date_str,  # Use the date as the label
        min_value=datetime_time(0, 0, 0),
        max_value=datetime_time(23, 59, 59),
        key="time_slider",
        on_change=on_time_slider_change,
        step=timedelta(seconds=1),
        label_visibility="hidden"  # Make the label visible
    )

    # Check if capture is currently running to set the initial button state
    is_capturing = st.session_state.video_capture.capture_thread and st.session_state.video_capture.capture_thread.is_alive()
    st.session_state.rt_active = is_capturing

    # Create placeholder for video - MOVED BELOW time controls
    # Add CSS to make the video container more compact but prevent clipping
    st.markdown("""
    <style>
    /* Make video container more compact but prevent clipping */
    div[data-testid="stImage"] > img {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    
    /* Remove extra padding from the main container but keep minimal spacing */
    .main .block-container {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    
    /* Make all elements in the main area more compact but prevent clipping */
    .main [data-testid="stVerticalBlock"] {
        gap: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    video_placeholder = st.empty()

   
    # Function to display the most recent image for the current camera
    def display_most_recent_image():
        """Display the most recent image for the current camera."""
        timestamp = datetime.now()
        st.session_state.hour = timestamp.hour
        st.session_state.minute = timestamp.minute
        st.session_state.second = timestamp.second
        update_image_display(direction="up")
        return


    # Display the most recent image on startup or when camera name changes
    if st.session_state.need_to_display_recent and not is_capturing:
        display_most_recent_image()
        st.session_state.need_to_display_recent = False

    # First, add this after initializing session state variables (around line 325-350)
    # Add a session state variable for status message
    if 'status_message' not in st.session_state:
        st.session_state.status_message = ""

    # Update the UI
    while True:
        # Check for errors and thread state
        if st.session_state.video_capture.error_event.is_set():
            # Force cleanup on error
            st.session_state.video_capture.stop_capture()
            error_placeholder.error(st.session_state.video_capture.last_error)
            status_placeholder.text("Status: Stopped")
            st.session_state.status_message = "Status: Stopped"
            if st.session_state.last_frame is not None:
                video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
            break

        # Update status and video frame
        is_capturing = st.session_state.video_capture.capture_thread and st.session_state.video_capture.capture_thread.is_alive()
        
        if is_capturing:
            # When capture is running
            if st.session_state.in_playback_mode:
                # In playback mode with active capture:
                # 1. Empty the queue without updating display
                while not st.session_state.video_capture.frame_queue.empty():
                    try:
                        _, fps, width, height = st.session_state.video_capture.frame_queue.get_nowait()
                        # Only update stats, don't touch the display
                        st.session_state.video_capture.current_fps = fps
                        st.session_state.video_capture.current_width = width
                        st.session_state.video_capture.current_height = height
                    except:
                        pass
                # 2. Make sure the playback frame stays visible
                if st.session_state.last_frame is not None:
                    video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    
                    # Update status message for playback mode
                    if hasattr(st.session_state, 'actual_timestamp') and st.session_state.actual_timestamp:
                        new_status = f"Viewing saved image from: {st.session_state.actual_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                        if st.session_state.status_message != new_status:
                            status_placeholder.text(new_status)
                            st.session_state.status_message = new_status
            else:
                # Normal real-time display mode
                if not st.session_state.video_capture.frame_queue.empty():
                    try:
                        frame, fps, width, height = st.session_state.video_capture.frame_queue.get_nowait()
                        if frame is not None:
                            # Update the current time in real-time mode
                            current_time = datetime.now()
                            st.session_state.hour = current_time.hour
                            st.session_state.minute = current_time.minute
                            st.session_state.second = current_time.second
                            
                            # Update the time display
                            time_display.markdown(
                                f'<div class="time-display">{format_time_12h(current_time.hour, current_time.minute, current_time.second)}</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Store and display the frame
                            st.session_state.last_frame = frame.copy()
                            video_placeholder.image(frame, channels="BGR", use_container_width=True)
                            st.session_state.video_capture.current_fps = fps
                            st.session_state.video_capture.current_width = width
                            st.session_state.video_capture.current_height = height
                            
                            # Check if we have a save status to show
                            if st.session_state.video_capture.image_just_saved:
                                new_status = st.session_state.video_capture.save_status
                                # Only update if the status has changed
                                if st.session_state.status_message != new_status:
                                    status_placeholder.text(new_status)
                                    st.session_state.status_message = new_status
                                
                                # Clear the image_just_saved flag after 3 seconds
                                if time.time() - st.session_state.video_capture.image_saved_time > 3:
                                    st.session_state.video_capture.image_just_saved = False
                            else:
                                # Update status with FPS
                                new_status = f"Live view - FPS: {fps}"
                                if st.session_state.status_message != new_status:
                                    status_placeholder.text(new_status)
                                    st.session_state.status_message = new_status
                    except:
                        pass
        else:
            # When capture is not running - crucial for playback with paused capture
            if st.session_state.in_playback_mode and st.session_state.last_frame is not None:
                # Keep showing the playback frame continuously
                video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                
                # Update status message for playback mode
                if hasattr(st.session_state, 'actual_timestamp') and st.session_state.actual_timestamp:
                    new_status = f"Viewing saved image from: {st.session_state.actual_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    if st.session_state.status_message != new_status:
                        status_placeholder.text(new_status)
                        st.session_state.status_message = new_status
                else:
                    new_status = "Viewing saved image"
                    if st.session_state.status_message != new_status:
                        status_placeholder.text(new_status)
                        st.session_state.status_message = new_status

        # Add a small delay to prevent overwhelming the CPU
        time.sleep(0.1)

if __name__ == "__main__":
    main() 