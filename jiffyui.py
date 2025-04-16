"""
jiffyui.py: UI components and logic for JiffyCam

This module contains functions for building and updating the Streamlit UI.
It uses HTTP client to connect to standalone JiffyCam server.
"""

from datetime import datetime, timedelta, time as datetime_time
import time
import os
import requests
import json
from io import BytesIO
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
#from streamlit_server_state import server_state, server_state_lock, no_rerun
import numpy as np  # Add this import for image creation
import cv2  # Add this import for image processing

from jiffyconfig import RESOLUTIONS   # Import necessary components from other modules
from jiffyget import jiffyget, get_locations, get_timestamp_range

# globals within jiffyui.py
#import jiffyglobals

# following just to avoid error when importing YOLO
import torch
torch.classes.__path__ = [] # add this line to manually set it to empty. 

# --- HTTP Client for JiffyCam Server ---

class JiffyCamClient:
    """Client for connecting to the standalone jiffycapture.py HTTP server."""
    
    def __init__(self, server_url="http://localhost:8080"):
        """Initialize the client with the server URL."""
        # Ensure server_url has protocol and doesn't end with slash
        if not server_url.startswith("http://") and not server_url.startswith("https://"):
            server_url = "http://" + server_url
        if server_url.endswith("/"):
            server_url = server_url[:-1]
            
        self.server_url = server_url
        self.connected = False
        self.last_error = None
        self.last_status = None
        self.last_frame = None
        self.last_frame_time = 0
        self.connection_check_time = 0
        self.status_check_time = 0
        self.error_event = threading.Event()  # Event for error handling
        self.is_capturing_flag = False
        
        print(f"Initialized HTTP client with server URL: {self.server_url}")
    
    def check_connection(self, force=False):
        """Check if the server is available."""
        # Only check every 5 seconds unless forced
        current_time = time.time()
        if not force and current_time - self.connection_check_time < 5:
            return self.connected
            
        self.connection_check_time = current_time
        
        try:
            response = requests.get(f"{self.server_url}/status", timeout=2)
            if response.status_code == 200:
                self.connected = True
                self.last_status = response.json()
                self.last_error = None
                self.status_check_time = current_time  # Update status check time
                self.error_event.clear()  # Clear any previous error
                return True
            else:
                self.connected = False
                self.last_error = f"Server responded with status code: {response.status_code}"
                self.error_event.set()  # Set error event
                return False
        except requests.RequestException as e:
            self.connected = False
            self.last_error = f"Connection error: {str(e)}"
            self.error_event.set()  # Set error event
            return False
    
    def is_connected(self):
        """Check if we're currently connected to the server."""
        if time.time() - self.connection_check_time > 10:  # Force check if it's been more than 10 seconds
            return self.check_connection(force=True)
        return self.connected
    
    def update_status_if_needed(self):
        """Update status information periodically without fetching a new frame."""
        current_time = time.time()
        # Only update status every 2 seconds to reduce server load
        if current_time - self.status_check_time > 2:
            self.get_status()
            return True
        return False
        
    def get_status(self):
        """Get the server status."""
        current_time = time.time()
        self.status_check_time = current_time
        
        try:
            response = requests.get(f"{self.server_url}/status", timeout=2)
            if response.status_code == 200:
                self.last_status = response.json()
                self.connected = True  # Also update connection status
                self.connection_check_time = current_time  # Update connection time too
                self.last_error = None
                self.error_event.clear()  # Clear any previous error
                return self.last_status
            else:
                self.last_error = f"Server responded with status code: {response.status_code}"
                self.error_event.set()  # Set error event
                return None
        except requests.RequestException as e:
            self.last_error = f"Connection error: {str(e)}"
            self.connected = False
            self.error_event.set()  # Set error event
            return None
    
    def get_frame(self):
        """Get the latest frame from the server."""
        # Only fetch a new frame every 1/30 second (30 FPS max)
        current_time = time.time()
        if current_time - self.last_frame_time < 0.033:
            return self.last_frame, self.last_status.get('fps', 0) if self.last_status else 0, 0, 0
            
        if not self.connected and not self.check_connection(force=False):
            return None, 0, 0, 0
            
        try:
            # Add timestamp to URL to prevent caching
            response = requests.get(f"{self.server_url}/image?t={int(current_time*1000)}", timeout=2)
            if response.status_code == 200:
                # Convert the response content to an OpenCV image
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Update last frame and time
                self.last_frame = frame
                self.last_frame_time = current_time
                
                # Use last known status without making another request
                if self.last_status:
                    try:
                        width, height = map(int, self.last_status.get('resolution', '0x0').split('x'))
                        fps = self.last_status.get('fps', 0)
                        return frame, fps, width, height
                    except (ValueError, AttributeError):
                        return frame, 0, 0, 0
                return frame, 0, 0, 0
            else:
                self.last_error = f"Server responded with status code: {response.status_code}"
                self.error_event.set()  # Set error event
                return None, 0, 0, 0
        except requests.RequestException as e:
            self.last_error = f"Connection error: {str(e)}"
            self.connected = False
            self.error_event.set()  # Set error event
            return None, 0, 0, 0
    
    def get_last_frame(self):
        """Get the cached last frame."""
        return self.last_frame
    
    # --- VideoCapture compatibility methods ---
    
    def is_capturing(self):
        """Check if capturing is currently active."""
        return self.is_capturing_flag and self.connected
    
    def start_capture(self, **kwargs):
        """Start capture - mark as capturing and connect to server."""
        if self.check_connection(force=True):
            self.is_capturing_flag = True
            return True
        return False
        
    def stop_capture(self):
        """Stop capture - mark as not capturing."""
        self.is_capturing_flag = False
        
    def error_event_is_set(self):
        """Check if there's an error."""
        return self.error_event.is_set()

# Add threading module for compatibility
import threading

# --- UI Helper Functions ---

def heartbeat():
    #print(f"heartbeat: {server_state.last_frame is not None}")

    """Heartbeat to check if the UI is still running.""" 
    if(st.session_state.autoplay_direction == "forward"):
        on_next_button(False)
    elif(st.session_state.autoplay_direction == "reverse"):
        on_prev_button(False)   

def set_autoplay(direction):
    """Set autoplay direction."""
    #global autoplay_direction
    st.session_state.autoplay_direction = direction

def format_time_12h(hour, minute, second):
    """Format time in 12-hour format with AM/PM indicator."""
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12}:{minute:02d}:{second:02d} {period}"

# --- Helper Functions ---

def get_available_recordings(data_dir):
    """Scan the data directory for available recordings (top-level folders)."""
    recordings = {}
    if data_dir is None or not os.path.isdir(data_dir):
        print(f"Warning: data_dir '{data_dir}' not found or not a directory.")
        return recordings

    try:
        for item_name in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item_name)
            # Only include directories as recordings
            if os.path.isdir(item_path):
                # Use the directory name as both key and value
                recordings[item_name] = item_name
    except OSError as e:
        print(f"Error scanning data_dir '{data_dir}': {e}")

    return recordings

# --- Callback Functions (must use st.session_state) ---

def on_recording_change():
    """Update session based on the value selected in the recording selectbox."""
    # The new value chosen by the user is ALREADY in st.session_state.selected_recording_key
    # We just need to update our separate st.session_state.session variable
    new_session = st.session_state.selected_recording_key

    # Update the main session variable used by other parts of the app
    st.session_state.session = new_session

    # Perform other actions needed when the session changes
    st.session_state.oldest_timestamp = None # Force range recalculation
    st.session_state.newest_timestamp = None
    st.session_state.need_to_display_recent = True # Show most recent for new recording
    st.session_state.last_displayed_timestamp = None # Force display update
    
    # Explicitly reset timestamps in jiffyget
    import jiffyget
    jiffyget.timestamps = None
    
    set_autoplay("None")
    st.session_state.in_playback_mode = True # Default to playback when changing recording
    if st.session_state.get('video_placeholder'):
         display_most_recent_image()

def toggle_rt_capture():
    """Toggle real-time capture on/off"""
    st.session_state.rt_capture = not st.session_state.rt_capture

    if st.session_state.rt_capture:
        st.session_state.in_playback_mode = False
        st.session_state.browsing_saved_images = False
        st.session_state.date = datetime.now().date()

        # Get HTTP client
        if not hasattr(st.session_state, 'http_client'):
            st.session_state.http_client = JiffyCamClient(st.session_state.http_server_url)
        
        # Force a connection check
        if not st.session_state.http_client.check_connection(force=True):
            st.error(f"Failed to connect to JiffyCam server: {st.session_state.http_client.last_error}")
            st.session_state.rt_capture = False
            return
            
        # Start capture
        st.session_state.http_client.start_capture()
    else:
        # Stop capture
        if hasattr(st.session_state, 'http_client'):
            st.session_state.http_client.stop_capture()
        st.session_state.in_playback_mode = True

def on_date_change():
    """Handle date picker change."""
    set_autoplay("None")

    st.session_state.browsing_date = st.session_state.date
    st.session_state.in_playback_mode = True 
    
    # Reset jiffyget timestamps cache for the new date
    import jiffyget
    jiffyget.timestamps = None
    
    # Make sure step_direction is set for proper playback control initialization
    if 'step_direction' not in st.session_state or not st.session_state.step_direction:
        st.session_state.step_direction = "down"

    # Count frames for the new date
    try:
        browse_date = st.session_state.date
        browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
        
        # Get timestamps for this date
        timestamps = jiffyget.get_timestamps(
            st.session_state.cam_name, 
            st.session_state.session, 
            st.session_state.data_dir, 
            browse_date_posix
        )
        
        # Count the number of frames
        if timestamps is not None:
            st.session_state.frames_detected = len(timestamps)
        else:
            st.session_state.frames_detected = 0
    except Exception as e:
        print(f"Error counting frames: {str(e)}")
        st.session_state.frames_detected = 0
    
    # Instead of checking need_to_display_recent flag, 
    # always display the most recent image for the selected date
    # First set time to end of day to find the latest image
    st.session_state.hour = 23
    st.session_state.minute = 59
    st.session_state.second = 59
    
    # Search backwards ("down") for the latest image of the day
    update_image_display(direction="down")
    
    # Reset flag to avoid duplicate loading
    st.session_state.need_to_display_recent = False

def on_prev_button(stopAuto=True):
    """Handle previous image button click."""
    if(stopAuto):
        set_autoplay("None")
    st.session_state.in_playback_mode = True
    # Decrement time logic
    tweek_time('down')

    # Fetch placeholders from session_state inside the update function
    if(stopAuto):
        st.session_state.step_direction = "down"    # let rerun handle the update  
    else:
        update_image_display(direction="down")

def on_next_button(stopAuto=True):
    """Handle next image button click."""
    if(stopAuto):
        set_autoplay("None")
    st.session_state.in_playback_mode = True
    # Increment time logic
    tweek_time('up')

    # Fetch placeholders from session_state inside the update function
    if(stopAuto):
        st.session_state.step_direction = "up"      # let rerun handle the update
    else:
        update_image_display(direction="up")

def tweek_time(direction):
    """Tweek the time by 1 second in the given direction."""
    dt = datetime.combine(st.session_state.date, datetime_time(st.session_state.hour, st.session_state.minute, st.session_state.second))
    dt += timedelta(seconds=1) if direction == "up" else -timedelta(seconds=1)
    st.session_state.browsing_date = dt.date()
    st.session_state.hour = dt.hour
    st.session_state.minute = dt.minute
    st.session_state.second = dt.second

def change_day(direction):
    """Change the date by one day in the specified direction.
    
    Args:
        direction: "next" or "prev" indicating which day to navigate to
    """
    # Reset jiffyget timestamps cache before changing day
    import jiffyget
    jiffyget.timestamps = None
    
    # Get current date from the session state
    current_date = st.session_state.date
    
    # Calculate the new date based on direction
    if direction == "next":
        new_date = current_date + timedelta(days=1)
    else:  # "prev"
        new_date = current_date - timedelta(days=1)
    
    # Check if the new date is within min/max range
    min_date = st.session_state.oldest_timestamp.date() if st.session_state.oldest_timestamp else None
    max_date = max(datetime.now().date(), st.session_state.newest_timestamp.date() if st.session_state.newest_timestamp else datetime.now().date())
    
    if min_date and new_date < min_date:
        new_date = min_date
    if max_date and new_date > max_date:
        new_date = max_date
    
    # Set step direction to ensure playback controls work after day change
    st.session_state.step_direction = "down"
    
    # Update the date in session state
    st.session_state.date = new_date
    # This alone won't trigger on_date_change since we're calling it explicitly from the button handlers

def toggle_live_pause():
    """Handle Live/Pause button click."""
    
    if st.session_state.in_playback_mode:
        # Go Live
        # Check if current browsing date is today
        current_date = datetime.now().date()
        browsing_date = st.session_state.get('browsing_date') or current_date
        
        if browsing_date != current_date:
            # Don't toggle to live mode if date isn't today
            return
            
        # Update browsing date/time to current (will be reflected in loop)
        current_time = datetime.now()
        st.session_state.browsing_date = current_time.date()
        st.session_state.in_playback_mode = False
        
        # If in HTTP mode, make sure rt_capture is turned on and check connection
        if st.session_state.get('use_http_mode', False):
            st.session_state.rt_capture = True
            
            # Force a connection check if we have a client
            if hasattr(st.session_state, 'http_client'):
                st.session_state.http_client.check_connection(force=True)
        
        # Let the main loop update hour/min/sec/slider when live
    else:
        # Pause
        st.session_state.in_playback_mode = True
        
        # No longer immediately reset display FPS - let it decay naturally
        # when no more frames are being displayed

def on_pause_button():
    """Handle pause button click."""
    set_autoplay("None")
    st.session_state.in_playback_mode = True
    st.session_state.step_direction = "None"
    
    # Force display update when pausing
    st.session_state.last_displayed_timestamp = None
    
    # No longer immediately reset display FPS - let it decay naturally
    # when no more frames are being displayed
    #print(f"on_pause_button: {get_is_capturing()}")

def on_fast_reverse_button():
    """Handle fast reverse button click."""
    set_autoplay("reverse")
    tweek_time('down')
    # Set to playback mode like other buttons do
    st.session_state.in_playback_mode = True

def on_fast_forward_button():
    """Handle fast forward button click."""
    set_autoplay("forward")
    tweek_time('up')
    # Set to playback mode like other buttons do
    st.session_state.in_playback_mode = True

def on_prev_day_button():
    """Handle previous day button click."""
    set_autoplay("None")
    
    # In-line function to force playback controls to work right after day change
    st.session_state.in_playback_mode = True
    
    # Change the day
    change_day("prev")
    
    # Directly trigger the date change handler instead of waiting for callback
    on_date_change()
    
def on_next_day_button():
    """Handle next day button click."""
    set_autoplay("None")
    
    # In-line function to force playback controls to work right after day change
    st.session_state.in_playback_mode = True
    
    # Change the day
    change_day("next")
    
    # Directly trigger the date change handler instead of waiting for callback
    on_date_change()

# --- UI Update Functions ---
def new_image_display(frame):
    if frame is None:
        return

    """Display a new image based on the current date and time."""
    # Track display FPS whenever we display a new frame, even in playback mode
    current_time = time.time()
    st.session_state.display_frame_count += 1
    
    # Calculate display FPS once per second
    if current_time - st.session_state.last_display_time >= 1.0:
        st.session_state.display_fps = st.session_state.display_frame_count / (current_time - st.session_state.last_display_time)
        st.session_state.display_frame_count = 0
        st.session_state.last_display_time = current_time
    
    video_placeholder = st.session_state.video_placeholder
    video_placeholder.image(frame, channels="BGR", use_container_width=True)

    timearrow_placeholder = st.session_state.timearrow_placeholder
    ta_img = generate_timeline_arrow()
    timearrow_placeholder.image(ta_img, channels="RGB", use_container_width=True)
    
def update_image_display(direction=None):
    """Update the image display based on the current date and time."""
    #print(f"update_image_display: {direction}")

    # Fetch placeholders from session state
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    time_display = st.session_state.time_display

    # --- Get necessary state for jiffyget ---
    session = st.session_state.get('session', 'Default') # Safely get session
    data_dir = st.session_state.get('data_dir', 'JiffyData') # Get data_dir from config
    browse_date = st.session_state.browsing_date # Get the date being browsed
    # --- End state retrieval ---

    # Find the closest image using jiffyget directly
    time_posix = datetime.combine(browse_date, datetime.min.time()) + timedelta(hours=st.session_state.hour, minutes=st.session_state.minute, seconds=st.session_state.second)
    time_posix = float(time_posix.timestamp())

    closest_image, timestamp, eof = jiffyget(
        time_posix,
        st.session_state.cam_name,
        session,       # Pass session variable
        data_dir,      # Pass data_dir variable
        direction
    )
    if(eof):
        set_autoplay("None")

    success = False
    if closest_image is not None:
        try:
            # Store frame and actual timestamp
            st.session_state.last_frame = closest_image.copy()
            dts = datetime.fromtimestamp(timestamp/1000)
            st.session_state.actual_timestamp = dts

            # Update time state to match the found image
            st.session_state.hour = dts.hour
            st.session_state.minute = dts.minute
            st.session_state.second = dts.second
            st.session_state.browsing_date = dts.date() # Keep browsing date in sync with found image

            # Update UI placeholders
            new_image_display(closest_image)

            # Update time display text
            time_display.markdown(
                f'<div class="time-display">{format_time_12h(dts.hour, dts.minute, dts.second)}</div>',
                unsafe_allow_html=True
            )

            # Update status text
            status_placeholder.markdown(f"<div style='padding: 5px 0;'>Viewing: {dts.strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
            st.session_state.status_message = f"Viewing: {dts.strftime('%Y-%m-%d %H:%M:%S')}" # Update internal status too
            success = True

        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            status_placeholder.error(f"Error displaying image: {str(e)}")
            st.session_state.status_message = f"Error displaying image: {str(e)}"
    
    if not success:
        set_autoplay("None")

        # No image found or error displaying
        status_placeholder.markdown("<div style='padding: 5px 0;'>No image found near specified time</div>", unsafe_allow_html=True)
        st.session_state.status_message = f"No image found near specified time"
        # Clear the last frame and the display placeholder
        st.session_state.last_frame = None
        video_placeholder.empty()

    return success

def display_most_recent_image():
    """Display the most recent image for the current camera."""

    # Use current time as starting point
    timestamp = datetime.now()
    st.session_state.hour = timestamp.hour
    st.session_state.minute = timestamp.minute
    st.session_state.second = timestamp.second
    st.session_state.browsing_date = timestamp.date()
    # st.session_state.date = timestamp.date() # REMOVED: Cannot modify widget state directly

    # Make sure we're in playback mode
    st.session_state.in_playback_mode = True

    # Make sure step_direction is set for proper playback control initialization
    if 'step_direction' not in st.session_state or not st.session_state.step_direction:
        st.session_state.step_direction = "down"
    
    # Reset jiffyget timestamps to force reload for current date
    import jiffyget
    jiffyget.timestamps = None
    
    # Count frames for today
    try:
        browse_date = timestamp.date()
        browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
        
        # Get timestamps for this date
        timestamps = jiffyget.get_timestamps(
            st.session_state.cam_name, 
            st.session_state.session, 
            st.session_state.data_dir, 
            browse_date_posix
        )
        
        # Count the number of frames
        if timestamps is not None:
            st.session_state.frames_detected = len(timestamps)
        else:
            st.session_state.frames_detected = 0
    except Exception as e:
        print(f"Error counting frames: {str(e)}")
        st.session_state.frames_detected = 0
    
    # Search backwards ("down") for the latest image before now
    update_image_display(direction="down")
    st.session_state.need_to_display_recent = False # Mark as done

# --- UI Building Functions ---
def build_sidebar():
    """Build the sidebar UI elements."""
    # Setup main sections
    with st.sidebar:
        st.title("JiffyCam")
        
        # Recording Selection
        st.header("Recordings")
        # Get data_dir from session state (e.g., loaded from config)
        data_dir = st.session_state.get('data_dir', 'JiffyData') # Default if not set

        # Scan for recordings only if not already cached in session state
        if 'available_recordings' not in st.session_state or not st.session_state.available_recordings:
            st.session_state.available_recordings = get_available_recordings(data_dir)

        # Use the cached/scanned recordings
        available_recordings = st.session_state.available_recordings

        recording_keys = list(available_recordings.keys())

        if not recording_keys:
            st.warning(f"No recordings found in '{data_dir}'.")
            # Initialize session to avoid errors later, assume cam_name is set elsewhere
            if 'session' not in st.session_state: st.session_state.session = "DefaultRecording"
        else:
            # Initialize the selectbox widget state key if it doesn't exist
            # This ensures the widget has a starting value without overriding user selection later
            if 'selected_recording_key' not in st.session_state:
                # Try to use the current session if it's valid, otherwise default to first key
                current_session = st.session_state.get('session', '')
                if current_session in recording_keys:
                    st.session_state.selected_recording_key = current_session
                else:
                    st.session_state.selected_recording_key = recording_keys[0]
                    # If we defaulted the widget key, also update the main session state
                    if st.session_state.session != st.session_state.selected_recording_key:
                         st.session_state.session = st.session_state.selected_recording_key

            st.selectbox(
                "Select Recording",
                options=recording_keys,
                key="selected_recording_key",
                on_change=on_recording_change,
                help="Select the recording session to view.",
                label_visibility="collapsed"
            )

        # Status Placeholders
        st.header("Status")
        status_placeholder = st.empty()
        server_status_placeholder = st.empty()
        
        # Add metrics for FPS
        st.subheader("Performance")
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            capture_fps_placeholder = st.empty()
        with metrics_cols[1]:
            display_fps_placeholder = st.empty()
        with metrics_cols[2]:
            frames_detected_placeholder = st.empty()
            
        error_placeholder = st.empty()

    return status_placeholder, error_placeholder, server_status_placeholder, capture_fps_placeholder, display_fps_placeholder, frames_detected_placeholder

def generate_timeline_image(width=1200, height=60):
    """Generate an image for the timeline bar based on available data."""
    #print(f"generate_timeline_image: {width}, {height}")
    session = st.session_state.get('session', 'Default')
    data_dir = st.session_state.get('data_dir', 'JiffyData')

    # Construct base path safely
    browse_date_posix = int(time.mktime(st.session_state.date.timetuple()))
    timestamps = get_locations(st.session_state.cam_name, session, data_dir, browse_date_posix*1000)
    if(timestamps is None):
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Create a blank image (dark gray background)
    background_color = (51, 51, 51)  # Dark gray
    mark_color = (255, 0, 0)  # Red in BGR format (Blue = 0, Green = 0, Red = 255) 
    hour_marker_color = (180, 180, 180)  # Light gray for hour markers
    special_marker_color = (255, 255, 0)  # Yellow in BGR format
    text_color = (220, 220, 220)  # Brighter light gray for text
    
    # Add equal padding for top and bottom margins
    text_padding = 20  # Pixels below timeline for text
    top_margin = 0  # Match top margin to text padding (was 12)
    timeline_y_start = top_margin  # Timeline now starts below top margin
    timeline_y_end = timeline_y_start + height
    total_height = timeline_y_end + text_padding  # Total image height
    
    # Start with a blank transparent image (including space for labels and top margin)
    rounded_img = np.zeros((total_height, width, 3), dtype=np.uint8)
    
    # Add radius for rounded corner effect (small radius)
    radius = int(height/2)
    
    # Draw a rounded rectangle for the timeline (top portion only)
    cv2.rectangle(rounded_img, (radius, timeline_y_start), (width-radius, timeline_y_end), background_color, -1)
    cv2.rectangle(rounded_img, (0, timeline_y_start+radius), (width, timeline_y_end-radius), background_color, -1)
    
    # Draw the four corners
    cv2.circle(rounded_img, (radius, timeline_y_start+radius), radius, background_color, -1)
    cv2.circle(rounded_img, (width-radius, timeline_y_start+radius), radius, background_color, -1)
    cv2.circle(rounded_img, (radius, timeline_y_end-radius), radius, background_color, -1)
    cv2.circle(rounded_img, (width-radius, timeline_y_end-radius), radius, background_color, -1)
    
    timeline_img = rounded_img
    
    # Define label positions in advance
    label_positions = {
        0: "12am",
        6: "6am",
        12: "12pm",
        18: "6pm",
        24: "12am"
    }
    
    # Set font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.33, min(0.5, width / 2000))  # Scale text based on timeline width
    font_thickness = 1
    
    # Add hour markers and time labels (24 hours)
    for hour in range(25):  # 0 to 24 hours (include 24 for end of day)
        position = hour / 24.0  # Convert hour to percentage of day
        x_pos = int(position * width)
        
        # Adjust thickness and height based on marker type
        if hour % 6 == 0:  # More prominent markers at 6-hour intervals
            marker_height = int(height * 0.7)  # 70% of total height
            thickness = max(1, int(width/1200))
            alpha = 0.5  # 50% opacity
            
            # Get time label for this hour
            time_label = label_positions.get(hour, "")
                
            if time_label:
                # Calculate text size to center it
                text_size, baseline = cv2.getTextSize(time_label, font, font_scale, font_thickness)
                
                # Center text below marker
                text_x = x_pos - text_size[0] // 2
                # Position text below timeline with enough room for descenders
                text_y = timeline_y_end + baseline + 8  # Adjusted positioning to avoid clipping descenders
                
                # Handle edge cases
                if hour == 0:  # Leftmost label (12am)
                    text_x = max(text_x, 2)  # Keep a minimum of 2px from left edge
                elif hour == 24:  # Rightmost label (12am)
                    text_x = min(text_x, width - text_size[0] - 2)  # Keep from right edge
                
                # Draw a slightly darker rectangle behind the text
                # Make the background rectangle taller to accommodate descenders
                cv2.rectangle(timeline_img, 
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + baseline + 2),
                            (30, 30, 30), -1)
                
                # Draw the text
                cv2.putText(timeline_img, time_label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        else:
            marker_height = int(height * 0.4)  # 40% of total height
            thickness = max(1, int(width/1500))
            alpha = 0.3  # 30% opacity
            
        # Skip markers at the very edges but still draw text
        if x_pos < radius or x_pos > width - radius:
            continue
            
        # Draw line from bottom
        start_y = timeline_y_end - 1
        end_y = timeline_y_end - marker_height
        
        # Draw semi-transparent hour marker lines
        overlay = timeline_img.copy()
        cv2.line(overlay, (x_pos, start_y), (x_pos, end_y), hour_marker_color, thickness)
        cv2.addWeighted(overlay, alpha, timeline_img, 1-alpha, 0, timeline_img)

    # Add marks for timestamps
    if timestamps:
        for position in timestamps:
            # Calculate pixel position
            x_pos = int(position * width)
            # Draw a red vertical line with alpha blending
            cv_thickness = max(1, int(width/1000))  # Scale thickness with width
            alpha = 0.8  # 80% opacity (slightly more visible than before)
            
            # Draw semi-transparent line
            overlay = timeline_img.copy()
            cv2.line(overlay, (x_pos, timeline_y_start), (x_pos, timeline_y_end), mark_color, cv_thickness)
            cv2.addWeighted(overlay, alpha, timeline_img, 1-alpha, 0, timeline_img)

    return timeline_img

def generate_timeline_arrow(width=1200, height=24):
    # Add special timestamp marker for current time
    if hasattr(st.session_state, 'hour') and hasattr(st.session_state, 'minute') and hasattr(st.session_state, 'second'):
        # Calculate position as percentage of day
        current_seconds = st.session_state.hour * 3600 + st.session_state.minute * 60 + st.session_state.second
        day_percentage = current_seconds / (24 * 60 * 60)
        #print(f"day_percentage: {day_percentage}")
        x_pos = int(day_percentage * width)
        radius = int(height/2)
        timeline_y_start = height -1
        timeline_y_end = 1
        timeline_img = np.zeros((height, width, 3), dtype=np.uint8)
        special_marker_color = (255, 255, 0)  # Yellow in BGR format

        # Skip if too close to edges
        if x_pos >= radius and x_pos <= width - radius:
            # Draw special marker (thicker line with yellow color)
            cv_thickness = max(2, int(width/800))  # Thicker than regular markers
            
            # Draw semi-transparent triangle ABOVE timeline (pointing down)
            triangle_height = int(height * 0.75)  # Adjusted to 75% of top margin height
            triangle_width = int(height * 0.75)  # Slightly wider triangle
            
            # Triangle points (pointing down)
            bottom_point = (x_pos, timeline_y_start - 1)  # Just above timeline
            left_point = (x_pos - triangle_width//2, timeline_y_start - triangle_height)
            right_point = (x_pos + triangle_width//2, timeline_y_start - triangle_height)
            
            # Draw the triangle
            triangle_pts = np.array([bottom_point, left_point, right_point], np.int32)
            triangle_pts = triangle_pts.reshape((-1, 1, 2))
            overlay = timeline_img.copy()
            cv2.fillPoly(overlay, [triangle_pts], special_marker_color)
            cv2.addWeighted(overlay, 0.8, timeline_img, 0.2, 0, timeline_img)

    return timeline_img

def on_timeline_click(coords):
    """Handle clicks on the timeline image."""
    set_autoplay("None")
    if coords is None or 'x' not in coords:
        return
    
    # Calculate the time position based on click position
    x = coords['x']
    width = coords.get('width', 1)
    if width <= 0:
        return  # Avoid division by zero
    
    # Calculate the position as a percentage of the day
    day_percentage = x / width
    day_percentage = max(0, min(1, day_percentage))
    
    # Convert to hours, minutes, seconds
    total_seconds = int(day_percentage * 24 * 60 * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    # Update state
    st.session_state.hour = hours
    st.session_state.minute = minutes
    st.session_state.second = seconds
    st.session_state.in_playback_mode = True
    set_autoplay("none")
    

def build_main_area():
    """Create the main UI area elements and return placeholders."""
    # Apply CSS for main area styling
    st.markdown("""
    <style>
    /* General layout */

    .block-container { padding-top: 0.7rem !important; padding-bottom: 0.7rem !important; }
    
    h1, h2, h3 { margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; padding: 0 !important; }
    hr { margin: 0 !important; padding: 0 !important; }
    /* Time Display */
    .time-display { font-size: 1.6rem; font-weight: 400; text-align: center; font-family: "Source Sans Pro", sans-serif; background: transparent; color: #333; border-radius: 5px; padding: 2px; margin: 2px 0; }
    @media (prefers-color-scheme: dark) { .time-display { color: #e0e0e0 !important; } }
    /* Date Picker */
    div[data-testid="stDateInput"] { margin: 10px 0 0 0 !important; padding: 6px 0 !important; display: flex !important; flex-direction: column !important; justify-content: center !important; }
    div[data-testid="stDateInput"] > div { margin-bottom: 0 !important; display: flex !important; align-items: center !important; height: 32px !important; }
    div[data-testid="stDateInput"] input { padding: 2px 0px !important; height: 32px !important; font-size: 14px !important; background-color: #2e2e2e !important; color: #fff !important; border: 1px solid #555 !important; border-radius: 5px !important; text-align: center !important; font-weight: bold !important; margin: 0 !important; }
    div[data-testid="stDateInput"] svg { fill: #fff !important; }

    /* Live button styling */
    button[data-testid="baseButton-primary"]:has(div:contains("Live")) {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
        color: white !important;
    }
    
    /* Day navigation buttons styling */
    button[data-testid="baseButton-secondary"]:has(div:contains("◀")),
    button[data-testid="baseButton-secondary"]:has(div:contains("▶")) {
        padding: 0 !important;
    }
    
    /* Centered controls container with margin for better positioning */
    .top-controls {
        max-width: 600px;
        margin: 25px auto 0 auto;
    }
    
    /* Remove all the vertical align stuff */
    </style>
    """, unsafe_allow_html=True)

    # --- Layout ---

    # Centered controls container
    st.markdown("<div class='top-controls'>", unsafe_allow_html=True)

    # Date Picker with navigation buttons in a single row with vertical alignment
    date_cols = st.columns([0.7, 5, 0.7])
    
    with date_cols[0]:  # Previous day button
        # Add a simple empty space above the button to create consistent positioning
        st.write("")
        st.button("◀", key="prev_day_button", use_container_width=True, help="Previous Day",
                 on_click=on_prev_day_button)
    
    with date_cols[1]:  # Date Picker
        min_date, max_date = None, datetime.now().date()
        if st.session_state.oldest_timestamp: min_date = st.session_state.oldest_timestamp.date()
        if st.session_state.newest_timestamp: max_date = max(max_date, st.session_state.newest_timestamp.date())
        st.date_input("Date", key="date",
                       # Callback no longer needs args
                       on_change=on_date_change,
                       help="Select date", label_visibility="collapsed",
                       min_value=min_date, max_value=max_date)
    
    with date_cols[2]:  # Next day button
        # Add a simple empty space above the button to create consistent positioning
        st.write("")
        st.button("▶", key="next_day_button", use_container_width=True, help="Next Day",
                 on_click=on_next_day_button)

    # Time controls row
    time_cols = st.columns([3, 1, 1, 1, 1, 1, 0.2, 1])
    with time_cols[0]: # Time Display Placeholder
        time_display = st.empty()
        # Set initial display value
        #time_display.markdown(f'<div class="time-display">{format_time_12h(st.session_state.hour, st.session_state.minute, st.session_state.second)}</div>', unsafe_allow_html=True)
    with time_cols[1]: # Fast Reverse Button
        st.button("⏪", key="fast_reverse_button", use_container_width=True, help="Fast Reverse",
                  on_click=on_fast_reverse_button)
    with time_cols[2]: # Prev Button
        st.button("◀️", key="prev_button", use_container_width=True, help="Previous",
                  # Callback no longer needs args
                  on_click=on_prev_button)
    with time_cols[3]: # Pause Button
        st.button("⏸️", key="pause_button", use_container_width=True, help="Pause",
                  on_click=on_pause_button)
    with time_cols[4]: # Next Button
        st.button("▶️", key="next_button", use_container_width=True, help="Next",
                  # Callback no longer needs args
                  on_click=on_next_button)
    with time_cols[5]: # Fast Forward Button
        st.button("⏩", key="fast_forward_button", use_container_width=True, help="Fast Forward",
                  on_click=on_fast_forward_button)
    with time_cols[6]: # Separator
        st.markdown('<div style="width:1px;background-color:#555;height:32px;margin:0 auto;"></div>', unsafe_allow_html=True)
    with time_cols[7]: # Live/Pause Button
        # Check if client is connected and status is available
        client_connected = False
        server_session = None
        if hasattr(st.session_state, 'http_client'):
            client = st.session_state.http_client
            if client.is_connected():
                client_connected = True
                # Attempt to get server session from last known status
                if client.last_status:
                    server_session = client.last_status.get('session') # Safely get session

        # Get UI session
        ui_session = st.session_state.get('session')

        # Determine if live button should be disabled
        live_disabled = True # Default to disabled
        if client_connected and server_session is not None and ui_session is not None:
            if server_session == ui_session:
                # Only enable if connected, sessions match AND the date is today
                current_date = datetime.now().date()
                browsing_date = st.session_state.get('browsing_date') or current_date
                if browsing_date == current_date:
                    live_disabled = False
                else:
                    live_disabled = True
                    button_help = "Live view is only available for current date"

        # is_capturing reflects if rt_capture is enabled and we have a connection
        # This doesn't determine enabled state anymore, but affects button style/help
        # is_capturing = st.session_state.rt_capture and client_connected

        in_playback = st.session_state.in_playback_mode

        # Always display "Live" text, but with different styles
        button_text = "Live"

        if in_playback:
            # Not in live mode - unfilled button
            if not 'button_help' in locals():  # Don't override if already set for date mismatch
                button_help = "Switch to live view" if not live_disabled else f"Server not capturing session '{ui_session}'"
            button_type = "secondary"  # Gray (unfilled) button
        else:
            # In live mode - filled red button
            button_help = "Pause live view"
            button_type = "primary"    # Red (filled) button via CSS

        st.button(button_text, key="live_btn", use_container_width=True, help=button_help,
                on_click=toggle_live_pause, disabled=live_disabled, type=button_type)

    # Time Arrow above timeline
    timearrow_placeholder = st.empty()
    #timearrow_img = generate_timeline_arrow()
    timearrow_img = np.zeros((24, 1200, 3), dtype=np.uint8)
    timearrow_placeholder.image(timearrow_img, channels="RGB", use_container_width=True)

    # Generate timeline image with appropriate width and increased height (50% taller)
    timeline_img = generate_timeline_image(width=1200, height=60)  # Increased from 40 to 60
    
    # Store previous click coordinates to avoid duplicate processing
    prev_coords = st.session_state.get('prev_timeline_coords', None)
    
    # Display the clickable image
    clicked_coords = streamlit_image_coordinates(
        timeline_img, 
        key="timeline_bar",
        use_column_width=True
    )
    #st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle timeline click only if it's a new click (different from previous)
    if clicked_coords and 'x' in clicked_coords:
        # Convert to tuple for comparison (dictionaries aren't hashable)
        current_click = (clicked_coords.get('x'), clicked_coords.get('y'))
        previous_click = prev_coords if prev_coords else (-1, -1)
        
        if current_click != previous_click:
            # This is a new click
            on_timeline_click(clicked_coords)
            # Save current click to avoid reprocessing
            st.session_state.prev_timeline_coords = current_click

    st.markdown("</div>", unsafe_allow_html=True) # Close centered container

    # Create Video Placeholder *after* the controls container
    video_placeholder = st.empty() 
     # reduces flicker but causes double-jump on timeline click:
    #if st.session_state.last_frame is not None and not st.session_state.in_playback_mode:
    #    video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)

    # Return placeholders needed outside this build function (by callbacks and main loop)
    return video_placeholder, time_display, timearrow_placeholder

#def update_server_state(frame):
#    if frame is not None: 
#        with no_rerun:      
#            with server_state_lock["last_frame"]:
#                server_state.last_frame = frame
#                server_state.timestamp = datetime.now()

# --- Main UI Update Loop (runs in jiffycam.py) ---
def run_ui_update_loop():
    """The main loop to update the UI based on capture state and interactions."""
    # Fetch placeholders from session state at the start of the loop
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    error_placeholder = st.session_state.error_placeholder
    time_display = st.session_state.time_display
    server_status_placeholder = st.session_state.server_status_placeholder
    capture_fps_placeholder = st.session_state.capture_fps_placeholder
    display_fps_placeholder = st.session_state.display_fps_placeholder
    frames_detected_placeholder = st.session_state.frames_detected_placeholder

    # --- Initial Image Display --- 
    is_capturing = False
    
    if hasattr(st.session_state, 'http_client'):
        # Only check connection status if we're not in playback mode
        if not st.session_state.in_playback_mode:
            has_connected_client = st.session_state.http_client.is_connected()
            is_capturing = st.session_state.rt_capture and has_connected_client
        else:
            # In playback mode, don't check connection
            is_capturing = st.session_state.rt_capture
    
    if st.session_state.need_to_display_recent and not is_capturing:
        display_most_recent_image() # Fetches placeholders from session_state
    elif st.session_state.last_frame is not None:
        update_image_display(st.session_state.step_direction)

    # Initialize server status update counter
    server_status_update_time = 0

    while True:
        try:
            heartbeat()

            # Update server connection status display (every 2 seconds)
            current_time = time.time()
            if current_time - server_status_update_time > 2:
                server_status_update_time = current_time
                connection_status = "Disconnected"
                server_session = "None"
                capture_fps = 0
                
                if hasattr(st.session_state, 'http_client'):
                    client = st.session_state.http_client
                    if client.is_connected():
                        connection_status = "Connected"
                        if client.last_status:
                            server_session = client.last_status.get('session', 'None')
                            is_capturing = client.last_status.get('capturing', False)
                            capture_fps = client.last_status.get('fps', 0)
                            if is_capturing:
                                connection_status = "Recording"
                
                # Store capture FPS in session state
                st.session_state.capture_fps = capture_fps
                
                # Count frames for current date
                if current_time - st.session_state.last_frames_count_update > 5:  # Update every 5 seconds
                    st.session_state.last_frames_count_update = current_time
                    try:
                        # Get current date in posix milliseconds
                        if st.session_state.in_playback_mode:
                            browse_date = st.session_state.browsing_date
                        else:
                            browse_date = datetime.now().date()
                        
                        browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
                        
                        # Get timestamps for this date
                        import jiffyget
                        timestamps = jiffyget.get_timestamps(
                            st.session_state.cam_name, 
                            st.session_state.session, 
                            st.session_state.data_dir, 
                            browse_date_posix
                        )
                        
                        # Count the number of frames
                        if timestamps is not None:
                            st.session_state.frames_detected = len(timestamps)
                        else:
                            st.session_state.frames_detected = 0
                    except Exception as e:
                        print(f"Error counting frames: {str(e)}")
                        st.session_state.frames_detected = 0
                
                # Display server status
                status_html = f"""
                <div style="padding: 5px 0;">
                    <div><b>Recording:</b> <span style="color: {'blue' if server_session != 'None' else 'gray'}">{server_session if server_session != 'None' else 'Idle'}</span></div>
                </div>
                """
                server_status_placeholder.markdown(status_html, unsafe_allow_html=True)
                
                # Update FPS metrics
                # Only show Capture FPS when connected to and viewing live data
                is_viewing_live = False
                if (not st.session_state.in_playback_mode and 
                    hasattr(st.session_state, 'http_client')):
                    client = st.session_state.http_client
                    is_viewing_live = (
                        client.is_connected() and
                        client.last_status and
                        server_session == st.session_state.session
                    )
                
                if is_viewing_live:
                    capture_fps_placeholder.metric("Capture FPS", f"{st.session_state.capture_fps:.1f}")
                else:
                    # Reset and hide Capture FPS when not viewing live
                    st.session_state.capture_fps = 0
                    capture_fps_placeholder.metric("Capture FPS", "0.0")
                
                # Set display FPS to 0 when in playback mode and no frames displayed recently
                if st.session_state.in_playback_mode and (current_time - st.session_state.last_display_time > 2.0):
                    st.session_state.display_fps = 0
                    
                display_fps_placeholder.metric("Display FPS", f"{st.session_state.display_fps:.1f}")
                frames_detected_placeholder.metric("Detections", st.session_state.frames_detected)

            # Check for errors first
            check_errors = False
            
            if hasattr(st.session_state, 'http_client'):
                # Only check connection if we're not in playback mode
                if (st.session_state.rt_capture and 
                    not st.session_state.in_playback_mode and
                    not st.session_state.http_client.is_connected()):
                    
                    error_placeholder.error(st.session_state.http_client.last_error or "Connection lost to JiffyCam server")
                    status_placeholder.markdown("<div style='padding: 5px 0;'>Status: Connection Error</div>", unsafe_allow_html=True)
                    st.session_state.status_message = "Status: Connection Error"
                    check_errors = True
            
            if check_errors:
                # Keep last frame visible on error if possible
                break # Exit loop
        
            # Check if we're capturing
            is_capturing = False
            
            if hasattr(st.session_state, 'http_client'):
                # Only check connection if we're not in playback mode
                if not st.session_state.in_playback_mode:
                    has_connected_client = st.session_state.http_client.is_connected()
                    is_capturing = st.session_state.rt_capture and has_connected_client
                else:
                    # In playback mode, just use rt_capture flag without checking connection
                    is_capturing = st.session_state.rt_capture

            current_status = st.session_state.get("status_message", "")
            new_status = current_status # Default to no change

            if is_capturing:
                # Get the latest frame from HTTP client
                if hasattr(st.session_state, 'http_client'):
                    # Only get a new frame from the server if we're in live view mode
                    # This prevents unnecessary HTTP requests when in playback mode
                    if not st.session_state.in_playback_mode:
                        frame = st.session_state.http_client.get_last_frame()
                        if frame is not None:
                            st.session_state.last_frame = frame

                if st.session_state.in_playback_mode:
                    # Playback Mode - Don't make any HTTP requests
                    # Ensure paused frame is displayed
                    if st.session_state.last_frame is not None:
                        ts = st.session_state.actual_timestamp
                        new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback Mode"
                    else:
                        new_status = "Playback Mode (No Frame)"
                        video_placeholder.empty() # Clear if no frame to show

                else: # Live View Mode (Capture Running)
                    if hasattr(st.session_state, 'http_client'):
                        # In live view mode, always get the latest frame
                        frame, fps, _, _ = st.session_state.http_client.get_frame()
                        
                        # Update status information periodically
                        #status_updated = st.session_state.http_client.update_status_if_needed()
                        
                        # Always update status to show we're using HTTP mode
                        if frame is not None:
                            status = st.session_state.http_client.last_status
                            if status:
                                fps = status.get('fps', 0)
                                frame_count = status.get('frame_count', 0)
                                new_status = f"Live View - FPS: {fps}, Frames: {frame_count}"
                            else:
                                new_status = "Live View"
                    
                    if frame is not None:
                        st.session_state.last_frame = frame
                        
                        # Update time display and slider for live view
                        current_time = datetime.now()
                        time_display.markdown(f'<div class="time-display">{format_time_12h(current_time.hour, current_time.minute, current_time.second)}</div>', unsafe_allow_html=True)
                        st.session_state.hour = current_time.hour
                        st.session_state.minute = current_time.minute
                        st.session_state.second = current_time.second
                        
                        # Display the frame
                        new_image_display(frame)

            else: # Capture Stopped
                if st.session_state.in_playback_mode:
                    # Playback Mode (Capture Stopped): Show paused frame
                    if st.session_state.last_frame is not None:
                        # Determine if we need to update the display
                        # We need to update if:
                        # 1. We're actively navigating (step_direction is set)
                        # 2. The image hasn't been displayed yet (timestamps don't match)
                        # 3. We just switched to playback mode (last_displayed_timestamp is None)
                        need_display_update = False
                        
                        if st.session_state.step_direction not in [None, "None"]:
                            need_display_update = True
                        elif 'last_displayed_timestamp' not in st.session_state or st.session_state.last_displayed_timestamp is None:
                            need_display_update = True
                        elif st.session_state.last_displayed_timestamp != st.session_state.actual_timestamp:
                            need_display_update = True
                        
                        if need_display_update:
                            # Only display if we need to update the image
                            new_image_display(st.session_state.last_frame)
                            st.session_state.last_displayed_timestamp = st.session_state.actual_timestamp
                            
                            # Reset step_direction to prevent continuous updates
                            if st.session_state.step_direction not in ["forward", "reverse"]:
                                st.session_state.step_direction = "None"

                        ts = st.session_state.actual_timestamp
                        new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback (Stopped)"
                    else:
                        new_status = "Playback (Stopped - No Frame)"
                        video_placeholder.empty()
                else:
                    # Not capturing and not in playback mode - show message to start capture
                    video_placeholder.info("Connect to JiffyCam server and start capture, or select a date/time to browse.")
                    if st.session_state.last_frame is not None:
                        new_image_display(st.session_state.last_frame)
                    new_status = "Status: Idle"

            # Update status placeholder with formatted status
            status_placeholder.markdown(f"<div style='padding: 5px 0;'>{new_status}</div>", unsafe_allow_html=True)
            st.session_state.status_message = new_status # Store new status

            # Main loop delay
            time.sleep(0.01)
            
        except Exception as e:
            # Handle any unexpected errors to prevent crashing
            error_message = f"Unexpected error: {str(e)}"
            print(error_message)
            error_placeholder.error(error_message)
            time.sleep(1)  # Wait a bit before continuing

    print("exit run_ui_update_loop!")
