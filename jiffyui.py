"""
jiffyui.py: Main UI component for JiffyCam browser

This module contains the main UI code for the JiffyCam browser interface.
"""

import os
import time
#import threading
from datetime import datetime, timedelta
from datetime import time as datetime_time
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import inspect

from jiffyget import (
    jiffyget, 
    get_timestamp_range, 
    get_active_sessions,
    get_session_port,
    get_timestamps,
    reset_timestamps
)
from jiffyvis import generate_timeline_image, generate_timeline_arrow, format_time_12h

# Import from extracted modules
try:
    from jiffyclient import JiffyCamClient
except ImportError:
    print("Warning: JiffyCamClient module not found. HTTP mode will be unavailable.")
    
# For normal rerun prevention in callback context
#no_rerun = threading.Lock()

# Import UI components (may be imported within individual functions as needed)
# import jiffyui_components

# following just to avoid error when importing YOLO
#import torch
#torch.classes.__path__ = [] # add this line to manually set it to empty. 

# Add threading module for compatibility
#import threading

# --- UI Helper Functions ---

def heartbeat():
    """Heartbeat to check if the UI is still running.""" 
    if(st.session_state.autoplay_direction == "up"):
        on_next_button(False)
        time.sleep(st.session_state.autoplay_interval)
    elif(st.session_state.autoplay_direction == "down"):
        on_prev_button(False)
        time.sleep(st.session_state.autoplay_interval)

    if(st.session_state.autoplay_step != None):
        on_navigation_button(st.session_state.autoplay_step, False)
        st.session_state.autoplay_step = None

def set_autoplay(direction):
    """Set autoplay direction."""
    #global autoplay_direction
    st.session_state.autoplay_direction = direction

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
                recordings[item_name] = item_path
    except Exception as e:
        print(f"Error scanning recordings: {str(e)}")
        
    return recordings

# --- Callback Functions (must use st.session_state) ---

def on_recording_change():
    """Update session based on the value selected in the recording selectbox."""
    # The new value chosen by the user is ALREADY in st.session_state.selected_recording_key
    # We just need to update our separate st.session_state.session variable
    new_session = st.session_state.selected_recording_key

    # Update the main session variable used by other parts of the app
    st.session_state.session = new_session
    #print(f"on_recording_change: {new_session}")

    # Check if the new session is active
    from jiffyget import get_active_sessions
    data_dir = st.session_state.get('data_dir', 'JiffyData')
    active_sessions = get_active_sessions(data_dir)
    
    # If the session is active, update the HTTP client
    if new_session in active_sessions:
        port = get_session_port(new_session, data_dir)
        if port:
            server_url = f"http://localhost:{port}"
            st.session_state.http_server_url = server_url
            
            try:
                # Update or create HTTP client
                if hasattr(st.session_state, 'http_client'):
                    # Update existing client
                    st.session_state.http_client.set_server_url(server_url)
                else:
                    # Create new client
                    from jiffyclient import JiffyCamClient
                    st.session_state.http_client = JiffyCamClient(server_url)
                
                # Try a test connection to verify server is actually responding
                if not st.session_state.http_client.check_connection(force=True):
                    print(f"Warning: Could not connect to server at {server_url}")
                    # Don't show errors here - just quietly fail to connect
            except Exception as e:
                print(f"Error connecting to server at {server_url}: {str(e)}")
                # Don't show errors here - just quietly fail to connect
    elif hasattr(st.session_state, 'http_client'):
        # If session is not active but we have a client, disconnect it
        # This is cleaner than keeping a stale connection
        try:
            st.session_state.http_client.disconnect()
        except Exception:
            pass

    # Clear existing timestamps and date range info
    st.session_state.oldest_timestamp = None # Force range recalculation
    st.session_state.newest_timestamp = None
    st.session_state.need_to_display_recent = True # Show most recent for new recording
    st.session_state.last_displayed_timestamp = None # Force display update
    
    # Clear the valid dates cache
    if 'valid_dates' in st.session_state:
        st.session_state.valid_dates = []
    
    # Explicitly reset timestamps in jiffyget to ensure fresh data
    #import jiffyget
    reset_timestamps()
    
    # Get the new date range for this recording with proper error handling
    try:
        # Force a complete refresh of the timestamp data for the new recording
        oldest, newest, timestamps = get_timestamp_range(st.session_state.cam_name, new_session, st.session_state.data_dir)

        # Store the results in session state
        st.session_state.oldest_timestamp = oldest
        st.session_state.newest_timestamp = newest
        
        # Build a list of all unique dates that have images
        unique_dates = set()
        #timestamps = get_timestamps(st.session_state.cam_name, new_session, st.session_state.data_dir, st.session_state.browsing_date)
        if timestamps:
            for ts, _ in timestamps:
                dt = datetime.fromtimestamp(ts / 1000)
                unique_dates.add(dt.date())
            st.session_state.valid_dates = sorted(list(unique_dates))
        else:
            st.session_state.valid_dates = []
        
        if not st.session_state.valid_dates:
            # No valid dates found - use today's date as fallback
            target_date = datetime.now().date()
            st.session_state.hour = 23
            st.session_state.minute = 59
            st.session_state.second = 59
        elif len(st.session_state.valid_dates) == 1:
            # Only one valid date - use it
            target_date = st.session_state.valid_dates[0]
            
            # Also set time to show the latest image on this date
            if newest:
                # Set time to the newest image's time
                st.session_state.hour = newest.hour
                st.session_state.minute = newest.minute
                st.session_state.second = newest.second
            else:
                # Set to end of day if no newest timestamp
                st.session_state.hour = 23
                st.session_state.minute = 59
                st.session_state.second = 59
        else:
            # Multiple valid dates - use newest unless specified otherwise
            if newest:
                target_date = newest.date()
                # Set time to the newest timestamp's time
                st.session_state.hour = newest.hour
                st.session_state.minute = newest.minute
                st.session_state.second = newest.second
            else:
                # Use the newest valid date
                target_date = st.session_state.valid_dates[-1]
                # Set to end of day
                st.session_state.hour = 23
                st.session_state.minute = 59
                st.session_state.second = 59
            
            # Make sure it's in the list of valid dates
            if target_date not in st.session_state.valid_dates:
                # Use the newest valid date
                target_date = st.session_state.valid_dates[-1]
                
        # IMPORTANT: Don't update st.session_state.date directly
        # Instead, store target date in a separate variable
        st.session_state.target_browsing_date = target_date
        
        # Flag that we need to update the date on next rerun
        # This will be handled by the build_main_area function
        st.session_state.needs_date_update = True
        
        # Update browsing_date directly (this isn't a widget)
        st.session_state.browsing_date = target_date
    except Exception as e:
        print(f"Error setting date range for new recording: {str(e)}")
        # Fall back to today's date if there's an error
        today = datetime.now().date()
        st.session_state.target_browsing_date = today
        st.session_state.browsing_date = today
        st.session_state.needs_date_update = True
        # Set time to end of day to find the latest image
        st.session_state.hour = 23
        st.session_state.minute = 59
        st.session_state.second = 59
    
    set_autoplay(None)
    st.session_state.in_playback_mode = True # Default to playback when changing recording

    # Reset UI state to avoid stale frames
    st.session_state.last_frame = None
    st.session_state.actual_timestamp = None
    
    #if st.session_state.get('video_placeholder'):
    #    display_most_recent_image()
    #st.session_state.date = st.session_state.browsing_date
    st.session_state.needs_date_update = True
    st.session_state.init_date = st.session_state.browsing_date


def toggle_rt_capture():
    """Toggle real-time capture on/off"""
    st.session_state.rt_capture = not st.session_state.rt_capture
    #print(f"toggle_rt_capture: {st.session_state.rt_capture}")

    if st.session_state.rt_capture:
        st.session_state.in_playback_mode = False
        #st.session_state.browsing_saved_images = False
        st.session_state.date = datetime.now().date()

        # Get session info
        current_session = st.session_state.session
        data_dir = st.session_state.get('data_dir', 'JiffyData')
        
        # Get active sessions
        active_sessions = get_active_sessions(data_dir)
        
        # Check if current session is active
        if current_session not in active_sessions:
            st.warning(f"Session '{current_session}' is not currently active on any server. Please select an active session or start a server for this session.")
            st.session_state.rt_capture = False
            return
        
        # Get the port for this session
        port = get_session_port(current_session, data_dir)
        if not port:
            st.error(f"Could not determine the server port for session '{current_session}'.")
            st.session_state.rt_capture = False
            return
        
        # Set the HTTP server URL
        server_url = f"http://localhost:{port}"
        st.session_state.http_server_url = server_url
        
        # Get HTTP client
        try:
            if not hasattr(st.session_state, 'http_client'):
                from jiffyclient import JiffyCamClient
                st.session_state.http_client = JiffyCamClient(server_url)
            else:
                # Update URL if client exists
                st.session_state.http_client.set_server_url(server_url)
            
            # Force a connection check
            if not st.session_state.http_client.check_connection(force=True):
                st.error(f"Failed to connect to JiffyCam server at {server_url}: {st.session_state.http_client.last_error}")
                st.session_state.rt_capture = False
                return
            
            # Start capture
            st.session_state.http_client.start_capture()
        except Exception as e:
            st.error(f"Error connecting to JiffyCam server at {server_url}: {str(e)}")
            st.session_state.rt_capture = False
            return
    else:
        # Stop capture
        if hasattr(st.session_state, 'http_client'):
            try:
                st.session_state.http_client.stop_capture()
                # No need for full disconnect, just stop the capture
            except Exception as e:
                print(f"Error stopping capture: {str(e)}")
        st.session_state.in_playback_mode = True

def on_date_change():
    """Handle date picker change."""
    set_autoplay(None)
    
    # Ensure we're not in a deadlock by forcibly resetting key state
    st.session_state.step_direction = "None"
    st.session_state.in_playback_mode = True
    
    # Completely reset timestamps cache to force fresh data load
    reset_timestamps()

    # Update browsing date to match selected date
    st.session_state.browsing_date = st.session_state.date
    
    # Force display refresh
    st.session_state.last_displayed_timestamp = None
    
    # Count frames for the new date - with robust error handling
    try:
        browse_date = st.session_state.date
        #print(f"browse_date: {browse_date}")
        browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
        
        # Get timestamps for this date
        timestamps = get_timestamps(
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
    # Add additional error handling to prevent deadlocks
    try:
        update_image_display(direction="down")
    except Exception as e:
        print(f"Error updating image display: {str(e)}")
        # Reset state to recover from error
        st.session_state.last_frame = None
        st.session_state.actual_timestamp = None
    
    # Reset flag to avoid duplicate loading
    st.session_state.need_to_display_recent = False

def on_navigation_button(direction, onclick=True):
    """Handle image navigation button click.
    
    Args:
        direction: "up" for next image or "down" for previous image
        stopAuto: Whether to stop autoplay mode when navigating
    """
    #print(f"on_navigation_button: {direction}, {onclick}, {inspect.stack()[1].function}")
    st.session_state.in_playback_mode = True

    # Increment/decrement time logic
    tweek_time(direction)
    #print(f"on_navigation_button: {direction}, {stopAuto}")
    if onclick:
        set_autoplay(None)
        st.session_state.autoplay_step = direction
        return
    
    #tweek_time(direction)

    st.session_state.in_playback_mode = True

    # Fetch placeholders from session_state inside the update function
    if onclick:
        st.session_state.step_direction = direction  # let rerun handle the update  
    else:
        update_image_display(direction=direction)

def on_prev_button(onclick=True):
    """Handle previous image button click."""
    on_navigation_button('down', onclick)

def on_next_button(onclick=True):
    """Handle next image button click."""
    on_navigation_button('up', onclick)

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
    # Completely reset timestamps cache to force fresh data
    reset_timestamps()

    # Get current date from the session state
    current_date = st.session_state.date
    #print(f"current_date: {current_date}")

    # Check if we have valid dates in session state
    if 'valid_dates' in st.session_state and st.session_state.valid_dates:
        valid_dates = sorted(st.session_state.valid_dates)
        #print(f"valid_dates: {valid_dates}")    
        # If we only have one valid date, we can't navigate
        if len(valid_dates) <= 1:
            print(f"Only one valid date available ({valid_dates[0] if valid_dates else 'none'}), can't change days")
            # Just refresh the current date's display
            st.session_state.step_direction = "down"
            st.session_state.last_displayed_timestamp = None
            return
            
        # Find current date in the valid dates list
        current_index = -1
        for i, date in enumerate(valid_dates):
            if date == current_date:
                current_index = i
                break
                
        # Calculate the new date based on direction
        if direction == "next" and current_index < len(valid_dates) - 1:
            new_date = valid_dates[current_index + 1]
        elif direction == "prev" and current_index > 0:
            new_date = valid_dates[current_index - 1]
        else:
            # We're at the end of the list, can't navigate further
            #print(f"At {'end' if direction == 'next' else 'beginning'} of valid dates, can't change days")
            # Just refresh the current date's display
            st.session_state.step_direction = "down"
            st.session_state.last_displayed_timestamp = None
            return
        
        #print(f"change_day: {current_index}, {new_date}, direction: {direction}")

    else:
        # Get valid date range from timestamps
        min_date = st.session_state.oldest_timestamp.date() if st.session_state.oldest_timestamp else None
        max_date = max(datetime.now().date(), st.session_state.newest_timestamp.date() if st.session_state.newest_timestamp else datetime.now().date())
        
        # If we have a single day range, don't try to change days
        if min_date and max_date and min_date == max_date:
            # Just refresh the current date's display
            st.session_state.step_direction = "down"
            st.session_state.last_displayed_timestamp = None
            return
        
        # Calculate the new date based on direction
        if direction == "next":
            new_date = current_date + timedelta(days=1)
        else:  # "prev"
            new_date = current_date - timedelta(days=1)
        
        # Check if the new date is within min/max range
        if min_date and new_date < min_date:
            new_date = min_date
        if max_date and new_date > max_date:
            new_date = max_date
    
    #print(f"new_date: {new_date}, min_date: {min_date}, max_date: {max_date}")
    # Reset state to ensure clean navigation
    st.session_state.step_direction = "down"
    st.session_state.last_displayed_timestamp = None
    st.session_state.actual_timestamp = None
    
    # Force resetting the frame display to avoid stale images
    st.session_state.last_frame = None
    
    # IMPORTANT: Don't modify st.session_state.date directly
    # Instead, set up for update on next rerun 
    st.session_state.target_browsing_date = new_date
    st.session_state.needs_date_update = True
    
    # Update browsing_date directly (this isn't a widget)
    st.session_state.browsing_date = new_date

def toggle_live_pause():
    """Handle Live/Pause button click."""
    #print(f"toggle_live_pause: {st.session_state.in_playback_mode}")
    st.session_state.autoplay_direction = None  # reset autoplay direction

    if st.session_state.in_playback_mode:
        # Go Live
        # Check if current browsing date is today
        current_date = datetime.now().date()
        browsing_date = st.session_state.get('browsing_date') or current_date
        
       #if browsing_date != current_date:
            # Don't toggle to live mode if date isn't today
            #return
            
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
        
        # No longer immediately reset FPS Display - let it decay naturally
        # when no more frames are being displayed

def on_pause_button():
    """Handle pause button click."""
    set_autoplay(None)
    st.session_state.in_playback_mode = True
    st.session_state.step_direction = "None"
    
    # Force display update when pausing
    st.session_state.last_displayed_timestamp = None
    st.session_state.autoplay_step = "current"

def on_fast_reverse_button():
    """Handle fast reverse button click."""
    if st.session_state.autoplay_direction == None:
        set_autoplay("down")
        tweek_time('down')
    else:
        st.session_state.autoplay_direction = None
        #st.session_state.needs_date_update = True

    # Set to playback mode like other buttons do
    st.session_state.in_playback_mode = True

def on_fast_forward_button():
    """Handle fast forward button click."""
    if st.session_state.autoplay_direction == None:
        set_autoplay("up")
        tweek_time('up')
    else:
        st.session_state.autoplay_direction = None
    
    # Set to playback mode like other buttons do
    st.session_state.in_playback_mode = True

def on_day_navigation_button(direction):
    """Handle day navigation button click.
    
    Args:
        direction: "next" or "prev" indicating which day to navigate to
    """
    set_autoplay(None)
    
    # In-line function to force playback controls to work right after day change
    st.session_state.in_playback_mode = True
    st.session_state.last_displayed_timestamp = None  # Force display update
    
    try:
        # Explicitly update timestamp range before changing days
        #import jiffyget
        reset_timestamps()
        oldest, newest, timestamps = get_timestamp_range(st.session_state.cam_name, st.session_state.session, st.session_state.data_dir)
        st.session_state.oldest_timestamp = oldest
        st.session_state.newest_timestamp = newest
        
        # Change the day
        change_day(direction)
        st.session_state.needs_date_update = True

        # Directly trigger the date change handler instead of waiting for callback
        #on_date_change()
    except Exception as e:
        print(f"Error changing to {direction} day: {str(e)}")
        # Reset key state on error to prevent deadlocks
        #import jiffyget
        reset_timestamps()
        st.session_state.step_direction = "None"
        st.session_state.last_frame = None

def on_prev_day_button():
    """Handle previous day button click."""
    on_day_navigation_button("prev")
    
def on_next_day_button():
    """Handle next day button click."""
    on_day_navigation_button("next")

# --- UI Update Functions ---
def new_image_display(frame):
    if frame is None:
        return

    """Display a new image based on the current date and time."""
    # Track FPS Display whenever we display a new frame, even in playback mode
    current_time = time.time()
    st.session_state.display_frame_count += 1
    
    # Calculate FPS Display once per second
    if current_time - st.session_state.last_display_time >= 1.0:
        st.session_state.display_fps = st.session_state.display_frame_count / (current_time - st.session_state.last_display_time)
        st.session_state.display_frame_count = 0
        st.session_state.last_display_time = current_time
    
    #print(f"new_image_display: {inspect.stack()[1].function}")
    if(not st.session_state.video_placeholder):     # delay creation to avoid flickering
        #print(f"new_image_display: video_placeholder is None, {inspect.stack()[1].function}")
        st.session_state.video_placeholder = st.image(frame, channels="BGR", use_container_width=True)
    else:
        #print(f"new_image_display: video_placeholder is not None, {inspect.stack()[1].function}")
        st.session_state.video_placeholder.image(frame, channels="BGR", use_container_width=True)
    
    #video_placeholder = st.session_state.video_placeholder
    #video_placeholder.image(frame, channels="BGR", use_container_width=True)

    timearrow_placeholder = st.session_state.timearrow_placeholder
    ta_img = generate_timeline_arrow()
    timearrow_placeholder.image(ta_img, channels="RGB", use_container_width=True)
    
def update_image_display(direction=None):
    """Update the image display based on the current date and time."""
    #print(f"update_image_display: {direction}, {inspect.stack()[1].function}")

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
    #print(f"update_image_display, {st.session_state.last_frame is None}, {direction}")
    if( False and direction == "current" and st.session_state.last_frame is not None):
        closest_image = st.session_state.last_frame
        #timestamp = st.session_state.last_timestamp
        timestamp = st.session_state.actual_timestamp
        #print(f"update_image_display, using last frame, {timestamp}")
    else:
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
            #print(f"update_image_display: eof: {eof}")
            set_autoplay(None)

    success = False
    if closest_image is not None:
        try:
            # Store frame and actual timestamp
            st.session_state.last_frame = closest_image.copy()
            dts = datetime.fromtimestamp(timestamp/1000)
            st.session_state.actual_timestamp = dts
            #st.session_state.last_timestamp = timestamp

            # Update time state to match the found image
            st.session_state.hour = dts.hour
            st.session_state.minute = dts.minute
            st.session_state.second = dts.second
            st.session_state.browsing_date = dts.date() # Keep browsing date in sync with found image

            # Update UI placeholders
            #print(f"update_image_display: {inspect.stack()[1].function}")
            new_image_display(closest_image)

            # Update time display text
            time_display.markdown(
                f'<div class="time-display">{format_time_12h(dts.hour, dts.minute, dts.second)}</div>',
                unsafe_allow_html=True
            )

            # Update status text
            # status_placeholder.markdown(f"<div style='padding: 5px 0;'>Viewing: {dts.strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
            st.session_state.status_message = f"Viewing: {dts.strftime('%Y-%m-%d %H:%M:%S')}" # Update internal status too
            success = True

        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            status_placeholder.error(f"Error displaying image: {str(e)}")
            st.session_state.status_message = f"Error displaying image: {str(e)}"
    
    if not success:
        set_autoplay(None)

        # No image found or error displaying
        # status_placeholder.markdown("<div style='padding: 5px 0;'>No image found near specified time</div>", unsafe_allow_html=True)
        st.session_state.status_message = f"No image found near specified time"
        # Clear the last frame and the display placeholder
        st.session_state.last_frame = None
        #video_placeholder.empty()

    return success

def display_most_recent_image():
    """Display the most recent image for the current camera."""

    # If we have a newest timestamp available, use its time directly
    if st.session_state.newest_timestamp:
        timestamp = st.session_state.newest_timestamp
        st.session_state.hour = timestamp.hour
        st.session_state.minute = timestamp.minute
        st.session_state.second = timestamp.second
        st.session_state.browsing_date = timestamp.date()
    else:
        # Otherwise use current time as starting point
        timestamp = datetime.now()
        st.session_state.hour = timestamp.hour
        st.session_state.minute = timestamp.minute
        st.session_state.second = timestamp.second
        st.session_state.browsing_date = timestamp.date()
        
        # If we don't have a newest timestamp, try to get it again
        try:
            #import jiffyget
            reset_timestamps()
            oldest, newest, timestamps = get_timestamp_range(st.session_state.cam_name, st.session_state.session, st.session_state.data_dir)
            if newest:
                st.session_state.newest_timestamp = newest
                st.session_state.oldest_timestamp = oldest
                # Update with newest timestamp info
                st.session_state.hour = newest.hour
                st.session_state.minute = newest.minute
                st.session_state.second = newest.second
                st.session_state.browsing_date = newest.date()
        except Exception as e:
            print(f"Error getting timestamp range in display_most_recent_image: {str(e)}")
    
    # st.session_state.date = timestamp.date() # REMOVED: Cannot modify widget state directly
    #print(f"display_most_recent_image: {st.session_state.browsing_date}")

    # Make sure we're in playback mode
    st.session_state.in_playback_mode = True

    # Make sure step_direction is set for proper playback control initialization
    if 'step_direction' not in st.session_state or not st.session_state.step_direction:
        st.session_state.step_direction = "down"
    
    # Reset jiffyget timestamps to force reload for current date
    #import jiffyget
    reset_timestamps()
    
    # Count frames for today
    try:
        browse_date = st.session_state.browsing_date
        browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
        
        # Get timestamps for this date
        timestamps = get_timestamps(
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
    st.session_state.needs_date_update = True

# --- UI Building Functions ---

def get_server_status():
    """Get the server status information for all active sessions.
    
    Returns:
        Tuple of (active_sessions, is_capturing, capture_fps, last_save_time, client_connected)
        where active_sessions is a list of active sessions (may be empty)
    """
    from jiffyget import get_active_sessions, get_session_port
    
    # Get data directory from session state
    data_dir = st.session_state.get('data_dir', 'JiffyData')
    current_session = st.session_state.get('session')
    
    # Get all active sessions by scanning config files and checking HTTP endpoints
    try:
        active_sessions = get_active_sessions(data_dir)
    except Exception as e:
        print(f"Error getting active sessions: {str(e)}")
        active_sessions = []
    
    # Default status values (for the current client connection)
    is_capturing = False
    capture_fps = 0
    last_save_time = None
    client_connected = False
    
    # Check if current session is active and has the correct port
    if current_session in active_sessions:
        # Get the correct port for this session
        correct_port = get_session_port(current_session, data_dir)
        current_port = st.session_state.get('http_server_port')
        
        # If the port has changed, update the client
        if correct_port and correct_port != current_port:
            #print(f"Updating HTTP port from {current_port} to {correct_port}")
            # Update session state
            st.session_state.dataserver_port = correct_port
            st.session_state.http_server_port = correct_port
            st.session_state.http_server_url = f"http://localhost:{correct_port}"
            
            # Recreate the client with the new URL if it exists
            if hasattr(st.session_state, 'http_client'):
                try:
                    from jiffyclient import JiffyCamClient
                    st.session_state.http_client = JiffyCamClient(st.session_state.http_server_url)
                    print(f"Connected dataserver client at {st.session_state.http_server_url}")
                except Exception as e:
                    print(f"Error connecting dataserver client: {str(e)}")
    
    # If we have an HTTP client connected, get its status
    if hasattr(st.session_state, 'http_client'):
        try:
            client = st.session_state.http_client
            # Force a connection check to get updated status
            if client.check_connection(force=True):
                client_connected = True
                # Get status from current client
                if client.last_status:
                    is_capturing = client.last_status.get('capturing', False)
                    capture_fps = client.last_status.get('fps', 0)
                    last_save_time = client.last_status.get('last_save_time', None)
        except Exception as e:
            print(f"Error checking HTTP client status: {str(e)}")
            # Reset connection status on error
            client_connected = False
    
    return active_sessions, is_capturing, capture_fps, last_save_time, client_connected

def build_sidebar():
    """Build the sidebar UI elements."""
    # Setup main sections
    with st.sidebar:
        # Apply CSS styling for metrics
        from jiffyui_components import apply_metrics_css, create_metric_display, create_fps_metrics_row, create_detection_metrics
        
        apply_metrics_css()
        
        st.title("JiffyCam")
        
        # Recording Selection removed from here - moved to main area

        # Status section header
        st.header("Status")
        
        # Create empty placeholders to maintain return compatibility
        status_placeholder = st.empty()
        server_status_placeholder = st.empty()
        
        # Create metrics displays
        capture_fps_placeholder, display_fps_placeholder = create_fps_metrics_row()
        frames_detected_placeholder, last_save_time_placeholder = create_detection_metrics()

        error_placeholder = st.empty()

    return status_placeholder, error_placeholder, server_status_placeholder, capture_fps_placeholder, display_fps_placeholder, frames_detected_placeholder, last_save_time_placeholder

def on_timeline_click(coords):
    """Handle clicks on the timeline image."""
    set_autoplay(None)
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
    st.session_state.last_displayed_timestamp = None  # Force display update for new position
    st.session_state.step_direction = "None"  # Reset step direction
    st.session_state.autoplay_step = "None"

    set_autoplay(None)

def build_main_area():
    """Create the main UI area elements and return placeholders."""
    # Import UI components
    from jiffyui_components import (
        apply_general_css, 
        create_date_navigation, 
        create_playback_controls, 
        create_live_button,
        create_empty_timeline_arrow,
        create_recording_selector,
        create_navigation_button
    )
    
    # Apply CSS for main area styling
    apply_general_css()

    # --- Layout ---

    # Centered controls container
    st.markdown("<div class='top-controls'>", unsafe_allow_html=True)

    # Check if we have a single-day recording
    single_day_recording = False
    if 'valid_dates' in st.session_state and len(st.session_state.valid_dates) == 1:
        single_day_recording = True
    
    # Create a row for recordings and date navigation
    # Use a wider first column for the recording selector to match time display column width
    date_nav_cols = st.columns([3, 0.7, 4, 0.7])
    
    # Add recording selector in the first column
    with date_nav_cols[0]:
        # Get data_dir from session state
        data_dir = st.session_state.get('data_dir', 'JiffyData') # Default if not set

        # Add vertical space to align with date picker
        st.write("")  # Space for alignment

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

        # Get the current active sessions for highlighting
        active_sessions, _, _, _, _ = get_server_status()
        
        # Create the recording selector
        create_recording_selector(
            options=recording_keys,
            current_selection=st.session_state.get('selected_recording_key'),
            on_change_handler=on_recording_change,
            help_text="Select the recording session to view.",
            active_sessions=active_sessions
        )
    
    # Previous day button
    with date_nav_cols[1]:
        st.write("")  # Space for alignment
        create_navigation_button(
            "◀", 
            "prev_day_button",
            "Previous Day" if not single_day_recording else "No previous days available",
            on_prev_day_button,
            disabled=single_day_recording
        )
    
    # Date picker placeholder
    with date_nav_cols[2]:
        date_picker_placeholder = st.empty()
    
    # Next day button
    with date_nav_cols[3]:
        st.write("")  # Space for alignment
        create_navigation_button(
            "▶", 
            "next_day_button",
            "Next Day" if not single_day_recording else "No more days available",
            on_next_day_button,
            disabled=single_day_recording
        )
    
    # Handle date picker creation - moved from the inline code in the column
    with date_nav_cols[2]:
        # Ensure we have the latest timestamp range data before setting up date picker
        if (st.session_state.oldest_timestamp is None or 
            st.session_state.newest_timestamp is None or 
            st.session_state.get('needs_date_update', False)):
            try:
                # Force a fresh recalculation of the timestamp range
                #import jiffyget
                reset_timestamps()
                oldest, newest, timestamps = get_timestamp_range(st.session_state.cam_name, st.session_state.session, st.session_state.data_dir)
                st.session_state.oldest_timestamp = oldest
                st.session_state.newest_timestamp = newest
                
                # Build a list of all unique dates that have images
                unique_dates = set()
                #timestamps = get_timestamps(st.session_state.cam_name, st.session_state.session, st.session_state.data_dir, st.session_state.browsing_date)
                if timestamps:
                    for ts, _ in timestamps:
                        dt = datetime.fromtimestamp(ts / 1000)
                        unique_dates.add(dt.date())
                    st.session_state.valid_dates = sorted(list(unique_dates))
                else:
                    st.session_state.valid_dates = []
                
            except Exception as e:
                print(f"Error recalculating timestamp range: {str(e)}, {inspect.stack()[1][3]}")
                st.session_state.valid_dates = []
        
        # Initialize min/max dates for the date picker
        min_date = None
        max_date = datetime.now().date()
        
        # Get data range from timestamps
        if st.session_state.oldest_timestamp: 
            min_date = st.session_state.oldest_timestamp.date()
        if st.session_state.newest_timestamp: 
            max_date = max(max_date, st.session_state.newest_timestamp.date())
        
        # Ensure min_date is not after max_date and set defaults if either is None
        if min_date is None:
            min_date = datetime(2000, 1, 1).date()  # Safe default if no minimum date
        if max_date is None:
            max_date = datetime.now().date()  # Today as default if no maximum date
        if min_date > max_date:
            min_date = max_date
        
        # Calculate a valid default date within bounds
        # Check if we have a pending date update from recording change
        if st.session_state.get('needs_date_update', False) and 'target_browsing_date' in st.session_state:
            # If we need to update the date, use the target date calculated in on_recording_change
            default_date = st.session_state.target_browsing_date
            
            # Clear the flags now that we've used them
            #st.session_state.needs_date_update = False
            
            # Make sure the date is within bounds and is a valid date with images
            if 'valid_dates' in st.session_state and st.session_state.valid_dates:
                # Find closest valid date
                if default_date not in st.session_state.valid_dates:
                    # Find the closest valid date
                    valid_dates = sorted(st.session_state.valid_dates)
                    # Use the newest date by default if target date is after all valid dates
                    if default_date > valid_dates[-1]:
                        default_date = valid_dates[-1]
                    # Use the oldest date if target date is before all valid dates
                    elif default_date < valid_dates[0]:
                        default_date = valid_dates[0]
                    else:
                        # Find the closest date
                        closest_date = min(valid_dates, key=lambda d: abs((default_date - d).days))
                        default_date = closest_date
                
                # Also update the browsing_date since it's not a widget state
                st.session_state.browsing_date = default_date
            else:
                # Fall back to bounds checking if no valid dates list
                if default_date < min_date:
                    default_date = min_date
                    # Also update the browsing_date
                    st.session_state.browsing_date = default_date
                elif default_date > max_date:
                    default_date = max_date
                    # Also update the browsing_date
                    st.session_state.browsing_date = default_date
                    
        elif 'date' in st.session_state:
            # Use the existing date value but validate it's within bounds and is a valid date
            default_date = st.session_state.date
            
            # Check if it's in valid dates first
            if 'valid_dates' in st.session_state and st.session_state.valid_dates:
                if default_date not in st.session_state.valid_dates:
                    # Find closest valid date
                    valid_dates = sorted(st.session_state.valid_dates)
                    if valid_dates:
                        closest_date = min(valid_dates, key=lambda d: abs((default_date - d).days))
                        default_date = closest_date
                        st.session_state.browsing_date = default_date
                    else:
                        # Fall back to today if no valid dates
                        default_date = max_date
                        st.session_state.browsing_date = default_date
            else:
                # Fall back to bounds checking if no valid dates list
                if default_date < min_date:
                    default_date = min_date
                    # Also update the browsing_date
                    st.session_state.browsing_date = min_date
                elif default_date > max_date:
                    default_date = max_date
                    # Also update the browsing_date
                    st.session_state.browsing_date = max_date
        elif 'init_date' in st.session_state:
            # Use the init_date value but validate it's within bounds
            default_date = st.session_state.init_date
            
            # Check if it's in valid dates first
            if 'valid_dates' in st.session_state and st.session_state.valid_dates:
                if default_date not in st.session_state.valid_dates:
                    # Find closest valid date
                    valid_dates = sorted(st.session_state.valid_dates)
                    if valid_dates:
                        closest_date = min(valid_dates, key=lambda d: abs((default_date - d).days))
                        default_date = closest_date
                        st.session_state.browsing_date = default_date
                    else:
                        # Fall back to today if no valid dates
                        default_date = max_date
                        st.session_state.browsing_date = default_date
            else:
                # Fall back to bounds checking if no valid dates list
                if default_date < min_date:
                    default_date = min_date
                    # Also update the browsing_date
                    st.session_state.browsing_date = min_date
                elif default_date > max_date:
                    default_date = max_date
                    # Also update the browsing_date
                    st.session_state.browsing_date = max_date
        else:
            # Initial default if neither session_state.date nor init_date exists yet
            if 'valid_dates' in st.session_state and st.session_state.valid_dates:
                # Use newest valid date
                valid_dates = sorted(st.session_state.valid_dates)
                default_date = valid_dates[-1]
                st.session_state.browsing_date = default_date
            else:
                # Fall back to max_date
                default_date = max_date
                if default_date < min_date:
                    default_date = min_date
                    # Also update the browsing_date
                    st.session_state.browsing_date = min_date
        
        # Create the date_input widget - ONLY use the value parameter, never set st.session_state.date directly
        from jiffyui_components import create_date_picker
        
        try:
            # Ensure default_date is valid and within min_date and max_date
            if default_date < min_date:
                default_date = min_date
            if default_date > max_date:
                default_date = max_date
                
            # If we have valid dates, only allow selecting those dates
            if 'valid_dates' in st.session_state and st.session_state.valid_dates:
                # For recordings with single day of data, just show that date
                if len(st.session_state.valid_dates) == 1:
                    only_date = st.session_state.valid_dates[0]
                    
                    create_date_picker(
                        value=only_date,
                        min_value=only_date,
                        max_value=only_date,
                        on_change_handler=on_date_change,
                        key="date",
                        help_text="Only one date available for this recording",
                        single_day=True
                    )
                    
                    # Ensure browsing_date is set correctly
                    st.session_state.browsing_date = only_date
                else:
                    # Multiple valid dates - create a date picker with only those dates
                    valid_dates = sorted(st.session_state.valid_dates)
                    
                    create_date_picker(
                        value=default_date,
                        min_value=valid_dates[0],
                        max_value=valid_dates[-1],
                        on_change_handler=on_date_change,
                        key="date",
                        help_text=f"Choose from {len(valid_dates)} days with data"
                    )
            else:
                # Standard date picker with min/max bounds
                create_date_picker(
                    value=default_date,
                    min_value=min_date,
                    max_value=max_date,
                    on_change_handler=on_date_change,
                    key="date",
                    help_text="Select date"
                )
        except Exception as e:
            print(f"Date picker error: {str(e)}")
            
            # Try a simpler fallback with a unique key
            try:
                create_date_picker(
                    value=default_date,
                    min_value=None,
                    max_value=None,
                    on_change_handler=on_date_change,
                    key="date_fallback",
                    help_text="Select date"
                )
            except Exception as e2:
                print(f"Fallback date picker error: {str(e2)}")
                
                # Final fallback - just show the date as text
                today = datetime.now().date()
                date_picker_placeholder.markdown(f"**Date:** {default_date.strftime('%Y-%m-%d')}")

    # Create playback controls
    handlers = {
        'fast_reverse': on_fast_reverse_button,
        'prev': on_prev_button,
        'pause': on_pause_button,
        'next': on_next_button,
        'fast_forward': on_fast_forward_button,
        # 'live' is handled separately
    }
    
    time_cols, time_display = create_playback_controls(handlers)

    # Handle Live button separately due to its conditional styling
    with time_cols[7]:
        # Check if client is connected and status is available
        client_connected = False
        
        try:
            active_sessions, is_capturing, capture_fps, last_save_time, client_connected = get_server_status()
            
            # Get UI session
            ui_session = st.session_state.get('session')
    
            # Determine if live button should be disabled
            live_disabled = True # Default to disabled
            button_help = ""
            
            if ui_session in active_sessions:
                # Check if we can get a valid dataserver port for this session
                port = get_session_port(ui_session, data_dir)
                if port:
                    # Only enable if session is active, has a port, AND the date is today
                    current_date = datetime.now().date()
                    browsing_date = st.session_state.get('browsing_date') or current_date
                    if True or browsing_date == current_date:
                        live_disabled = False
                    else:
                        live_disabled = True
                        button_help = "Live view is only available for current date"
                else:
                    live_disabled = True
                    button_help = f"No dataserver port configured for session '{ui_session}'"
            else:
                live_disabled = True
                button_help = f"Session '{ui_session}' is not active on any server"
        except Exception as e:
            print(f"Error checking server status: {str(e)}")
            live_disabled = True
            button_help = "Error connecting to server"
            active_sessions = []
            ui_session = st.session_state.get('session')
    
        # Create the live button with appropriate styling
        create_live_button(
            handler=toggle_live_pause,
            is_enabled=not live_disabled,
            is_live_mode=not st.session_state.in_playback_mode,
            help_text=button_help,
            session_name=ui_session,
            active_sessions=active_sessions
        )

    # Time Arrow above timeline
    timearrow_placeholder, timearrow_img = create_empty_timeline_arrow(width=1200, height=24)

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
    #video_placeholder = st.empty()   # delay creation to avoid flickering
    video_placeholder = None

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
    last_save_time_placeholder = st.session_state.get('last_save_time_placeholder', None)
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
    
    #if(not st.session_state.autoplay_direction):
    if st.session_state.needs_date_update:
        update_image_display("down")
        #change_day("down")
        st.session_state.needs_date_update = False
    elif (st.session_state.need_to_display_recent and not is_capturing):
        display_most_recent_image() # Fetches placeholders from session_state
        st.session_state.needs_to_display_recent = False
    elif st.session_state.autoplay_step == None and st.session_state.last_frame is not None:
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
                active_sessions, is_capturing, capture_fps, last_save_time, client_connected = get_server_status()
                if is_capturing:
                    connection_status = "Recording"
                
                # Store FPS Camera in session state
                st.session_state.capture_fps = capture_fps
                
                # Count frames for current date
                if False and current_time - st.session_state.last_frames_count_update > 5:  # Update every 5 seconds
                    st.session_state.last_frames_count_update = current_time
                    try:
                        # Get current date in posix milliseconds
                        if st.session_state.in_playback_mode:
                            browse_date = st.session_state.browsing_date
                        else:
                            browse_date = datetime.now().date()
                        
                        #print(f"browse_date: {browse_date}")
                        browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
                        
                        # Get timestamps for this date
                        #import jiffyget
                        timestamps = get_timestamps(
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
                # status_html = f"""
                # <div style="padding: 5px 0;">
                #     <div><b>Recording:</b> <span style="color: {'blue' if server_session != None else 'gray'}">{server_session if server_session != None else 'Idle'}</span></div>
                # </div>
                # """
                # server_status_placeholder.markdown(status_html, unsafe_allow_html=True)
                
                # Update FPS metrics
                # Only show FPS Camera when connected to and viewing live data
                is_viewing_live = False
                if (not st.session_state.in_playback_mode and 
                    hasattr(st.session_state, 'http_client')):
                    client = st.session_state.http_client
                    current_session = st.session_state.session
                    is_viewing_live = (
                        client.is_connected() and
                        client.last_status and
                        current_session in active_sessions
                    )
                
                if is_viewing_live:
                    # Format to 1 decimal place and show the metric
                    capture_fps_placeholder.metric(
                        "FPS Camera",
                        f"{st.session_state.capture_fps:.0f}",
                        delta=None,
                        #help="Frames per second being captured by the camera",
                        border=True
                    )
                else:
                    # Reset and hide FPS Camera when not viewing live
                    st.session_state.capture_fps = 0
                    capture_fps_placeholder.metric(
                        "FPS Camera",
                        "  ---",
                        delta=None,
                        #help="Frames per second being captured by the camera",
                        border=True
                    )
                
                # Set FPS Display to 0 when in playback mode and no frames displayed recently
                if st.session_state.in_playback_mode and (current_time - st.session_state.last_display_time > 2.0):
                    st.session_state.display_fps = 0
                    
                display_fps_placeholder.metric(
                    "FPS Display",
                    f"{st.session_state.display_fps:.0f}",
                    delta=None,
                    #help="Frames per second being displayed",
                    border=True
                )
                
                frames_detected_placeholder.metric(
                    "Detections",
                    st.session_state.frames_detected,
                    delta=None,
                    #help="Number of frames detected for current date",
                    border=True
                )

                # Display Last Detectiond time (as string)
                if last_save_time:
                    last_save_display = str(last_save_time)
                else:
                    last_save_display = "---"
                    
                # Last Detection UI component has been removed, but keep the data flow
                # The last_save_time_placeholder will be None, so we should skip updating it
                if last_save_time_placeholder is not None:
                    last_save_time_placeholder.metric(
                        "Last Detection",
                        last_save_display,
                        delta=None,
                        help="Time of last image detection and save",
                        border=True
                    )
            # Check for errors first
            check_errors = False
            
            if hasattr(st.session_state, 'http_client'):
                # Only check connection if we're not in playback mode
                if (st.session_state.rt_capture and 
                    not st.session_state.in_playback_mode and
                    not st.session_state.http_client.is_connected()):
                    
                    error_placeholder.error(st.session_state.http_client.last_error or "Connection lost to JiffyCam server")
                    # status_placeholder.markdown("<div style='padding: 5px 0;'>Status: Connection Error</div>", unsafe_allow_html=True)
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
                        try:
                            frame = st.session_state.http_client.get_last_frame()
                            timestamp = datetime.now().timestamp() * 1000   # meh, this is not the actual timestamp
                            #print(f"timestamp: {timestamp}")
                            if frame is not None:
                                st.session_state.last_frame = frame
                                #st.session_state.last_timestamp = timestamp
                                st.session_state.actual_timestamp = timestamp
                        except Exception as e:
                            print(f"Error getting last frame: {str(e)}")
                            error_placeholder.error(f"Connection error: {str(e)}")
                            st.session_state.rt_capture = False
                            is_capturing = False

                if st.session_state.in_playback_mode:
                    # Playback Mode - Don't make any HTTP requests
                    # Ensure paused frame is displayed
                    if st.session_state.last_frame is not None:
                        ts = st.session_state.actual_timestamp
                        new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback Mode"
                    else:
                        new_status = "Playback Mode (No Frame)"
                        if(video_placeholder):
                            video_placeholder.empty() # Clear if no frame to show

                else: # Live View Mode (Capture Running)
                    if hasattr(st.session_state, 'http_client'):
                        try:
                            # In live view mode, always get the latest frame
                            frame, fps, _, _ = st.session_state.http_client.get_frame()
                            
                            # Always update status to show we're using HTTP mode
                            if frame is not None:
                                status = st.session_state.http_client.last_status
                                if status:
                                    fps = status.get('fps', 0)
                                    frame_count = status.get('frame_count', 0)
                                    new_status = f"Live View - Frames: {frame_count}"
                                else:
                                    new_status = "Live View"

                                # Update time display and slider for live view
                                current_time = datetime.now()
                                time_display.markdown(f'<div class="time-display">{format_time_12h(current_time.hour, current_time.minute, current_time.second)}</div>', unsafe_allow_html=True)
                                st.session_state.hour = current_time.hour
                                st.session_state.minute = current_time.minute
                                st.session_state.second = current_time.second
                                                            
                                st.session_state.last_frame = frame
                                st.session_state.actual_timestamp = current_time.timestamp() * 1000
                                # Display the frame
                                new_image_display(frame)
                            else:
                                # No frame received
                                new_status = "Live View - Waiting for frames"
                        except Exception as e:
                            # Handle errors gracefully
                            print(f"Error getting frame from HTTP client: {str(e)}")
                            error_placeholder.error(f"Error connecting to server: {str(e)}")
                            st.session_state.rt_capture = False
                            st.session_state.in_playback_mode = True
                            is_capturing = False
                    else:
                        # No HTTP client available
                        new_status = "Live View - No HTTP client"
                        st.session_state.in_playback_mode = True
                        is_capturing = False

            else: # Not capturing
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
                            #new_image_display(st.session_state.last_frame)
                            st.session_state.last_displayed_timestamp = st.session_state.actual_timestamp
                            
                            # Reset step_direction to prevent continuous updates
                            if st.session_state.step_direction not in ["up", "down"]:
                                st.session_state.step_direction = "None"

                        ts = st.session_state.actual_timestamp
                        new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback (Stopped)"
                    else:
                        new_status = "Playback (Stopped - No Frame)"
                        if(video_placeholder):
                            video_placeholder.empty()
                else:
                    # Not capturing and not in playback mode - show message to start capture
                    #video_placeholder.info("Connect to JiffyCam server and start capture, or select a date/time to browse.")
                    if st.session_state.last_frame is not None:
                        new_image_display(st.session_state.last_frame)
                    new_status = "Status: Idle"

            # Update status placeholder with formatted status
            # status_placeholder.markdown(f"<div style='padding: 5px 0;'>{new_status}</div>", unsafe_allow_html=True)
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

def initialize_session_state():
    """Initialize session state variables."""
    # Load config for default values
    from jiffyconfig import JiffyConfig
    from collections import OrderedDict
    
    # Check if data_dir is already set in session state (from command line arguments)
    if 'data_dir' in st.session_state:
        config_manager = JiffyConfig(data_dir=st.session_state.data_dir)
    else:
        config_manager = JiffyConfig()
    config = config_manager.config
    
    # UI interaction state flags
    if 'in_playback_mode' not in st.session_state: st.session_state.in_playback_mode = True
    if 'rt_capture' not in st.session_state: st.session_state.rt_capture = False
    if 'need_to_display_recent' not in st.session_state: st.session_state.need_to_display_recent = True
    #if 'live_button_clicked' not in st.session_state: st.session_state.live_button_clicked = False
    if 'status_message' not in st.session_state: st.session_state.status_message = "Initializing..."
    #if 'image_just_saved' not in st.session_state: st.session_state.image_just_saved = False
    if 'step_direction' not in st.session_state: st.session_state.step_direction = None
    if 'autoplay_direction' not in st.session_state: st.session_state.autoplay_direction = None
    
    # Default to HTTP mode
    if 'use_http_mode' not in st.session_state: st.session_state.use_http_mode = True
    
    # Initialize dataserver_port from config
    if 'dataserver_port' not in st.session_state: 
        # Try to get the dataserver_port from config, but use None if not found
        st.session_state.dataserver_port = int(config.get('dataserver_port', 0)) or None
    
    # Set http_server_port to match dataserver_port for consistency
    if 'http_server_port' not in st.session_state:
        st.session_state.http_server_port = st.session_state.dataserver_port
    
    # Build HTTP server URL with the configured port (if available)
    if 'http_server_url' not in st.session_state and st.session_state.dataserver_port:
        st.session_state.http_server_url = f"http://localhost:{st.session_state.dataserver_port}"
    
    # Configuration related state (derived from config)
    if 'cam_name' not in st.session_state: st.session_state.cam_name = config.get('cam_name', 'cam0')
    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = config.get('data_dir', 'JiffyData')
        if not os.path.exists(st.session_state.data_dir):
            os.makedirs(st.session_state.data_dir, exist_ok=True)

    #if 'save_interval' not in st.session_state: st.session_state.save_interval = int(config.get('save_interval', 60))
    
    # Time/Date related state
    current_time = datetime.now()
    if 'hour' not in st.session_state: st.session_state.hour = current_time.hour
    if 'minute' not in st.session_state: st.session_state.minute = current_time.minute
    if 'second' not in st.session_state: st.session_state.second = current_time.second
    
    # Default date to today (will be adjusted after timestamp range is determined)
    if 'init_date' not in st.session_state: st.session_state.init_date = current_time.date()
    
    # Image/Timestamp state
    if 'last_frame' not in st.session_state: st.session_state.last_frame = None
    if 'actual_timestamp' not in st.session_state: st.session_state.actual_timestamp = None
    if 'oldest_timestamp' not in st.session_state: st.session_state.oldest_timestamp = None
    if 'newest_timestamp' not in st.session_state: st.session_state.newest_timestamp = None

    # Add performance tracking variables
    if 'capture_fps' not in st.session_state: st.session_state.capture_fps = 0
    if 'display_fps' not in st.session_state: st.session_state.display_fps = 0
    if 'last_display_time' not in st.session_state: st.session_state.last_display_time = time.time()
    if 'display_frame_count' not in st.session_state: st.session_state.display_frame_count = 0
    if 'frames_detected' not in st.session_state: st.session_state.frames_detected = 0
    if 'last_frames_count_update' not in st.session_state: st.session_state.last_frames_count_update = 0
    if 'last_displayed_timestamp' not in st.session_state: st.session_state.last_displayed_timestamp = None

    # Check if we are supposed to use HTTP mode
    if 'http_client' not in st.session_state and st.session_state.get('use_http_mode', False):
        # Import the required modules first
        try:
            from jiffyget import get_active_sessions, get_session_port
            from jiffyclient import JiffyCamClient
            
            # Get the current session and check if it's active
            current_session = st.session_state.session
            data_dir = st.session_state.data_dir
            
            # First check if data directory exists
            if os.path.exists(data_dir):
                # Check if the current session is active
                try:
                    active_sessions = get_active_sessions(data_dir)
                    
                    # Only initialize HTTP client if the current session is active
                    if current_session in active_sessions:
                        port = get_session_port(current_session, data_dir)
                        if port:
                            # Update session state with the correct port
                            st.session_state.dataserver_port = port
                            st.session_state.http_server_port = port
                            
                            # Update the server URL with the correct port
                            st.session_state.http_server_url = f"http://localhost:{port}"
                            
                            # Initialize the client with the correct server URL but don't connect yet
                            st.session_state.http_client = JiffyCamClient(st.session_state.http_server_url)
                            # No immediate connection check - will be done when actually needed
                        else:
                            print(f"No valid port found for session '{current_session}'")
                    else:
                        print(f"Session '{current_session}' is not currently active")
                except Exception as e:
                    print(f"Error checking active sessions: {str(e)}")
                    # Don't create HTTP client if there was an error
            else:
                print(f"Data directory does not exist: {data_dir}")
                # Don't create HTTP client if data directory doesn't exist
        except ImportError as e:
            print(f"Could not import required modules: {str(e)}")
            # Don't create HTTP client if imports fail
        except Exception as e:
            print(f"Unexpected error during HTTP client initialization: {str(e)}")
            # Don't create HTTP client on any error

    # Get initial timestamp range - only on first load
    if 'oldest_timestamp' not in st.session_state or st.session_state.oldest_timestamp is None:
        try:
            oldest, newest, timestamps = get_timestamp_range(st.session_state.cam_name, st.session_state.session, st.session_state.data_dir)
            st.session_state.oldest_timestamp = oldest
            st.session_state.newest_timestamp = newest
            
            # After getting timestamp range, ensure init_date is within valid range
            current_date = st.session_state.init_date
            min_date = oldest.date() if oldest else None
            max_date = max(datetime.now().date(), newest.date() if newest else datetime.now().date())
            
            # Ensure init_date is within valid range
            if min_date and current_date < min_date:
                st.session_state.init_date = min_date
            elif max_date and current_date > max_date:
                st.session_state.init_date = max_date
                
        except Exception as e:
            print(f"Error getting timestamp range: {str(e)}")

    if 'browsing_date' not in st.session_state and 'init_date' in st.session_state: 
        st.session_state.browsing_date = st.session_state.init_date
    if 'date' not in st.session_state and 'init_date' in st.session_state:
        st.session_state.date = st.session_state.init_date
