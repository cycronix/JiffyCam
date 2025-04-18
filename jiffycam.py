"""
vidcap_streamlit: Streamlit-based video capture utility

A modern version of vidcap that uses Streamlit for the UI to capture video 
from a camera and send it to a CloudTurbine (CT) server.
"""

import os
# import threading # Keep only if other non-UI threads are used
from typing import Optional, Dict, Any
import time

# from queue import Queue # No longer needed here
from datetime import datetime, time as datetime_time
from collections import OrderedDict

import streamlit as st
#from streamlit_server_state import server_state, server_state_lock

# Import Jiffy modules
# from jiffyput import jiffyput # Not directly used here
from jiffyget import get_timestamp_range # get_frame moved to jiffyui 
from jiffyconfig import RESOLUTIONS, JiffyConfig

# Import UI functions from the new module
from jiffyui import (
    build_sidebar, 
    build_main_area, 
    run_ui_update_loop,
    JiffyCamClient,
    # Callbacks and helpers are internal to jiffyui now
)

# --- Main Application Logic ---
def main():       
    # Page Config (Should be first Streamlit command)
    st.set_page_config(
        page_title="JiffyCam",
        page_icon="🎥",
        layout="wide"
    )
    
    # --- Initialize Session State --- 
    # Needs to happen *before* accessing state in UI build or logic
    #if 'is_capturing' not in server_state:
    #    server_state.is_capturing = False   
    #if 'last_frame' not in server_state:
    #    server_state.last_frame = None
    #if 'timestamp' not in server_state:
    #    server_state.timestamp = datetime.now()

    # Core components
    #if('slave_mode' not in st.session_state):                          # new session
    #    st.session_state.slave_mode = server_state.is_capturing        # some other session is capturing, so we're in slave mode

    #print(f"server_state: {server_state.is_capturing}")
    #print(f"st.session_state.slave_mode: {st.session_state.slave_mode}")

    # Load config
    config_manager = JiffyConfig()
    config = config_manager.config
    
    # UI interaction state flags
    if 'in_playback_mode' not in st.session_state: st.session_state.in_playback_mode = True
    if 'rt_capture' not in st.session_state: st.session_state.rt_capture = False
    if 'need_to_display_recent' not in st.session_state: st.session_state.need_to_display_recent = True
    if 'live_button_clicked' not in st.session_state: st.session_state.live_button_clicked = False
    if 'status_message' not in st.session_state: st.session_state.status_message = "Initializing..."
    if 'image_just_saved' not in st.session_state: st.session_state.image_just_saved = False
    if 'step_direction' not in st.session_state: st.session_state.step_direction = None
    if 'autoplay_direction' not in st.session_state: st.session_state.autoplay_direction = None
    
    # Default to HTTP mode
    if 'use_http_mode' not in st.session_state: st.session_state.use_http_mode = True
    
    # Initialize dataserver_port from config
    if 'dataserver_port' not in st.session_state: 
        st.session_state.dataserver_port = int(config.get('dataserver_port', 8080))
    
    # Set http_server_port to match dataserver_port for consistency
    if 'http_server_port' not in st.session_state:
        st.session_state.http_server_port = st.session_state.dataserver_port
    
    # Build HTTP server URL with the configured port
    if 'http_server_url' not in st.session_state:
        st.session_state.http_server_url = f"http://localhost:{st.session_state.dataserver_port}"
    
    # Initialize HTTP client if in HTTP mode
    if st.session_state.use_http_mode and 'http_client' not in st.session_state:
        st.session_state.http_client = JiffyCamClient(st.session_state.http_server_url)
    
    # Configuration related state (derived from config)
    # Ensure device_aliases is OrderedDict
    aliases = config.get('device_aliases', {'Default': '0'})
    if 'device_aliases' not in st.session_state: st.session_state.device_aliases = OrderedDict(aliases) 
    else: st.session_state.device_aliases = OrderedDict(st.session_state.device_aliases) # Ensure type

    if 'cam_name' not in st.session_state: st.session_state.cam_name = config.get('cam_name', 'cam0')
    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = config.get('data_dir', 'JiffyData')
        os.makedirs(st.session_state.data_dir, exist_ok=True)
    if 'resolution' not in st.session_state:
        config_res = config.get('resolution', '1080p (1920x1080)')
        matched_key = None
        if isinstance(config_res, str) and 'x' in config_res:
            for key, (w,h) in RESOLUTIONS.items():
                if f"{w}x{h}" == config_res: matched_key = key; break
        st.session_state.resolution = matched_key or config_res # Use key if found, else config value
    if 'save_interval' not in st.session_state: st.session_state.save_interval = int(config.get('save_interval', 60))
    
    # Determine initial cam_device (path/ID) and selected_device_alias (UI key)
    if 'selected_device_alias' not in st.session_state or 'cam_device' not in st.session_state:
        config_device_val = config.get('cam_device', 'Default') # This is likely the alias key
        available_aliases = st.session_state.device_aliases
        
        if config_device_val in available_aliases:
            st.session_state.selected_device_alias = config_device_val
            st.session_state.cam_device = available_aliases[config_device_val]
        else: # Config value might be a direct path/ID (legacy or manual edit)
            st.session_state.cam_device = config_device_val 
            # Find the alias key that matches this path/ID
            matching_alias = None
            for alias, path in available_aliases.items():
                if path == config_device_val:
                    matching_alias = alias
                    break
            # Set selected alias to the match, or fallback to first available alias
            st.session_state.selected_device_alias = matching_alias or list(available_aliases.keys())[0] 

    if 'session' not in st.session_state: st.session_state.session = st.session_state.selected_device_alias
    
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

    # Get initial timestamp range
    if 'oldest_timestamp' not in st.session_state or st.session_state.oldest_timestamp is None:
        try:
            oldest, newest = get_timestamp_range(st.session_state.cam_name, st.session_state.session, st.session_state.data_dir)
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
            st.session_state.oldest_timestamp = None
            st.session_state.newest_timestamp = None

    if 'browsing_date' not in st.session_state and 'init_date' in st.session_state: 
        st.session_state.browsing_date = st.session_state.init_date
    if 'time_slider' not in st.session_state: 
         st.session_state.time_slider = datetime_time(st.session_state.hour, st.session_state.minute, st.session_state.second)
    
    # Add performance tracking variables
    if 'capture_fps' not in st.session_state: st.session_state.capture_fps = 0
    if 'display_fps' not in st.session_state: st.session_state.display_fps = 0
    if 'last_display_time' not in st.session_state: st.session_state.last_display_time = time.time()
    if 'display_frame_count' not in st.session_state: st.session_state.display_frame_count = 0
    if 'frames_detected' not in st.session_state: st.session_state.frames_detected = 0
    if 'last_frames_count_update' not in st.session_state: st.session_state.last_frames_count_update = 0
    if 'last_displayed_timestamp' not in st.session_state: st.session_state.last_displayed_timestamp = None
    #if 'last_save_time' not in st.session_state: st.session_state.last_save_time = None

    # --- Build UI --- 
    # Call UI builders and store returned placeholders in session_state
    # These keys ('status_placeholder', etc.) must match those used in jiffyui callbacks
    st.session_state.status_placeholder, st.session_state.error_placeholder, st.session_state.capture_fps_placeholder, \
        st.session_state.display_fps_placeholder, st.session_state.frames_detected_placeholder, st.session_state.last_save_time_placeholder = build_sidebar()
    st.session_state.video_placeholder, st.session_state.time_display, st.session_state.timearrow_placeholder = \
        build_main_area()

    st.session_state.video_placeholder.info("Initialize capture or select time.")
    st.session_state.status_placeholder.markdown("<div style='padding: 5px 0;'>Status: Idle</div>", unsafe_allow_html=True)
    st.session_state.status_message = "Status: Idle"

    # --- Run Main UI Update Loop --- 
    #print(f"is_capturing: {get_is_capturing()}")
    run_ui_update_loop() # Loop fetches placeholders from session_state

# --- Entry Point --- 
if __name__ == "__main__":
    main() 
    