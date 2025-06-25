"""
vidcap_streamlit: Streamlit-based video capture utility

A modern version of vidcap that uses Streamlit for the UI to capture video 
from a camera and send it to a CloudTurbine (CT) server.
"""

import os
import time
import yaml
import argparse
from typing import Optional, Dict, Any
from datetime import datetime, time as datetime_time
from collections import OrderedDict

import streamlit as st
import logging
import streamlit as st

import logging
logging.basicConfig(level=logging.ERROR)
st_logger = logging.getLogger('streamlit')
st_logger.setLevel(logging.ERROR)

# Import Jiffy modules
from jiffyget import get_timestamp_range, get_active_sessions, get_session_port # get_frame moved to jiffyui 
from jiffyconfig import JiffyConfig

# Import UI functions from the new modules
from jiffyui import (
    build_sidebar, 
    build_main_area, 
    run_ui_update_loop,
    # Callbacks and helpers are internal to jiffyui now
)
# UI components are imported directly in the relevant functions
#import jiffyui_components
from jiffyclient import JiffyCamClient

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# --- Parse Command Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='JiffyCam Streamlit Web Interface')
    parser.add_argument('data_dir', nargs='?', default=None, 
                        help='Optional data directory to use instead of default JiffyData')
    return parser.parse_args()

# --- Main Application Logic ---
def main():       
    # Parse command line arguments
    args = parse_args()
    
    # Page Config (Should be first Streamlit command)
    st.set_page_config(
        page_title="JiffyCam",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load config with the data_dir from command line if provided
    data_dir = args.data_dir if args.data_dir else 'JiffyData'
    config_manager = JiffyConfig(data_dir=data_dir)
    config = config_manager.config
    
    # Ensure data_dir is set in the config
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    # UI interaction state flags
    if 'in_playback_mode' not in st.session_state:          st.session_state.in_playback_mode = True
    if 'rt_capture' not in st.session_state:                st.session_state.rt_capture = False
    if 'need_to_display_recent' not in st.session_state:    st.session_state.need_to_display_recent = False # mjm:  False for now
    if 'status_message' not in st.session_state:            st.session_state.status_message = "Initializing..."
    if 'step_direction' not in st.session_state:            st.session_state.step_direction = None
    if 'autoplay_direction' not in st.session_state:        st.session_state.autoplay_direction = None
    if 'autoplay_step' not in st.session_state:             st.session_state.autoplay_step = False
    if 'autoplay_interval' not in st.session_state:         st.session_state.autoplay_interval = 0.05
    if 'needs_date_update' not in st.session_state:         st.session_state.needs_date_update = True
    if 'dataserver_port' not in st.session_state:           st.session_state.dataserver_port = int(config.get('dataserver_port', 8080))
    if 'video_placeholder' not in st.session_state:         st.session_state.video_placeholder = None
    
    # Set http_server_port to match dataserver_port for consistency
    if 'http_server_port' not in st.session_state:
        st.session_state.http_server_port = st.session_state.dataserver_port
    
    # Build HTTP server URL with the configured port
    if 'http_server_url' not in st.session_state:
        st.session_state.http_server_url = f"http://localhost:{st.session_state.dataserver_port}"
    
    # Initialize HTTP client if in HTTP mode and we have a valid URL
    if 'http_client' not in st.session_state:
        try:
            st.session_state.http_client = JiffyCamClient(st.session_state.http_server_url)
        except Exception as e:
            print(f"Error initializing HTTP client: {str(e)}")
    
    # Set data directory and ensure it exists
    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = data_dir
        os.makedirs(st.session_state.data_dir, exist_ok=True)
    
    # Initialize cam_name
    if 'cam_name' not in st.session_state:
        st.session_state.cam_name = 'cam0'
    
    # Get list of existing directories in data_dir
    existing_dirs = [d for d in os.listdir(st.session_state.data_dir) 
                    if os.path.isdir(os.path.join(st.session_state.data_dir, d))]
    
    # Initialize session to first existing directory or 'Default' if none exist
    if 'session' not in st.session_state:
        st.session_state.session = existing_dirs[0] if existing_dirs else 'Default'
    
    # Time/Date related state
    current_time = datetime.now()
    if 'hour' not in st.session_state: st.session_state.hour = current_time.hour
    if 'minute' not in st.session_state: st.session_state.minute = current_time.minute
    if 'second' not in st.session_state: st.session_state.second = current_time.second
    if 'microsecond' not in st.session_state: st.session_state.microsecond = current_time.microsecond
    
    # Default date to today (will be adjusted after timestamp range is determined)
    if 'init_date' not in st.session_state: st.session_state.init_date = current_time.date()
    
    # Image/Timestamp state
    if 'last_frame' not in st.session_state: st.session_state.last_frame = None
    if 'last_timestamp' not in st.session_state: st.session_state.last_timestamp = 0
    if 'actual_timestamp' not in st.session_state: st.session_state.actual_timestamp = None
    if 'oldest_timestamp' not in st.session_state: st.session_state.oldest_timestamp = None
    if 'newest_timestamp' not in st.session_state: st.session_state.newest_timestamp = None

    # Get initial timestamp range
    if ('oldest_timestamp' not in st.session_state or st.session_state.oldest_timestamp is None):
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
            st.session_state.oldest_timestamp = None
            st.session_state.newest_timestamp = None

    if 'browsing_date' not in st.session_state and 'init_date' in st.session_state: 
        st.session_state.browsing_date = st.session_state.init_date
    
    #print(f"st.session_state: {st.session_state}")

    # Add performance tracking variables
    if 'frames_detected' not in st.session_state: st.session_state.frames_detected = 0
    if 'last_displayed_timestamp' not in st.session_state: st.session_state.last_displayed_timestamp = None

    # --- Build UI --- 
    # Call UI builders and store returned placeholders in session_state
    # These keys ('status_placeholder', etc.) must match those used in jiffyui callbacks
    
    # Initialize placeholder session state variables before calling UI builders
    st.session_state.status_placeholder, st.session_state.error_placeholder = build_sidebar()
    
    #st.session_state.video_placeholder, 
    foo, \
        st.session_state.time_display, st.session_state.timearrow_placeholder, st.session_state.timeline_placeholder = \
        build_main_area()
    #st.session_state.video_placeholder = None

    #st.session_state.video_placeholder.info("Initialize capture or select time.")
    # st.session_state.status_placeholder.markdown("<div style='padding: 5px 0;'>Status: Idle</div>", unsafe_allow_html=True)
    st.session_state.status_message = "Status: Idle"

    # --- Run Main UI Update Loop --- 
    #print(f"running main ui update loop")
    #st.session_state.needs_date_update = True
    run_ui_update_loop() # Loop fetches placeholders from session_state

def seconds_since_midnight(dt):
    return (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

# --- Entry Point --- 
if __name__ == "__main__":
    main() 
    