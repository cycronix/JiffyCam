"""
jiffyget.py: Helper module for JiffyCam to handle image retrieval

This module provides functions to find and load saved camera images,
"""

import os
import glob
import time
import yaml
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import cv2

# Cache for active sessions to reduce HTTP requests
cached_active_sessions = {}
cache_expiry_time = 0

def get_active_sessions(data_dir: str, cache_timeout: int = 5) -> List[str]:
    """Check which sessions are currently active by querying their HTTP endpoints.
    
    Args:
        data_dir: Base data directory containing session folders
        cache_timeout: Number of seconds to cache results (default: 5)
    
    Returns:
        List of active session names
    """
    global cached_active_sessions, cache_expiry_time
    
    # Check if we have a valid cache
    current_time = time.time()
    if current_time < cache_expiry_time and cached_active_sessions:
        return list(cached_active_sessions.keys())
    
    # Reset cache
    cached_active_sessions = {}
    cache_expiry_time = current_time + cache_timeout
    
    # Get all session directories
    if not os.path.exists(data_dir):
        return []
    
    session_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    active_sessions = []
    
    for session in session_dirs:
        # Check if session has a config file
        config_path = os.path.join(data_dir, session, 'jiffycam.yaml')
        if not os.path.exists(config_path):
            continue
        
        # Read config to get HTTP server port
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get server port from config
            port = config.get('dataserver_port', 8080)
            
            # Try to connect to the server
            server_url = f"http://localhost:{port}/status"
            try:
                response = requests.get(server_url, timeout=0.5)
                if response.status_code == 200:
                    # Server is active - save both session name and port
                    status_data = response.json()
                    
                    # Make sure the session claimed by the server matches this session
                    server_session = status_data.get('session')
                    active_session = status_data.get('active_session', server_session)
                    
                    if active_session == session:
                        cached_active_sessions[session] = {
                            'port': port,
                            'status': status_data
                        }
                        active_sessions.append(session)
            except requests.RequestException:
                # Connection failed - server not active
                pass
                
        except Exception as e:
            # Error reading config or connecting - continue to next session
            print(f"Error checking session {session}: {str(e)}")
            continue
    
    return active_sessions

def get_session_port(session: str, data_dir: str) -> Optional[int]:
    """Get the HTTP server port for a session.
    
    Args:
        session: Session name
        data_dir: Base data directory containing session folders
    
    Returns:
        Port number if session is active, None otherwise
    """
    global cached_active_sessions, cache_expiry_time
    
    # Check if we need to refresh the cache
    current_time = time.time()
    if current_time >= cache_expiry_time:
        get_active_sessions(data_dir)
    
    # Check if session is in the cache
    if session in cached_active_sessions:
        return cached_active_sessions[session]['port']
    
    # If not in cache but cache is still valid, session is not active
    if current_time < cache_expiry_time:
        return None
    
    # Otherwise, try to read the config file directly
    config_path = os.path.join(data_dir, session, 'jiffycam.yaml')
    if not os.path.exists(config_path):
        # If the session-specific config doesn't exist, don't return a default port
        print(f"No config file found for session '{session}' at {config_path}")
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Only return the port if it's explicitly specified in the config
        if config and 'dataserver_port' in config:
            return config['dataserver_port']
        else:
            print(f"No dataserver_port specified in config for session '{session}'")
            return None
    except Exception as e:
        print(f"Error reading config for session '{session}': {str(e)}")
        return None

def reset_timestamps():
    global timestamp_cache
    timestamp_cache = {}
    return

def reset_image_cache():
    global image_cache
    image_cache = {}
    #print("cleared image_cache")
    return   

def jiffyget(time_posix: float, cam_name: str, 
             session: str, data_dir: str, 
             direction: str = "down"):
    """Find the closest image to the given time.
    
    Args:
        time_posix: POSIX timestamp (float)
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        direction: Direction to search ("up" or "down")
        
    Returns:
        Tuple of (frame, timestamp) or None if no image found
    """
    #print(f"jiffyget: time_posix: {time_posix}")
    # Create target datetime for the selected time
    target_timestamp = time_posix * 1000.
    browse_date = datetime.fromtimestamp(time_posix).date()
    browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)

    # Construct the base directory path
    base_dir = os.path.join(data_dir, session, str(browse_date_posix))
    if not os.path.exists(base_dir):
        return None, None, True
    
    # Get timestamps data if not already available
    timestamp_data = get_timestamps(cam_name, session, data_dir, browse_date_posix)        
    if not timestamp_data:
        return None, None, True
    
    #print(f"jiffyget: timestamp_data: {timestamp_data}")
    # Find the closest timestamp based on direction
    timestamp_data.sort()
    oldest_timestamp = timestamp_data[0][0]
    newest_timestamp = timestamp_data[-1][0] 
    #print(f"oldest_timestamp: {oldest_timestamp}, newest_timestamp: {newest_timestamp}, target_timestamp: {target_timestamp}")
    # If closest timestamp is outside range of timestamps, use oldest or newest

    eof = False
    if(target_timestamp <= oldest_timestamp):
        closest_dir = timestamp_data[0][1]
        closest_timestamp = oldest_timestamp
        if(direction == "down"):
            eof = True
        #print(f"jiffyget: eof: {eof}, direction: {direction}")
    elif(target_timestamp >= newest_timestamp):
        closest_dir = timestamp_data[-1][1]
        closest_timestamp = newest_timestamp
        if(direction == "up"):
            eof = True
        #print(f"jiffyget: eof: {eof}, direction: {direction}")
    else:
        eof = False
        closest_dir = None
        closest_timestamp = None 
        if direction == "up":  # Find at-or-after for increasing time
            for timestamp, dir_path in timestamp_data:
                if timestamp >= target_timestamp:
                    closest_dir = dir_path
                    closest_timestamp = timestamp
                    break
        elif direction == "down":  # Find at-or-before for decreasing time (default behavior)
            for timestamp, dir_path in timestamp_data:
                if timestamp <= target_timestamp:
                    closest_dir = dir_path
                    closest_timestamp = timestamp
                else:
                    #print(f"jiffyget: timestamp: {timestamp}, dir_path: {dir_path}, target_timestamp: {target_timestamp}")
                    break
        else:
            dt = 0
            prev_dt = 0
            prev_timestamp = None
            prev_dir = None
            for timestamp, dir_path in timestamp_data:
                dt = timestamp - target_timestamp
                if dt > 0:
                    if dt < -prev_dt:
                        closest_dir = dir_path
                        closest_timestamp = timestamp
                    else:
                        closest_dir = prev_dir
                        closest_timestamp = prev_timestamp
                    break
                prev_dt = dt
                prev_timestamp = timestamp
                prev_dir = dir_path

    if closest_dir is None:
        return None, None, eof

    # Get the image file in the closest directory
    base_name = os.path.basename(cam_name)
    image_path = os.path.join(closest_dir, f"{base_name}.jpg")
    
    if os.path.exists(image_path):
        #print(f"jiffyget: image_path: {image_path}, closest_timestamp: {closest_timestamp}")
        # Check if image is already cached
        global image_cache
        if False and image_path in image_cache:
            frame = image_cache[image_path]
        else:
            # Read the image and cache it
            frame = cv2.imread(image_path)
            #image_cache[image_path] = frame    # mjm:  don't cache images
            #print(f"jiffyget: frame: {image_path}, shape: {frame.shape}")
        #closest_datetime = datetime.fromtimestamp(closest_timestamp / 1000)
        return frame, closest_timestamp, eof
    
    return None, None, eof

def get_timestamp_range(cam_name: str, session: str, data_dir: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the oldest and newest timestamps available for the camera across all dates.
    
    Args:
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        
    Returns:
        Tuple of (oldest_timestamp, newest_timestamp) as datetime objects, or (None, None) if no images
    """
        
    # Get all valid timestamps for this recording
    all_timestamps = get_timestamps(cam_name, session, data_dir, None)
    
    if all_timestamps is None or len(all_timestamps) == 0:
        #print(f"No valid timestamps found for session {session}")    # warning printed in get_timestamps
        return None, None, None

    # Convert to datetime objects
    oldest = datetime.fromtimestamp(all_timestamps[0][0] / 1000)
    newest = datetime.fromtimestamp(all_timestamps[-1][0] / 1000)
    
    # Get a list of all unique dates that actually have images
    unique_dates = set()
    for ts, _ in all_timestamps:
        dt = datetime.fromtimestamp(ts / 1000)
        unique_dates.add(dt.date())
            
    return oldest, newest, all_timestamps

import inspect
timestamp_cache = {}
image_cache = {}
def get_timestamps(cam_name: str, session: str, data_dir: str, browse_date):
    """Get all timestamps for the camera.
    
    Args:
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        browse_date: POSIX timestamp of the browse date (int)
    Returns:
        List of timestamp tuples (timestamp, dir_path) for specified browse_date
    """
    global timestamp_cache

    # Get all timestamps for the session
    if browse_date is None:
        reset_image_cache()  # clear image cache when browsing all dates
        # mjm:  this gets called from get_timestamp_range with browse_date = None
        #print(f"get_timestamps: browsing all dates in {session}")
        # When browsing_date is None, we need to scan all date directories
        base_dir = os.path.join(data_dir, session)
        if not os.path.exists(base_dir):
            print(f"Directory doesn't exist: {base_dir}")
            return None
            
        # Scan for date directories (millisecond timestamp directories)
        date_dirs = glob.glob(os.path.join(base_dir, "*"))
        
        # Initialize collection for all timestamps across dates
        all_timestamps = []
        
        # Process each date directory to extract timestamps
        for date_dir in date_dirs:
            # Skip non-directories and non-numeric directories
            if not os.path.isdir(date_dir):
                continue
                
            date_name = os.path.basename(date_dir)
            try:
                # Try to convert directory name to timestamp (milliseconds)
                date_timestamp = int(date_name)
                
                # Also check if this directory actually contains images
                # by checking for subdirectories that contain the cam_name.jpg file
                base_name = os.path.basename(cam_name)
                has_images = False
                
                # Get timestamp directories within this date directory
                time_dirs = glob.glob(os.path.join(date_dir, "*"))
                
                # Process each timestamp directory
                for time_dir in time_dirs:
                    if not os.path.isdir(time_dir):
                        continue
                        
                    # Check if this time_dir contains an image for this camera
                    image_path = os.path.join(time_dir, f"{base_name}.jpg")
                    if not os.path.exists(image_path):
                        continue
                    
                    # We found an image, so this date is valid
                    has_images = True
                        
                    # Get the time offset within the day
                    time_offset = os.path.basename(time_dir)
                    try:
                        # Combine date timestamp and time offset
                        timestamp = date_timestamp + int(time_offset)
                        all_timestamps.append((timestamp, time_dir))
                    except ValueError:
                        # Skip if time offset can't be converted to int
                        continue
                        
                # If no valid images were found in this date directory, we should log it
                if not has_images:
                    print(f"No valid images found for camera {cam_name} in date directory: {date_dir}")
                    
            except ValueError:
                # Skip if date directory name can't be converted to int
                continue
                
        if len(all_timestamps) == 0:
            print(f"No timestamps found for session: {session}")
            return None
        else:
            # Sort all timestamps from all date directories
            all_timestamps.sort()

        timestamp_cache = {}      # clear cache when browsing all dates
        #print("cleared timestamp_cache")
        return all_timestamps
    else:
        # Regular case: browsing within a specific date
        base_dir = os.path.join(data_dir, session, str(browse_date))
        if not os.path.exists(base_dir):
            return None
            
        if base_dir in timestamp_cache:
            return timestamp_cache[base_dir]
        
        #print(f"get_timestamps: base_dir: {base_dir}, browse_date: {browse_date}, caller: {inspect.stack()[1].function}")

        dirpaths = glob.glob(os.path.join(base_dir, "*"))
        #print(f"get_timestamps: session: {session}, browse_date: {browse_date}, base_dir: {base_dir}, caller: {inspect.stack()[1].function}")
        base_name = os.path.basename(cam_name)
        timestamps = []  
        for dir_path in dirpaths:
            if not os.path.isdir(dir_path):
                continue
                
            # Check if this directory contains an image for this camera
            image_path = os.path.join(dir_path, f"{base_name}.jpg")
            if not os.path.exists(image_path):
                continue
                
            parts = os.path.normpath(dir_path).split(os.sep)
            try:
                timestamp = int(browse_date + float(parts[-1]))   
                timestamps.append((timestamp, dir_path))
            except ValueError:
                continue    

        #print(f"base_dir: {base_dir}, session: {session}, date_timestamps: {len(timestamps)}, caller: {inspect.stack()[1].function}")
        if len(timestamps) == 0:
            return None

        timestamps.sort()
        timestamp_cache[base_dir] = timestamps
        return timestamps

def get_locations(cam_name: str, session: str, data_dir: str, browse_date: int):
    """Get all timestamps for the camera.
    
    Args:
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        browse_date: POSIX timestamp of the browse date (int)
    Returns:
        List of float detection positions 0-1 for specified browse_date
    """
    reset_timestamps()
    
    timestamp_data = get_timestamps(cam_name, session, data_dir, browse_date)
    if timestamp_data is None:
        return None

    locations = []       
    for timestamp in timestamp_data:
        location = (timestamp[0]-browse_date) / 86400000.    # fraction of day (msec/msec)
        locations.append(location)

    #locations.sort()
    return locations  
