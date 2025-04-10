"""
jiffyget.py: Helper module for JiffyCam to handle image retrieval

This module provides functions to find and load saved camera images,
"""

import os
import glob
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
import cv2

global timestamps
timestamps = None

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
    global timestamps
    #print(f"jiffyget: time_posix: {time_posix}, cam_name: {cam_name}, session: {session}, data_dir: {data_dir}, direction: {direction}")
    # Create target datetime for the selected time
    target_timestamp = time_posix * 1000.
    browse_date = datetime.fromtimestamp(time_posix).date()
    browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)

    # Construct the base directory path
    base_dir = os.path.join(data_dir, session, str(browse_date_posix))
    #print(f"base_dir: {base_dir}, cam_name: {cam_name}, session: {session}, data_dir: {data_dir}")
    if not os.path.exists(base_dir):
        return None, None, True
    
    # Convert directory names to timestamps and filter by current date
    if(not timestamps):
        timestamps = get_timestamps(cam_name, session, data_dir, browse_date_posix)
    if(not timestamps):
        return None, None, True
    
    # Find the closest timestamp based on direction
    timestamps.sort()
    oldest_timestamp = timestamps[0][0]
    newest_timestamp = timestamps[-1][0] 
    #print(f"oldest_timestamp: {oldest_timestamp}, newest_timestamp: {newest_timestamp}, target_timestamp: {target_timestamp}")
    # If closest timestamp is outside range of timestamps, use oldest or newest
    if(target_timestamp <= oldest_timestamp):
        closest_dir = timestamps[0][1]
        closest_timestamp = timestamps[0][0]
        eof = True
    elif(target_timestamp >= newest_timestamp):
        closest_dir = timestamps[-1][1]
        closest_timestamp = newest_timestamp
        eof = True
    else:
        eof = False
        closest_dir = None
        closest_timestamp = None 
        if direction == "up":  # Find at-or-after for increasing time
            for timestamp, dir_path in timestamps:
                if timestamp >= target_timestamp:
                    closest_dir = dir_path
                    closest_timestamp = timestamp
                    break
        elif direction == "down":  # Find at-or-before for decreasing time (default behavior)
            for timestamp, dir_path in timestamps:
                if timestamp <= target_timestamp:
                    closest_dir = dir_path
                    closest_timestamp = timestamp
                else:
                    break
        else:
            dt = 0
            prev_dt = 0
            prev_timestamp = None
            prev_dir = None
            for timestamp, dir_path in timestamps:
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
        # Read the image and convert timestamp to datetime for display
        frame = cv2.imread(image_path)
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
    global timestamps
    if(timestamps is None):
        timestamps = get_timestamps(cam_name, session, data_dir, None)

    if(timestamps is None):
        return None, None

    # Convert to datetime objects
    oldest = datetime.fromtimestamp(timestamps[0][0] / 1000)
    newest = datetime.fromtimestamp(timestamps[-1][0] / 1000)
    return oldest, newest    

def get_timestamps(cam_name: str, session: str, data_dir: str, browse_date: int):
    """Get all timestamps for the camera.
    
    Args:
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        browse_date: POSIX timestamp of the browse date (int)
    Returns:
        List of float detection positions 0-1 for specified browse_date
    """
    global timestamps

    # get all timestamps for the session
    if(browse_date is None):
        browse_date = 0
        base_dir = os.path.join(data_dir, session)
    else:
        base_dir = os.path.join(data_dir, session, str(browse_date))

    dirpaths = glob.glob(os.path.join(base_dir, "*"))

    timestamps = []  
    for dir_path in dirpaths:
        parts = os.path.normpath(dir_path).split(os.sep)
        timestamp = int(browse_date + float(parts[-1]))   
        timestamps.append((timestamp, dir_path))    

    if(len(timestamps) == 0):
        return None

    timestamps.sort()
    #print(f"timestamps: {timestamps}, base_dir: {base_dir}")    
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
    global timestamps

    timestamps = get_timestamps(cam_name, session, data_dir, browse_date)
    if(timestamps is None):
        return None

    locations = []       
    for timestamp in timestamps:
        location = (timestamp[0]-browse_date) / 86400000.    # fraction of day (msec/msec)
        locations.append(location)

    #locations.sort()
    return locations  
