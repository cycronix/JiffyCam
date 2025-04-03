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
        # Create target datetime for the selected time
    target_timestamp = time_posix * 1000.
    browse_date = datetime.fromtimestamp(time_posix).date()
    browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
    browse_delta = time_posix - browse_date_posix

    # Construct the base directory path
    base_dir = os.path.join(data_dir, session, str(browse_date_posix))
    #print(f"base_dir: {base_dir}, cam_name: {cam_name}, session: {session}, data_dir: {data_dir}")
    if not os.path.exists(base_dir):
        return None
    
    #browse_check = time_posix - (browse_date_posix + browse_delta)
    #print(f"browse_date_posix: {browse_date_posix}, time_posix: {time_posix}")
    #print(f"browse_delta: {browse_delta}, browse_check: {browse_check}")

    # Get all timestamp directories
    timestamp_dirs = glob.glob(os.path.join(base_dir, "*"))
    #print(f"timestamp_dirs: {timestamp_dirs}")
    if not timestamp_dirs:
        return None
    
    # Convert directory names to timestamps and filter by current date
    timestamps = []
    for dir_path in timestamp_dirs:
        try:
            #print(f"dir_path: {dir_path}")
            parts = os.path.normpath(dir_path).split(os.sep)

            if len(parts) >= 2:
                timestamp = int(parts[-2]) + int(parts[-1])
            else:
                timestamp = int(os.path.basename(dir_path))

            # Only include timestamps from the specified date
            #dt = datetime.fromtimestamp(timestamp / 1000)
            #if dt.date() == browse_date:

            timestamps.append((timestamp, dir_path))
        except ValueError:
            continue
    
    # If no timestamps found for the specified date, return None
    if not timestamps:
        return None
    
    # Sort timestamps
    timestamps.sort()
    
    # Find the closest timestamp based on direction
    closest_timestamp = None
    closest_dir = None
    if direction == "up":  # Find at-or-after for increasing time
        for timestamp, dir_path in timestamps:
            if timestamp >= target_timestamp:
                closest_timestamp = timestamp
                closest_dir = dir_path
                break
    else:  # Find at-or-before for decreasing time (default behavior)
        for timestamp, dir_path in timestamps:
            if timestamp <= target_timestamp:
                closest_timestamp = timestamp
                closest_dir = dir_path
            else:
                break
    
    if closest_dir is None:
        # If no match found and we have timestamps, return the oldest or newest based on direction
        # but only if they are from the same date
        if timestamps:
            if direction == "up":
                # If going up in time and no match, return the newest (last) timestamp
                closest_timestamp, closest_dir = timestamps[-1]
            else:
                # If going down in time and no match, return the oldest (first) timestamp
                closest_timestamp, closest_dir = timestamps[0]
        else:
            return None
    
    # Get the image file in the closest directory
    base_name = os.path.basename(cam_name)
    image_path = os.path.join(closest_dir, f"{base_name}.jpg")
    
    if os.path.exists(image_path):
        # Read the image and convert timestamp to datetime for display
        frame = cv2.imread(image_path)
        closest_datetime = datetime.fromtimestamp(closest_timestamp / 1000)
        return frame, closest_datetime
    
    return None

def get_timestamp_range(cam_name: str, session: str, data_dir: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the oldest and newest timestamps available for the camera.
    
    Args:
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        
    Returns:
        Tuple of (oldest_timestamp, newest_timestamp) as datetime objects, or (None, None) if no images
    """
    # Construct the base directory path
    base_dir = os.path.join(data_dir, session, os.path.dirname(cam_name))
    
    if not os.path.exists(base_dir):
        return None, None
    
    # Get all timestamp directories
    timestamp_dirs = glob.glob(os.path.join(base_dir, "*"))
    if not timestamp_dirs:
        return None, None
    
    # Convert directory names to timestamps
    timestamps = []
    for dir_path in timestamp_dirs:
        try:
            dir_name = os.path.basename(dir_path)
            timestamp = int(dir_name)
            timestamps.append(timestamp)
        except ValueError:
            continue
    
    if not timestamps:
        return None, None
    
    # Sort timestamps
    timestamps.sort()
    
    # Convert to datetime objects
    oldest = datetime.fromtimestamp(timestamps[0] / 1000)
    newest = datetime.fromtimestamp(timestamps[-1] / 1000)
    
    return oldest, newest

def detect_locations(cam_name: str, session: str, data_dir: str, browse_date: float):
    """Get all timestamps for the camera.
    
    Args:
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        browse_date: POSIX timestamp of the browse date (float)
    Returns:
        List of float detection positions 0-1 for specified browse_date
    """
    # Construct the base directory path
    base_dir = os.path.join(data_dir, session, os.path.dirname(cam_name))
    if not os.path.exists(base_dir):
        return []
    
    # Get all timestamp directories
    browse_date_str = str(int(browse_date)*1000)
    timestamp_dirs = glob.glob(os.path.join(base_dir, browse_date_str, "*"))
    if not timestamp_dirs:
        return []
    
    # Convert directory names to timestamps
    timestamps = []
    for dir_path in timestamp_dirs:
        try:
            dir_name = os.path.basename(dir_path)
            timestamp = float(dir_name) / 86400000
            timestamps.append(timestamp)
        except ValueError:
            continue
    
    timestamps.sort()
    return timestamps  