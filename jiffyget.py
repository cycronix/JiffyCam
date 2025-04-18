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
    
    # Get timestamps data if not already available
    timestamp_data = timestamps
    if not timestamp_data:
        timestamp_data = get_timestamps(cam_name, session, data_dir, browse_date_posix)
        # Update the global timestamps
        timestamps = timestamp_data
        
    if not timestamp_data:
        return None, None, True
    
    # Find the closest timestamp based on direction
    timestamp_data.sort()
    oldest_timestamp = timestamp_data[0][0]
    newest_timestamp = timestamp_data[-1][0] 
    #print(f"oldest_timestamp: {oldest_timestamp}, newest_timestamp: {newest_timestamp}, target_timestamp: {target_timestamp}")
    # If closest timestamp is outside range of timestamps, use oldest or newest
    if(target_timestamp <= oldest_timestamp):
        closest_dir = timestamp_data[0][1]
        closest_timestamp = timestamp_data[0][0]
        eof = True
    elif(target_timestamp >= newest_timestamp):
        closest_dir = timestamp_data[-1][1]
        closest_timestamp = newest_timestamp
        eof = True
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
    
    # Force a fresh scan to ensure we have up-to-date information
    timestamps = None
    
    # Get all valid timestamps for this recording
    all_timestamps = get_timestamps(cam_name, session, data_dir, None)
    
    if all_timestamps is None or len(all_timestamps) == 0:
        print(f"No valid timestamps found for {cam_name} in session {session}")
        return None, None

    # Convert to datetime objects
    oldest = datetime.fromtimestamp(all_timestamps[0][0] / 1000)
    newest = datetime.fromtimestamp(all_timestamps[-1][0] / 1000)
    
    # Get a list of all unique dates that actually have images
    unique_dates = set()
    for ts, _ in all_timestamps:
        dt = datetime.fromtimestamp(ts / 1000)
        unique_dates.add(dt.date())
    
    # Store the timestamps in the global variable for future reference
    timestamps = all_timestamps
    
    # Log the date range we found
    #print(f"Valid date range for {session}: {oldest.date()} to {newest.date()} ({len(unique_dates)} days with data)")
    
    return oldest, newest

def get_timestamps(cam_name: str, session: str, data_dir: str, browse_date: int):
    """Get all timestamps for the camera.
    
    Args:
        cam_name: Camera name
        session: Session name
        data_dir: Data directory path
        browse_date: POSIX timestamp of the browse date (int)
    Returns:
        List of timestamp tuples (timestamp, dir_path) for specified browse_date
    """
    global timestamps

    # Get all timestamps for the session
    if browse_date is None:
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
            print(f"No timestamps found for camera {cam_name} in session {session}")
            return None
            
        # Sort all timestamps from all date directories
        all_timestamps.sort()
        return all_timestamps
    else:
        # Regular case: browsing within a specific date
        base_dir = os.path.join(data_dir, session, str(browse_date))
        if not os.path.exists(base_dir):
            return None
            
        dirpaths = glob.glob(os.path.join(base_dir, "*"))
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

        if len(timestamps) == 0:
            return None

        timestamps.sort()
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

    timestamp_data = get_timestamps(cam_name, session, data_dir, browse_date)
    if timestamp_data is None:
        return None

    # Store the data in global timestamps for other functions to access
    timestamps = timestamp_data

    locations = []       
    for timestamp in timestamp_data:
        location = (timestamp[0]-browse_date) / 86400000.    # fraction of day (msec/msec)
        locations.append(location)

    #locations.sort()
    return locations  
