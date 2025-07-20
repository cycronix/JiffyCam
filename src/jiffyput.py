"""
jiffyput.py: Helper module for JiffyCam to handle frame processing and storage

This module provides functions to process and save video frames,
"""

import os
import cv2
import time
import glob
from datetime import datetime, timedelta
from shutil import rmtree
from pathlib import Path
from jiffydetect import detect



prevframe = None
def jiffyput(cam_name, frame, time_posix: float, session, data_dir, weights_path, save_frame: bool, detect_frame: bool, save_days=None, enable_tiling=False):
    """
    Process and save a video frame.

    Args:
        cam_name (str): The name of the camera
        frame: The video frame to process
        time_posix (float): The frame timestamp
        session (str): The session name
        data_dir (str): The data directory
        weights_path (str): Path to the YOLO weights file
        save_frame (bool): Whether to save the frame
        detect_frame (bool): Whether to run detection on the frame
        save_days (int, optional): Number of days to keep data before considering it old
        enable_tiling (bool): Whether to enable tiling for large images (default: False)

    Returns:
        The processed frame (which may be modified by detection)
    """

    try:
        #print(f"jiffyput: save_frame: {save_frame}, detect_frame: {detect_frame}")
        # Set JPEG quality to 95
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        #detect_mode = True
        if detect_frame:
            tryframe = detect(frame, weights_path, enable_tiling)  # pass enable_tiling parameter
            if tryframe is not None:       # ONLY save detections!  (to do: heartbeat saves)
                save_frame = True  # save the detection frame
                frame = tryframe

        if not save_frame:
            return None

        # Create save directory using timestamp
        #timestamp_ms = int(time_posix * 1000)
        browse_date = datetime.fromtimestamp(time_posix).date()
        browse_date_ms = int(time.mktime(browse_date.timetuple()) * 1000)
        browse_delta_ms = int(time_posix * 1000 - browse_date_ms)

        # The data_dir already contains the session, so we don't need to add it again
        save_dir = os.path.join(data_dir, str(browse_date_ms))
        
        # Ensure directory exists with proper error handling
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating save directory {save_dir}: {str(e)}")
            return None

        # Save frame as JPEG
        save_path = os.path.join(save_dir, str(browse_delta_ms))
        
        # Ensure subdirectory exists with proper error handling
        try:
            os.makedirs(save_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating save subdirectory {save_path}: {str(e)}")
            return None

        # Save the image with proper error handling
        image_path = os.path.join(save_path, os.path.basename(cam_name) + '.jpg')
        
        # Verify the frame is valid before saving
        if frame is None or frame.size == 0:
            print(f"Invalid frame data, skipping save")
            return None
            
        # Save the image
        cv2.imwrite(image_path, frame, encode_param)
        print(f"Saved image to: {image_path}")

        # Verify the file was actually created
        if not os.path.exists(image_path):
            print(f"Warning: Image file was not created at {image_path}")
            return None

        # Check for old data if save_days is specified and greater than 0
        if save_days is not None and save_days > 0:
            check_old_data(data_dir, save_days, time_posix)

    except Exception as e:
        error_msg = f"Error sending frame: {str(e)}"
        print(error_msg)
        return None

    return frame


def check_old_data(data_dir: str, save_days: int, current_time_posix: float):
    """
    Check for data older than save_days and print debug messages.
    
    Args:
        data_dir (str): The data directory to check
        save_days (int): Number of days to keep data
        current_time_posix (float): Current timestamp in POSIX format
    """
    if(save_days is None or save_days <= 0):     # fire-wall
        return
    
    try:
        # Calculate the cutoff date (current_day - save_days)
        current_date = datetime.fromtimestamp(current_time_posix).date()
        cutoff_date = current_date - timedelta(days=save_days)
        cutoff_date_ms = int(time.mktime(cutoff_date.timetuple()) * 1000)
        
        # Find all date directories in the data directory
        date_dirs = glob.glob(os.path.join(data_dir, "*"))
        #old_dates = []
        old_folders = []

        for date_dir in date_dirs:
            if not os.path.isdir(date_dir):     # skip non-directories  
                continue
                
            date_name = os.path.basename(date_dir)
            try:
                # Try to convert directory name to timestamp (milliseconds)
                date_timestamp_ms = int(date_name)
                
                # Check if this date is older than the cutoff
                if date_timestamp_ms < cutoff_date_ms:
                    # Convert back to readable date for debug message
                    old_date = datetime.fromtimestamp(date_timestamp_ms / 1000).date()
                    #old_dates.append(str(old_date))
                    old_folders.append(date_dir)
            except ValueError:
                # Skip directories that aren't numeric timestamps
                continue
        
        # Delete old folders with proper error handling
        if(old_folders):
            old_folders.sort()
            for old_folder in old_folders:
                basename = os.path.basename(old_folder)
                #print('old_folder: ' + old_folder + ', basename: ' + basename + ', data_dir: '+ data_dir +', contains: ' + str(old_folder.startswith(data_dir)))
                if(basename.isnumeric() and old_folder.startswith(data_dir)):   # double check that it's numeric and in the data_dir
                    print(f"DELETE old folder: {old_folder}")
                    try:
                        rmtree(old_folder)
                    except OSError as e:
                        print(f"Error deleting old folder {old_folder}: {str(e)}")
        
    except Exception as e:
        print(f"Error checking old data: {str(e)}")
