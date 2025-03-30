"""
jiffyput.py: Helper module for JiffyCam to handle frame processing and storage

This module provides functions to process and save video frames,
"""

import cv2
import os
import time
from datetime import datetime
from jiffydetect import detect

def jiffyput(cam_name, frame, ftime, save_frame, session, config, state=None):
    """
    Process and save a video frame.
    
    Args:
        cam_name (str): The name of the camera
        frame: The video frame to process
        ftime (float): The frame timestamp
        save_frame (bool): Whether to save the frame
        session (str): The session name
        config (dict): Configuration dictionary with settings like data_dir
        state (dict, optional): Dictionary to track state variables like last_save_time
        
    Returns:
        The processed frame (which may be modified by detection)
    """
    try:
        # Set JPEG quality to 95
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        
        if save_frame:
            detect_mode = True
            if detect_mode:
                tryframe = detect(frame, None)
                if tryframe is None:       # ONLY save detections!  (to do: heartbeat saves)
                    return frame
                else:
                    frame = tryframe

            # Get the data directory from the config
            data_dir = config.get('data_dir', 'JiffyData')
            
            # Create save directory using session and timestamp
            save_dir = os.path.join(data_dir, session, os.path.dirname(cam_name))
            os.makedirs(save_dir, exist_ok=True)
            
            # Save frame as JPEG
            timestamp_ms = int(ftime * 1000)
            save_path = os.path.join(save_dir, str(timestamp_ms))
            os.makedirs(save_path, exist_ok=True)
            
            # Save the image
            image_path = os.path.join(save_path, os.path.basename(cam_name) + '.jpg')
            cv2.imwrite(image_path, frame, encode_param)
            
            # Update state if provided
            if state is not None:
                # Set flag when image is saved and track the time it was saved
                state['image_just_saved'] = True
                state['image_saved_time'] = time.time()
                
                # Update last save time
                state['last_save_time'] = time.time()
                
                # Format human-readable timestamp
                timestamp_readable = datetime.fromtimestamp(ftime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Update save status message with more detailed info
                #state['save_status'] = f"Frame saved at {timestamp_readable} in {os.path.basename(save_path)}"
                state['save_status'] = f"Frame saved: {timestamp_readable}"

    except Exception as e:
        error_msg = f"Error sending frame: {str(e)}"
        if state is not None:
            state['last_error'] = error_msg
        print(error_msg)
        return None

    return frame
