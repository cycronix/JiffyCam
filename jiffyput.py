"""
jiffyput.py: Helper module for JiffyCam to handle frame processing and storage

This module provides functions to process and save video frames,
"""

import os
import cv2
import time
from datetime import datetime, timedelta

from jiffydetect import detect

prevframe = None
def jiffyput(cam_name, frame, time_posix: float, session, data_dir):
    """
    Process and save a video frame.

    Args:
        cam_name (str): The name of the camera
        frame: The video frame to process
        ftime (float): The frame timestamp
        session (str): The session name
        data_dir (str): The data directory

    Returns:
        The processed frame (which may be modified by detection)
    """

    global prevframe

    try:
        # Set JPEG quality to 95
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

        detect_mode = True
        if detect_mode:
            tryframe = detect(frame, prevframe)  # pass previous frame to reject no-motion detections
            if tryframe is None:       # ONLY save detections!  (to do: heartbeat saves)
                return None
                
            prevframe = frame
            frame = tryframe

        # Create save directory using session and timestamp
        save_dir = os.path.join(data_dir, session, os.path.dirname(cam_name))
        os.makedirs(save_dir, exist_ok=True)

        # Save frame as JPEG
        timestamp_ms = int(time_posix * 1000)
        browse_date = datetime.fromtimestamp(time_posix).date()
        browse_date_ms = int(time.mktime(browse_date.timetuple()) * 1000)
        browse_delta_ms = int(time_posix * 1000 - browse_date_ms)

        #save_path = os.path.join(save_dir, str(timestamp_ms))
        save_path = os.path.join(save_dir, str(browse_date_ms), str(browse_delta_ms))
        os.makedirs(save_path, exist_ok=True)

        # Save the image
        image_path = os.path.join(save_path, os.path.basename(cam_name) + '.jpg')
        cv2.imwrite(image_path, frame, encode_param)
        #print(f"save to image_path: {image_path}")

    except Exception as e:
        error_msg = f"Error sending frame: {str(e)}"
        print(error_msg)
        return None

    return frame
