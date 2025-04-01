"""
jiffyput.py: Helper module for JiffyCam to handle frame processing and storage

This module provides functions to process and save video frames,
"""

import os
import cv2

from jiffydetect import detect

prevframe = None
def jiffyput(cam_name, frame, ftime, session, data_dir):
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
        timestamp_ms = int(ftime * 1000)
        save_path = os.path.join(save_dir, str(timestamp_ms))
        os.makedirs(save_path, exist_ok=True)

        # Save the image
        image_path = os.path.join(save_path, os.path.basename(cam_name) + '.jpg')
        cv2.imwrite(image_path, frame, encode_param)

    except Exception as e:
        error_msg = f"Error sending frame: {str(e)}"
        print(error_msg)
        return None

    return frame
