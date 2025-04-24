"""
jiffyclient.py: HTTP Client for JiffyCam

This module provides a client for connecting to the JiffyCam server via HTTP.
It handles fetching status, image frames, and managing connection.
"""

import time
import threading
import requests
import numpy as np
import cv2

class JiffyCamClient:
    """Client for connecting to the standalone jiffycapture.py HTTP server."""
    
    def __init__(self, server_url):
        """Initialize the client with the server URL."""
        if not server_url:
            raise ValueError("Server URL must be provided - no default port is available")
            
        self.set_server_url(server_url)
        self.connected = False
        self.last_error = None
        self.last_status = None
        self.last_frame = None
        self.last_frame_time = 0
        self.connection_check_time = 0
        self.status_check_time = 0
        self.error_event = threading.Event()  # Event for error handling
        self.is_capturing_flag = False
        
        # Don't automatically log or try to connect - wait for explicit connection request
        # print(f"Initialized HTTP client with server URL: {self.server_url}")
    
    def set_server_url(self, server_url):
        """Update the server URL.
        
        Args:
            server_url: New server URL
        """
        # Ensure server_url has protocol and doesn't end with slash
        if not server_url.startswith("http://") and not server_url.startswith("https://"):
            server_url = "http://" + server_url
        if server_url.endswith("/"):
            server_url = server_url[:-1]
            
        self.server_url = server_url
        # Reset connection state when changing server
        self.connected = False
        self.connection_check_time = 0
    
    def disconnect(self):
        """Disconnect from the server and clean up resources."""
        self.connected = False
        self.last_error = None
        self.is_capturing_flag = False
    
    def check_connection(self, force=False):
        """Check if the server is available."""
        # Only check every 5 seconds unless forced
        current_time = time.time()
        if not force and current_time - self.connection_check_time < 5:
            return self.connected
            
        self.connection_check_time = current_time
        
        try:
            response = requests.get(f"{self.server_url}/status", timeout=2)
            if response.status_code == 200:
                self.connected = True
                self.last_status = response.json()
                self.last_error = None
                self.status_check_time = current_time  # Update status check time
                self.error_event.clear()  # Clear any previous error
                return True
            else:
                self.connected = False
                self.last_error = f"Server responded with status code: {response.status_code}"
                self.error_event.set()  # Set error event
                return False
        except requests.RequestException as e:
            self.connected = False
            self.last_error = f"Connection error: {str(e)}"
            self.error_event.set()  # Set error event
            return False
    
    def is_connected(self):
        """Check if we're currently connected to the server."""
        if time.time() - self.connection_check_time > 10:  # Force check if it's been more than 10 seconds
            return self.check_connection(force=True)
        return self.connected
    
    def update_status_if_needed(self):
        """Update status information periodically without fetching a new frame."""
        current_time = time.time()
        # Only update status every 2 seconds to reduce server load
        if current_time - self.status_check_time > 2:
            self.get_status()
            return True
        return False
        
    def get_status(self):
        """Get the server status."""
        current_time = time.time()
        self.status_check_time = current_time
        print(f"Getting status from {self.server_url}/status")
        
        try:
            response = requests.get(f"{self.server_url}/status", timeout=2)
            if response.status_code == 200:
                self.last_status = response.json()
                self.connected = True  # Also update connection status
                self.connection_check_time = current_time  # Update connection time too
                self.last_error = None
                self.error_event.clear()  # Clear any previous error
                return self.last_status
            else:
                self.last_error = f"Server responded with status code: {response.status_code}"
                self.error_event.set()  # Set error event
                return None
        except requests.RequestException as e:
            self.last_error = f"Connection error: {str(e)}"
            self.connected = False
            self.error_event.set()  # Set error event
            return None
    
    def get_frame(self):
        """Get the latest frame from the server."""
        # Only fetch a new frame every 1/30 second (30 FPS max)
        current_time = time.time()
        if current_time - self.last_frame_time < 0.033:
            return self.last_frame, self.last_status.get('fps', 0) if self.last_status else 0, 0, 0
            
        if not self.connected and not self.check_connection(force=False):
            return None, 0, 0, 0
            
        try:
            # Add timestamp to URL to prevent caching
            response = requests.get(f"{self.server_url}/image?t={int(current_time*1000)}", timeout=2)
            if response.status_code == 200:
                # Convert the response content to an OpenCV image
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Update last frame and time
                self.last_frame = frame
                self.last_frame_time = current_time
                
                # Use last known status without making another request
                if self.last_status:
                    try:
                        width, height = map(int, self.last_status.get('resolution', '0x0').split('x'))
                        fps = self.last_status.get('fps', 0)
                        return frame, fps, width, height
                    except (ValueError, AttributeError):
                        return frame, 0, 0, 0
                return frame, 0, 0, 0
            else:
                self.last_error = f"Server responded with status code: {response.status_code}"
                self.error_event.set()  # Set error event
                return None, 0, 0, 0
        except requests.RequestException as e:
            self.last_error = f"Connection error: {str(e)}"
            self.connected = False
            self.error_event.set()  # Set error event
            return None, 0, 0, 0
    
    def get_last_frame(self):
        """Get the cached last frame."""
        return self.last_frame
    
    # --- VideoCapture compatibility methods ---
    
    def is_capturing(self):
        """Check if capturing is currently active."""
        return self.is_capturing_flag and self.connected
    
    def start_capture(self, **kwargs):
        """Start capture - mark as capturing and connect to server."""
        if self.check_connection(force=True):
            self.is_capturing_flag = True
            return True
        return False
        
    def stop_capture(self):
        """Stop capture - mark as not capturing."""
        self.is_capturing_flag = False
        
    def error_event_is_set(self):
        """Check if there's an error."""
        return self.error_event.is_set() 