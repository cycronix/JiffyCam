"""
jiffycapture.py: Video capture functionality for JiffyCam

This module provides the core video capture functionality for the JiffyCam application.
It can be used as a module or run as a standalone script with an optional HTTP server.

Usage examples:
    python jiffycapture.py WebCam                # Use device alias 'WebCam'
    python jiffycapture.py --device WebCam       # Same as above
    python jiffycapture.py WebCam --runtime 60   # Run for 60 seconds
    python jiffycapture.py --interval 10         # Save frames every 10 seconds
"""

import time
import threading
import gc
import sys
import argparse
#import io
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from queue import Queue
from datetime import datetime
from typing import Optional, Tuple
import os

import cv2
#import numpy as np
#from streamlit_server_state import server_state, server_state_lock, no_rerun

from jiffyput import jiffyput
from jiffyconfig import JiffyConfig

#import jiffyglobals

# Global variable to store the VideoCapture instance for HTTP server access
global_capture_instance = None

class JiffyHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for JiffyCam that serves the last captured frame."""
    
    def _set_headers(self, content_type='text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')  # Enable CORS
        self.end_headers()
    
    def _send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # Enable CORS
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Override to suppress logging of HTTP requests."""
        pass
    
    def _handle_root(self):
        """Handle the root path request"""
        self._set_headers()
        html = """
        <html>
        <head>
            <title>JiffyCam Server</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; max-width: 1200px; margin: 0 auto; }
                h1 { color: #333; }
                .container { display: flex; flex-direction: column; }
                .image-container { margin-top: 20px; }
                img { max-width: 100%; border: 1px solid #ddd; }
                ul { padding-left: 20px; }
                li { margin-bottom: 8px; }
                .status { background-color: #f5f5f5; padding: 10px; border-radius: 4px; margin-top: 20px; }
                .refresh-controls { margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>JiffyCam HTTP Server</h1>
            <p>Server is running and capturing images.</p>
            <div class="container">
                <ul>
                    <li><a href="/image" target="_blank">Raw image endpoint</a></li>
                    <li><a href="/status" target="_blank">JSON status endpoint</a></li>
                </ul>
                
                <div class="refresh-controls">
                    Auto-refresh: 
                    <select id="refresh-rate" onchange="updateRefreshRate()">
                        <option value="1000">1 second</option>
                        <option value="2000">2 seconds</option>
                        <option value="5000" selected>5 seconds</option>
                        <option value="10000">10 seconds</option>
                        <option value="0">Off</option>
                    </select>
                    <button onclick="refreshImage()">Refresh Now</button>
                </div>
                
                <div class="image-container">
                    <h2>Live View</h2>
                    <img src="/image" id="stream" style="max-width: 100%;">
                </div>
                
                <div class="status" id="status-area">Loading status...</div>
            </div>
            
            <script>
                let refreshInterval;
                let refreshRate = 5000;
                
                // Function to refresh the image
                function refreshImage() {
                    document.getElementById('stream').src = '/image?' + new Date().getTime();
                }
                
                // Function to update the refresh rate
                function updateRefreshRate() {
                    const rate = document.getElementById('refresh-rate').value;
                    refreshRate = parseInt(rate);
                    
                    // Clear existing interval if any
                    if (refreshInterval) {
                        clearInterval(refreshInterval);
                    }
                    
                    // Set new interval if not disabled
                    if (refreshRate > 0) {
                        refreshInterval = setInterval(refreshImage, refreshRate);
                    }
                }
                
                // Function to refresh status
                function refreshStatus() {
                    fetch('/status')
                        .then(response => response.json())
                        .then(data => {
                            let statusHtml = '<h2>Capture Status</h2>';
                            statusHtml += `<p>Capturing: <strong>${data.capturing ? 'Active' : 'Stopped'}</strong></p>`;
                            statusHtml += `<p>FPS: <strong>${data.fps}</strong></p>`;
                            statusHtml += `<p>Resolution: <strong>${data.resolution}</strong></p>`;
                            statusHtml += `<p>Total Frames: <strong>${data.frame_count}</strong></p>`;
                            statusHtml += `<p>Last Save: <strong>${data.last_save_time || 'None'}</strong></p>`;
                            if (data.error) {
                                statusHtml += `<p style="color: red;">Error: ${data.error}</p>`;
                            }
                            document.getElementById('status-area').innerHTML = statusHtml;
                        })
                        .catch(err => {
                            document.getElementById('status-area').innerHTML = '<p style="color: red;">Error fetching status</p>';
                        });
                }
                
                // Initialize refreshes
                updateRefreshRate();
                refreshStatus();
                setInterval(refreshStatus, 2000);  // Update status every 2 seconds
            </script>
        </body>
        </html>
        """
        self.wfile.write(html.encode())
    
    def _handle_image(self):
        """Handle the image path request"""
        if global_capture_instance and global_capture_instance.last_frame is not None:
            frame = global_capture_instance.last_frame
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                self._set_headers('image/jpeg')
                if self.command != 'HEAD':  # Only send body for GET, not HEAD
                    self.wfile.write(buffer.tobytes())
            else:
                self._set_headers()
                if self.command != 'HEAD':
                    self.wfile.write(b"No image available")
        else:
            self._set_headers()
            if self.command != 'HEAD':
                self.wfile.write(b"No image available")
    
    def _handle_status(self):
        """Handle the status path request"""
        if global_capture_instance:
            frame_data = global_capture_instance.get_frame()
            _, fps, width, height = frame_data if frame_data[0] is not None else (None, 0, 0, 0)
            
            # Ensure we include the current session
            current_session = global_capture_instance.current_session
            
            status = {
                "capturing": global_capture_instance.is_capturing(),
                "frame_count": global_capture_instance.frame_count,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "last_save_time": datetime.fromtimestamp(global_capture_instance.last_save_time).strftime('%Y-%m-%d %H:%M:%S') if global_capture_instance.last_save_time > 0 else None,
                "error": global_capture_instance.last_error,
                "session": current_session,
                "active_session": current_session  # More explicit name for UI
            }
            #print(f"status: {status}")
            self._send_json_response(status)
        else:
            self._send_json_response({"error": "Capture not initialized"})
    
    def do_HEAD(self):
        """Handle HEAD requests"""
        if self.path == '/':
            self._set_headers()
        elif self.path.startswith('/image'):
            self._set_headers('image/jpeg')
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
        else:
            # Handle 404
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        #print(f"do_GET: {self.path}")
        if self.path == '/':
            self._handle_root()
        elif self.path.startswith('/image'):
            self._handle_image()
        elif self.path == '/status':
            self._handle_status()
        else:
            # Handle 404
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

def run_http_server(port=8080):
    """Run the HTTP server on the specified port."""
    server = HTTPServer(('', port), JiffyHTTPHandler)
    print(f"Starting HTTP server on port {port}")
    print(f"You can view the latest image at: http://localhost:{port}/image")
    print(f"You can check status at: http://localhost:{port}/status")
    print(f"You can view the web UI at: http://localhost:{port}/")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True  # Set as daemon so it will be stopped when the main program exits
    server_thread.start()
    return server

class VideoCapture:
    def __init__(self, config_file='jiffycam.yaml', session=None, data_dir='JiffyData', require_config_exists=False):
        """Initialize video capture functionality.
        
        Args:
            config_file (str): Base name of the configuration file (default: 'jiffycam.yaml')
            session (str, optional): Session name for config lookup. If provided, will look for
                                     config in data_dir/<session>/config_file
            data_dir (str): Base directory for data and configuration (default: 'JiffyData')
            require_config_exists (bool): If True, fail if config file doesn't exist
        
        Raises:
            FileNotFoundError: If require_config_exists is True and the config file doesn't exist
            ValueError: If the config file exists but is empty/invalid
        """
        self.stop_event = threading.Event()
        self.frame_count = 0
        
        # Now load the configuration
        self.config_manager = JiffyConfig(yaml_file=config_file, session=session, data_dir=data_dir, require_config_exists=require_config_exists)
        self.config = self.config_manager.config
        
        # Validate that we have a valid configuration
        if not self.config:
            raise ValueError(f"Empty or invalid configuration loaded")
        
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        self.last_error = None
        self.error_event = threading.Event()  # Add error event
        # Add status tracking variables
        self.current_fps = 0
        self.current_width = 0
        self.current_height = 0
        self.last_save_time = 0  # Track last save time
        self.save_status = ""  # Track save status message
        self.skip_first_save = True  # Flag to skip the first save
        #self.image_just_saved = False  # Flag to track when an image was just saved
        self.image_saved_time = 0  # Track when the image was saved
        self.last_frame = None
        self.current_session = session  # Initialize current_session with provided session value
        self.running = False

    def handle_error(self, error_msg: str):
        """Handle errors by setting error message and stopping capture"""
        self.last_error = error_msg
        self.error_event.set()
        self.stop_event.set()
        self.running = False

    def send_frame(self, cam_name: str, frame, ftime: float, session: str):
        """Send a frame to the data server."""
        try:
            # Process and save the frame
            result = jiffyput(cam_name, frame, ftime, session, self.config_manager.data_dir, self.config.get('weights', 'models/yolov8l.pt'))
            if result is not None:
                self.frame_count += 1
                self.last_save_time = ftime
                return True
            return False
        except Exception as e:
            error_msg = f"Error sending frame: {str(e)}"
            self.handle_error(error_msg)
            return False

    def capture_video(self, cam_device, cam_name, width, height, save_interval, session):
        """Initialize and run video capture loop - simplified direct version."""
        
        # Check for runtime limit in config
        runtime_limit = 0
        if 'runtime' in self.config:
            runtime_limit = int(self.config.get('runtime', 0))
        start_time = time.time()
        
        # Reset state variables
        self.stop_event.clear()
        self.error_event.clear()
        self.frame_count = 0
        self.last_error = None
        self.current_fps = 0
        self.current_width = 0
        self.current_height = 0
        self.last_save_time = 0
        self.save_status = ""
        self.skip_first_save = True
        #self.image_just_saved = False
        self.image_saved_time = 0
        self.current_session = session
        self.running = True
        
        # Handle numeric string sources by converting to integer
        source = cam_device
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        # For debugging
        print(f"Opening video capture with source: {source}")

        # Set the RTSP transport protocol to TCP or UDP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"  # or "rtsp_transport;udp"

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.handle_error(f"Error: Could not open video source: {source}")
            return

        # Configure camera
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if width != 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height != 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        save_frame = False  # Initialize as False to skip first frame
        last_fps_time = time.time()
        self.last_save_time = 0  # Initialize last save time
        frames_this_second = 0
        current_fps = 0
        consecutive_failures = 0  # Track consecutive read failures
        frame_count_since_start = 0  # Counter for frames since starting capture
        
        last_status_time = 0  # For periodic status updates

        try:
            while not self.stop_event.is_set():
                # Check runtime limit
                if runtime_limit > 0 and (time.time() - start_time) >= runtime_limit:
                    print(f"Runtime limit of {runtime_limit} seconds reached. Stopping capture.")
                    break
                    
                # Periodic status update (every 60 seconds)
                current_time = time.time()
                if current_time - last_status_time >= 60:
                    print(f"Capturing at {current_fps} FPS, {width}x{height}, Total frames: {self.frame_count}")
                    last_status_time = current_time
                
                # Read frame
                ret, frame = cap.read()
                if ret:
                    consecutive_failures = 0  # Reset failure counter on success
                    current_time = time.time()
                    frame_count_since_start += 1  # Increment frame counter
                    frame_copy = frame.copy()

                    # Regular interval saving for subsequent frames
                    if save_interval > 0 and (current_time - self.last_save_time) >= save_interval:
                        save_frame = True
                        self.last_save_time = current_time

                    # Update FPS calculation
                    frames_this_second += 1
                    if current_time - last_fps_time >= 1.0:
                        current_fps = frames_this_second
                        frames_this_second = 0
                        last_fps_time = current_time

                    if save_frame:
                        frame = self.send_frame(cam_name, frame, time.time(), session)
                    if self.error_event.is_set():  # Check if error occurred in send_frame
                        break
                    save_frame = False
                    self.frame_count += 1

                    # Update frame queue
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame, current_fps, width, height))
                    else:
                        try:
                            self.frame_queue.get_nowait()  # Remove old frame
                            self.frame_queue.put((frame, current_fps, width, height))
                        except:
                            pass

                    if frame is not None:
                        self.last_frame = frame_copy   # mjm framecopy in hopes of stop flashing imager   

                    time.sleep(0.01)  # Small delay to prevent overwhelming the system
                else:
                    #print(f"Video capture read failed")
                    consecutive_failures += 1
                    if consecutive_failures >= 10:  # Stop after 10 consecutive failures
                        self.handle_error("Video capture failed repeatedly - stopping capture")
                        break
                    self.last_error = "Video capture read failed"
                    time.sleep(1)
        finally:
            self.running = False
            cap.release()
            print("Camera released")

    def stop_capture(self):
        """Stop the capture."""
        print("stopping capture")
        self.stop_event.set()
        self.running = False
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass

        try:
            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def get_last_frame(self):
        """Get the last frame from the queue."""
        return self.last_frame

    def get_frame(self) -> Tuple:
        """Get the latest frame from the queue.
        
        Returns:
            Tuple of (frame, fps, width, height) or None if queue is empty
        """
        if not self.frame_queue.empty():
            try:
                return self.frame_queue.get_nowait()
            except:
                pass
        return None, 0, 0, 0

    def is_capturing(self) -> bool:
        """Check if capture is currently running.
        
        Returns:
            bool: True if capture is running, False otherwise
        """
        return self.running

def parse_args():
    """Parse command line arguments for standalone mode."""
    parser = argparse.ArgumentParser(
        description='JiffyCam Video Capture - a tool for capturing and storing video frames.',
        epilog='Example usage: python jiffycapture.py /path/to/data/folder     # Use data path containing jiffycam.yaml'
    )
    
    parser.add_argument('data_path', type=str, metavar='DATA_PATH',
                        help='Path to folder containing jiffycam.yaml and data folders')
    parser.add_argument('--config', type=str, default='jiffycam.yaml',
                        help='Base name of the YAML configuration file')
    parser.add_argument('--name', type=str,
                        help='Override camera name')
    parser.add_argument('--resolution', type=str,
                        help='Resolution in format WxH (e.g. 1920x1080)')
    parser.add_argument('--interval', type=int,
                        help='Interval between saved frames in seconds')
    parser.add_argument('--runtime', type=int, default=0,
                        help='Run for specified number of seconds then exit (0 for indefinite)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for the HTTP server (default: from config or 8080)')
    
    args = parser.parse_args()
    
    return args

def run_standalone():
    """Run JiffyCam in standalone mode."""
    global global_capture_instance  # Use the global variable for the HTTP server
    
    args = parse_args()
    
    # Get the data directory from the data_path argument
    data_dir = args.data_path
    
    # Initialize cam_device with default value
    cam_device = '0'
    
    # Try loading the config file for the specified session
    try:
        # Construct the config path without duplicating the session name
        config_path = os.path.join(data_dir, args.config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        session_config_manager = JiffyConfig(yaml_file=args.config, session=None, data_dir=data_dir, require_config_exists=True)
        session_config = session_config_manager.config

        # We managed to load this config, so proceed with this session
        print(f"Loaded configuration from {config_path}")
        
        # Get the session name from the data directory
        session = os.path.basename(data_dir)
        
        # Determine device to use based on the session name
        #if session in device_aliases:
        #    cam_device = device_aliases[session]
        #    print(f"Using device '{cam_device}' for session '{session}'")
        #else:
        # If the session name isn't found in device_aliases, use the default device
        #cam_device = device_aliases.get('Default', '0')
        #print(f"Using default device '{cam_device}' for session '{session}'")

        capture = VideoCapture(config_file=args.config, session=None, data_dir=data_dir, require_config_exists=True)
        global_capture_instance = capture
        config = capture.config
        
        # Ensure device_aliases are set in the config
        #if 'device_aliases' not in config or not config['device_aliases']:
        #    config['device_aliases'] = device_aliases
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure the data path contains a valid jiffycam.yaml configuration file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        print("Please ensure the configuration file is valid.")
        sys.exit(1)
    
    # Common setup code
    cam_name = args.name or config.get('cam_name', 'cam0')
    cam_device = session_config.get('cam_device', '0')
    print(f"cam_device: {cam_device}")

    resolution_str = args.resolution or config.get('resolution', '1920x1080')
    try:
        width, height = map(int, resolution_str.split('x'))
    except (ValueError, AttributeError, IndexError):
        width, height = 1920, 1080
        print(f"Warning: Invalid resolution format '{resolution_str}', using default 1920x1080")
    
    save_interval = args.interval if args.interval is not None else int(config.get('save_interval', 60))
    
    # Set runtime limit if specified
    if args.runtime > 0:
        config['runtime'] = args.runtime
        capture.config = config
    
    print(f"Starting capture with:")
    print(f"  Data Path: {data_dir}")
    print(f"  Session: {session}")
    print(f"  Camera Device: {cam_device}")
    print(f"  Camera Name: {cam_name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Save Interval: {save_interval} seconds")
    print(f"  Config File: {capture.config_manager.yaml_file}")
    print(f"  Runtime: {args.runtime if args.runtime > 0 else 'Indefinite'} seconds")
    
    # Start HTTP server
    port = args.port
    if 'dataserver_port' in config:
        port = int(config['dataserver_port'])  # Use port from config if available
    
    http_server = run_http_server(port)
    print(f"HTTP server started on port {port}")
    
    # Start capture in the main thread
    start_time = time.time()
    try:
        # Run the capture directly (not in a thread)
        capture.capture_video(
            cam_device=cam_device,
            cam_name=cam_name,
            width=width,
            height=height,
            save_interval=save_interval,
            session=session
        )
    except KeyboardInterrupt:
        print("\nCapture stopped by user")
    finally:
        # Clean up
        capture.stop_capture()
        print("Capture stopped")
        
        # Stop HTTP server
        http_server.shutdown()
        print("HTTP server stopped")
        
        # Clear global reference
        global_capture_instance = None

if __name__ == "__main__":
    run_standalone() 