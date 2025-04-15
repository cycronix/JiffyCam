"""
jiffycapture.py: Video capture functionality for JiffyCam

This module provides the core video capture functionality for the JiffyCam application.
It can be used as a module or run as a standalone script with an optional HTTP server.
"""

import time
import threading
import gc
import sys
import argparse
import io
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from queue import Queue
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
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
            
            status = {
                "capturing": global_capture_instance.is_capturing(),
                "frame_count": global_capture_instance.frame_count,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "last_save_time": datetime.fromtimestamp(global_capture_instance.last_save_time).strftime('%Y-%m-%d %H:%M:%S') if global_capture_instance.last_save_time > 0 else None,
                "error": global_capture_instance.last_error,
                "session": global_capture_instance.current_session
            }
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
    def __init__(self, config_file='jiffycam.yaml'):
        #print("Initializing VideoCapture")

        """Initialize video capture functionality."""
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.capture_thread: Optional[threading.Thread] = None
        self.config_manager = JiffyConfig(yaml_file=config_file)
        self.config = self.config_manager.config
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
        self.image_just_saved = False  # Flag to track when an image was just saved
        self.image_saved_time = 0  # Track when the image was saved
        self.last_frame = None
        self.current_session = None # Track the currently active session name
    def handle_error(self, error_msg: str):
        """Handle errors by setting error message and stopping capture"""
        self.last_error = error_msg
        self.error_event.set()
        self.stop_event.set()
        #with server_state_lock["is_capturing"]:
        #    server_state.is_capturing = False

    def send_frame(self, cam_name: str, frame, ftime: float, session: str):
        """Send frame to server and optionally save to disk."""
        try:
            # Call jiffyput function
            result = jiffyput(cam_name, frame, ftime, session, self.config.get('data_dir', 'JiffyData'))
            if result is None:
                return None
        
            # Update class attributes
            self.image_just_saved = True
            self.image_saved_time = self.last_save_time = time.time()
            self.save_status = f"Frame saved: {datetime.fromtimestamp(ftime).strftime('%Y-%m-%d %H:%M:%S')}"
                
        except Exception as e:
            self.handle_error(f"Error sending frame: {str(e)}")
            return None

        return result

    def capture_video_loop(self, source, cam_name: str, cam_width: int, cam_height: int, save_interval: int, session: str):
        """Initialize and run video capture loop."""
        
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.handle_error(f"Error: Could not open video source: {source}")
            return

        # Configure camera
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cam_width != 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        if cam_height != 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

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

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                consecutive_failures = 0  # Reset failure counter on success
                current_time = time.time()
                frame_count_since_start += 1  # Increment frame counter

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
                    self.last_frame = frame
                    #print(f"last_frame: {self.last_frame is not None}")

                time.sleep(0.01)  # Small delay to prevent overwhelming the system
            else:
                consecutive_failures += 1
                if consecutive_failures >= 10:  # Stop after 10 consecutive failures
                    self.handle_error("Video capture failed repeatedly - stopping capture")
                    break
                self.last_error = "Video capture read failed"
                time.sleep(1)

        cap.release()

    def start_capture(self, resolution_str=None, session=None, cam_device=None, cam_name=None, 
                      save_interval=None, device_aliases=None, selected_device_alias=None):
        """Start video capture with the given parameters.
        
        Args:
            resolution_str: Resolution string in format "widthxheight" or a key from RESOLUTIONS
            session: Session name
            cam_device: Camera device identifier
            cam_name: Camera name
            save_interval: Interval between saved frames
            device_aliases: Dictionary of device aliases
            selected_device_alias: Currently selected device alias
        """
        # Set up configuration with provided or default values
        if cam_device is None:
            cam_device = self.config.get('cam_device', '0')
        if device_aliases is None:
            device_aliases = self.config.get('device_aliases', {'USB0': '0', 'USB1': '1', 'Default': '0'})
        if session is None:
            # Use selected_device_alias if provided, otherwise try to find camera alias from cam_device
            if selected_device_alias:
                session = selected_device_alias
            else:
                # Find alias for the camera device (if exists)
                session = 'Default'  # fallback
                for alias, device in device_aliases.items():
                    if device == cam_device:
                        session = alias
                        break
        if cam_name is None:
            cam_name = self.config.get('cam_name', 'cam0')
        if save_interval is None:
            save_interval = int(self.config.get('save_interval', 60))
        
        # Process resolution
        if resolution_str is None:
            resolution_str = self.config.get('resolution', '1920x1080')
            
        # Create configuration dictionary
        config = {
            'cam_device': selected_device_alias or cam_device,
            'cam_name': cam_name,
            'save_interval': save_interval,
            'device_aliases': device_aliases
        }
        
        # Preserve data_dir if it exists in the current config
        if hasattr(self, 'config') and isinstance(self.config, dict) and 'data_dir' in self.config:
            config['data_dir'] = self.config['data_dir']
        
        # Add resolution to config
        config['resolution'] = resolution_str
        
        # Save configuration
        self.config = config
        self.config_manager.save_config(self.config)
        
        # Parse resolution string for capture thread
        try:
            width, height = map(int, resolution_str.split('x'))
        except (ValueError, AttributeError, IndexError):
            # Default to 1080p if parsing fails
            width, height = 1920, 1080
        
        # Start capture thread
        self.start_capture_thread(
            cam_device=cam_device,
            width=width,
            height=height,
            session=session,
            cam_name=cam_name,
            save_interval=save_interval
        )

    def stop_capture(self):
        """Stop the capture thread."""
        print("stopping capture")
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None
        self.current_session = None # Clear current session on stop
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass

        try:
            # Release PyTorch resources explicitly
            if hasattr(self, 'model') and self.model is not None:
                self.model = None
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Warning during cleanup: {e}")
            # Continue shutdown despite errors
        
       # with server_state_lock["is_capturing"]:
       #     server_state.is_capturing = False

    def start_capture_thread(self, cam_device, width, height, session, cam_name, save_interval):
        """Start video capture in a separate thread."""
        if self.capture_thread and self.capture_thread.is_alive():
            return

        self.stop_event.clear()
        self.error_event.clear()  # Clear error event
        self.frame_count = 0
        self.last_error = None
        self.current_fps = 0
        self.current_width = 0
        self.current_height = 0
        self.last_save_time = 0
        self.save_status = ""
        self.skip_first_save = True  # Reset the skip flag when starting capture
        self.image_just_saved = False  # Reset the image_just_saved flag
        self.image_saved_time = 0  # Reset the image_saved_time variable
        self.current_session = session # Store the active session name
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self.capture_video_loop,
            args=(
                cam_device,
                cam_name,
                width,
                height,
                int(save_interval),
                session
            ),
            daemon=True
        )
        self.capture_thread.start()
        #with server_state_lock["is_capturing"]:
        #    server_state.is_capturing = True

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
            bool: True if capture thread is running, False otherwise
        """
        return self.capture_thread and self.capture_thread.is_alive()
        #with server_state_lock["is_capturing"]:
        #    return server_state.is_capturing 

def parse_args():
    """Parse command line arguments for standalone mode."""
    parser = argparse.ArgumentParser(description='JiffyCam Video Capture')
    parser.add_argument('--config', type=str, default='jiffycam.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--device', type=str,
                        help='Override camera device (use device alias or direct ID/path)')
    parser.add_argument('--name', type=str,
                        help='Override camera name')
    parser.add_argument('--session', type=str, default='Default',
                        help='Session name for captured frames')
    parser.add_argument('--resolution', type=str,
                        help='Resolution in format WxH (e.g. 1920x1080)')
    parser.add_argument('--interval', type=int,
                        help='Interval between saved frames in seconds')
    parser.add_argument('--data-dir', type=str,
                        help='Directory to save captured frames')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available camera devices and exit')
    parser.add_argument('--runtime', type=int, default=0,
                        help='Run for specified number of seconds then exit (0 for indefinite)')
    parser.add_argument('--http', action='store_true',
                        help='Start an HTTP server to serve the latest frame')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for the HTTP server (default: from config or 8080)')
    
    return parser.parse_args()

def list_available_cameras():
    """List available camera devices."""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print("Available camera devices:")
    for idx in available_cameras:
        print(f"  {idx}")
    print("Note: Network cameras require full URLs (e.g. rtsp://...)")

def run_standalone():
    """Run JiffyCam in standalone mode."""
    global global_capture_instance  # Use the global variable for the HTTP server
    
    args = parse_args()
    
    if args.list_devices:
        list_available_cameras()
        return
    
    # Initialize video capture with config file
    capture = VideoCapture(config_file=args.config)
    global_capture_instance = capture  # Set the global instance for HTTP access
    
    # Override config with command line arguments
    config = capture.config
    
    if args.device:
        # Check if the device is an alias in the config
        if args.device in config.get('device_aliases', {}):
            cam_device = config['device_aliases'][args.device]
            # Use device alias as session if session not specified
            if args.session == 'Default':
                args.session = args.device
        else:
            cam_device = args.device
    else:
        # Use device from config, resolving alias if needed
        cam_device_cfg = config.get('cam_device', '0')
        if cam_device_cfg in config.get('device_aliases', {}):
            cam_device = config['device_aliases'][cam_device_cfg]
            # Use device alias as session if session not specified
            if args.session == 'Default':
                for alias, device in config.get('device_aliases', {}).items():
                    if device == cam_device:
                        args.session = alias
                        break
        else:
            cam_device = cam_device_cfg
    
    cam_name = args.name or config.get('cam_name', 'cam0')
    session = args.session
    
    resolution_str = args.resolution or config.get('resolution', '1920x1080')
    try:
        width, height = map(int, resolution_str.split('x'))
    except (ValueError, AttributeError, IndexError):
        width, height = 1920, 1080
        print(f"Warning: Invalid resolution format '{resolution_str}', using default 1920x1080")
    
    save_interval = args.interval if args.interval is not None else int(config.get('save_interval', 60))
    
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    print(f"Starting capture with:")
    print(f"  Camera: {cam_device} (alias: {args.device if args.device else config.get('cam_device', 'Default')})")
    print(f"  Camera Name: {cam_name}")
    print(f"  Session: {session}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Save Interval: {save_interval} seconds")
    print(f"  Data Directory: {config.get('data_dir', 'JiffyData')}")
    print(f"  Runtime: {args.runtime if args.runtime > 0 else 'Indefinite'} seconds")
    
    # Start HTTP server if requested
    http_server = None
    if True or args.http:                           # mjm always start http server
        # Use port from arguments or from config if available
        port = args.port if args.port != 8080 else int(config.get('dataserver_port', 8080))
        http_server = run_http_server(port)
        print(f"HTTP server started on port {port}")
    
    # Start capture
    capture.start_capture_thread(
        cam_device=cam_device,
        width=width,
        height=height,
        session=session,
        cam_name=cam_name,
        save_interval=save_interval
    )
    
    try:
        start_time = time.time()
        
        while capture.is_capturing():
            # Check if we should exit due to runtime limit
            if args.runtime > 0 and time.time() - start_time >= args.runtime:
                print(f"Runtime limit of {args.runtime} seconds reached. Stopping capture.")
                break
                
            # Print status every 60 seconds
            if int(time.time()) % 60 == 0:
                frame_data = capture.get_frame()
                if frame_data[0] is not None:
                    _, fps, width, height = frame_data
                    print(f"Capturing at {fps} FPS, {width}x{height}, Total frames: {capture.frame_count}")
                
            # Check for errors
            if capture.error_event.is_set():
                print(f"Error: {capture.last_error}")
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nCapture stopped by user")
    finally:
        # Clean up
        capture.stop_capture()
        print("Capture stopped")
        
        # Stop HTTP server if it was started
        if http_server:
            print("Stopping HTTP server...")
            http_server.shutdown()
        
        # Clear global reference
        global_capture_instance = None

if __name__ == "__main__":
    run_standalone() 