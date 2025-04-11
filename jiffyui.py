"""
jiffyui.py: UI components and logic for JiffyCam

This module contains functions for building and updating the Streamlit UI.
"""

from datetime import datetime, timedelta, time as datetime_time
import time
import os
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np  # Add this import for image creation
import cv2  # Add this import for image processing

from jiffyconfig import RESOLUTIONS   # Import necessary components from other modules
from jiffyget import jiffyget, get_locations

# globals within jiffyui.py
#autoplay_direction = "None"

# following just to avoid error when importing YOLO
import torch
torch.classes.__path__ = [] # add this line to manually set it to empty. 

# TODO:  fix double-click required and other Live toggle issues

# --- UI Helper Functions ---
def heartbeat():
    """Heartbeat to check if the UI is still running.""" 
    #global autoplay_direction
    #print(f"autoplay_direction: {st.session_state.autoplay_direction}")
    if(st.session_state.autoplay_direction == "forward"):
        on_next_button(False)
    elif(st.session_state.autoplay_direction == "reverse"):
        on_prev_button(False)   

def set_autoplay(direction):
    """Set autoplay direction."""
    #global autoplay_direction
    st.session_state.autoplay_direction = direction

def format_time_12h(hour, minute, second):
    """Format time in 12-hour format with AM/PM indicator."""
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12}:{minute:02d}:{second:02d} {period}"

# --- Callback Functions (must use st.session_state) ---

def on_device_alias_change():
    """Update cam_device when device alias is changed"""
    selected_alias = st.session_state.selected_device_alias
    if selected_alias in st.session_state.device_aliases:
        st.session_state.cam_device = st.session_state.device_aliases[selected_alias]
        st.session_state.session = selected_alias # Session matches alias
        st.session_state.oldest_timestamp = None # Force range recalculation
        st.session_state.newest_timestamp = None
        st.session_state.need_to_display_recent = True # Show most recent for new device

def toggle_rt_capture():
    """Toggle real-time capture on/off"""
    st.session_state.rt_capture = not st.session_state.rt_capture

    if st.session_state.rt_capture:
        st.session_state.in_playback_mode = False

        resolution = st.session_state.resolution

        if resolution in RESOLUTIONS:
            width, height = RESOLUTIONS[resolution]
            resolution_str = f"{width}x{height}"
        else:
            resolution_str = resolution # Assume format "WxH"
            if not isinstance(resolution_str, str) or 'x' not in resolution_str:
                resolution_str = "1920x1080" # Default

        st.session_state.browsing_saved_images = False
        st.session_state.date = datetime.now().date()

        # Call start_capture on the video_capture object stored in session state
        st.session_state.video_capture.start_capture(
            resolution_str=resolution_str,
            session=st.session_state.session,
            cam_device=st.session_state.cam_device,
            cam_name=st.session_state.cam_name,
            save_interval=st.session_state.save_interval,
            device_aliases=st.session_state.device_aliases,
            selected_device_alias=st.session_state.selected_device_alias
        )
    else:
        st.session_state.video_capture.stop_capture()
        st.session_state.in_playback_mode = True   

def on_date_change():
    """Handle date picker change."""
    set_autoplay("None")

    st.session_state.browsing_date = st.session_state.date
    st.session_state.in_playback_mode = True 

    # Fetch placeholders from session_state inside the update function
    update_image_display(direction="down")

def on_prev_button(stopAuto=True):
    """Handle previous image button click."""
    if(stopAuto):
        set_autoplay("None")
    st.session_state.in_playback_mode = True
    # Decrement time logic
    tweek_time('down')

    # Fetch placeholders from session_state inside the update function
    if(stopAuto):
        st.session_state.step_direction = "down"    # let rerun handle the update  
    else:
        update_image_display(direction="down")

def on_next_button(stopAuto=True):
    """Handle next image button click."""
    if(stopAuto):
        set_autoplay("None")
    st.session_state.in_playback_mode = True
    # Increment time logic
    tweek_time('up')

    # Fetch placeholders from session_state inside the update function
    if(stopAuto):
        st.session_state.step_direction = "up"      # let rerun handle the update
    else:
        update_image_display(direction="up")

def tweek_time(direction):
    """Tweek the time by 1 second in the given direction."""
    dt = datetime.combine(st.session_state.date, datetime_time(st.session_state.hour, st.session_state.minute, st.session_state.second))
    dt += timedelta(seconds=1) if direction == "up" else -timedelta(seconds=1)
    st.session_state.browsing_date = dt.date()
    st.session_state.hour = dt.hour
    st.session_state.minute = dt.minute
    st.session_state.second = dt.second

def toggle_live_pause():
    """Handle Live/Pause button click."""
    #set_autoplay("None")
    
    if st.session_state.in_playback_mode:
        # Go Live
        # Update browsing date/time to current (will be reflected in loop)
        current_time = datetime.now()
        st.session_state.browsing_date = current_time.date()
        st.session_state.in_playback_mode = False
        # Let the main loop update hour/min/sec/slider when live
    else:
        # Pause
        st.session_state.in_playback_mode = True

def on_pause_button():
    """Handle pause button click."""
    set_autoplay("None")
    st.session_state.in_playback_mode = True
    st.session_state.step_direction = "None"
    
def on_fast_reverse_button():
    """Handle fast reverse button click."""
    set_autoplay("reverse")
    tweek_time('down')
    # Set to playback mode like other buttons do
    st.session_state.in_playback_mode = True

def on_fast_forward_button():
    """Handle fast forward button click."""
    set_autoplay("forward")
    tweek_time('up')
    # Set to playback mode like other buttons do
    st.session_state.in_playback_mode = True

# --- UI Update Functions ---
def new_image_display(frame):
    """Display a new image based on the current date and time."""
    video_placeholder = st.session_state.video_placeholder
    video_placeholder.image(frame, channels="BGR", use_container_width=True)

    timearrow_placeholder = st.session_state.timearrow_placeholder
    ta_img = generate_timeline_arrow()
    timearrow_placeholder.image(ta_img, channels="RGB", use_container_width=True)
    
def update_image_display(direction=None):
    """Update the image display based on the current date and time."""
    #print(f"update_image_display: {direction}")

    # Fetch placeholders from session state
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    time_display = st.session_state.time_display

    # --- Get necessary state for jiffyget ---
    session = st.session_state.get('session', 'Default') # Safely get session
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData') # Get data_dir from config
    browse_date = st.session_state.browsing_date # Get the date being browsed
    # --- End state retrieval ---

    # Find the closest image using jiffyget directly
    time_posix = datetime.combine(browse_date, datetime.min.time()) + timedelta(hours=st.session_state.hour, minutes=st.session_state.minute, seconds=st.session_state.second)
    time_posix = float(time_posix.timestamp())

    closest_image, timestamp, eof = jiffyget(
        time_posix,
        st.session_state.cam_name,
        session,       # Pass session variable
        data_dir,      # Pass data_dir variable
        direction
    )
    if(eof):
        set_autoplay("None")

    success = False
    if closest_image is not None:
        try:
            # Store frame and actual timestamp
            st.session_state.last_frame = closest_image.copy()
            dts = datetime.fromtimestamp(timestamp/1000)
            st.session_state.actual_timestamp = dts

            # Update time state to match the found image
            st.session_state.hour = dts.hour
            st.session_state.minute = dts.minute
            st.session_state.second = dts.second
            st.session_state.browsing_date = dts.date() # Keep browsing date in sync with found image

            # Update UI placeholders
            new_image_display(closest_image)

            # Update time display text
            time_display.markdown(
                f'<div class="time-display">{format_time_12h(dts.hour, dts.minute, dts.second)}</div>',
                unsafe_allow_html=True
            )

            # Update status text
            status_placeholder.text(f"Viewing: {dts.strftime('%Y-%m-%d %H:%M:%S')}")
            st.session_state.status_message = f"Viewing: {dts.strftime('%Y-%m-%d %H:%M:%S')}" # Update internal status too
            success = True

        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            status_placeholder.error(f"Error displaying image: {str(e)}")
            st.session_state.status_message = f"Error displaying image: {str(e)}"
    
    if not success:
        set_autoplay("None")

        # No image found or error displaying
        status_placeholder.text(f"No image found near specified time")
        st.session_state.status_message = f"No image found near specified time"
        # Keep showing the last valid frame if one exists
        if st.session_state.last_frame is not None:
            #video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
            new_image_display(st.session_state.last_frame)
        else: # No previous frame, clear the placeholder
            video_placeholder.empty()

    return success

def display_most_recent_image():
    """Display the most recent image for the current camera."""

    # Use current time as starting point
    timestamp = datetime.now()
    st.session_state.hour = timestamp.hour
    st.session_state.minute = timestamp.minute
    st.session_state.second = timestamp.second
    st.session_state.browsing_date = timestamp.date()
    # st.session_state.date = timestamp.date() # REMOVED: Cannot modify widget state directly

    # Search backwards ("down") for the latest image before now
    update_image_display(direction="down")
    st.session_state.need_to_display_recent = False # Mark as done

# --- UI Building Functions ---
def build_sidebar(show_resolution_setting):
    """Create the sidebar UI elements and return placeholders."""
    with st.sidebar:
        st.title("JiffyCam")
        st.header("Settings")

        # Device Selection
        device_aliases = list(st.session_state.device_aliases.keys())
        st.selectbox("Device", options=device_aliases, key='selected_device_alias',
                     on_change=on_device_alias_change, help="Select camera device")

        # Resolution Selection (Optional)
        if show_resolution_setting:
            resolution_options = list(RESOLUTIONS.keys())
            if st.session_state.resolution not in RESOLUTIONS and st.session_state.resolution not in resolution_options:
                resolution_options.append(st.session_state.resolution)
            st.selectbox("Resolution", options=resolution_options, key="resolution",
                         help="Select camera resolution")

        # Save Interval
        st.number_input("Save Interval (s)", key='save_interval', min_value=0,
                        help="Seconds between saves (0=disable)")

        # Capture Button
        is_capturing = st.session_state.video_capture.is_capturing()
        button_text = "Stop Capture" if is_capturing else "Start Capture"
        button_type = "secondary" if is_capturing else "primary"
        st.markdown("---")
        st.button(button_text, key="rt_toggle", on_click=toggle_rt_capture,
                  help="Toggle real-time capture", type=button_type)
        if is_capturing: st.success("Capture Active")

        # Status Placeholders (created here, returned for main loop)
        st.header("Status")
        status_placeholder = st.empty()
        error_placeholder = st.empty()

    return status_placeholder, error_placeholder

def generate_timeline_image(width=1200, height=60):
    """Generate an image for the timeline bar based on available data."""
    #print(f"generate_timeline_image: {width}, {height}")
    session = st.session_state.get('session', 'Default')
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')

    # Construct base path safely
    browse_date_posix = int(time.mktime(st.session_state.date.timetuple()))
    timestamps = get_locations(st.session_state.cam_name, session, data_dir, browse_date_posix*1000)
    if(timestamps is None):
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Create a blank image (dark gray background)
    background_color = (51, 51, 51)  # Dark gray
    mark_color = (255, 0, 0)  # Red in BGR format (Blue = 0, Green = 0, Red = 255) 
    hour_marker_color = (180, 180, 180)  # Light gray for hour markers
    special_marker_color = (255, 255, 0)  # Yellow in BGR format
    text_color = (220, 220, 220)  # Brighter light gray for text
    
    # Add equal padding for top and bottom margins
    text_padding = 20  # Pixels below timeline for text
    top_margin = 0  # Match top margin to text padding (was 12)
    timeline_y_start = top_margin  # Timeline now starts below top margin
    timeline_y_end = timeline_y_start + height
    total_height = timeline_y_end + text_padding  # Total image height
    
    # Start with a blank transparent image (including space for labels and top margin)
    rounded_img = np.zeros((total_height, width, 3), dtype=np.uint8)
    
    # Add radius for rounded corner effect (small radius)
    radius = int(height/2)
    
    # Draw a rounded rectangle for the timeline (top portion only)
    cv2.rectangle(rounded_img, (radius, timeline_y_start), (width-radius, timeline_y_end), background_color, -1)
    cv2.rectangle(rounded_img, (0, timeline_y_start+radius), (width, timeline_y_end-radius), background_color, -1)
    
    # Draw the four corners
    cv2.circle(rounded_img, (radius, timeline_y_start+radius), radius, background_color, -1)
    cv2.circle(rounded_img, (width-radius, timeline_y_start+radius), radius, background_color, -1)
    cv2.circle(rounded_img, (radius, timeline_y_end-radius), radius, background_color, -1)
    cv2.circle(rounded_img, (width-radius, timeline_y_end-radius), radius, background_color, -1)
    
    timeline_img = rounded_img
    
    # Define label positions in advance
    label_positions = {
        0: "12am",
        6: "6am",
        12: "12pm",
        18: "6pm",
        24: "12am"
    }
    
    # Set font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.33, min(0.5, width / 2000))  # Scale text based on timeline width
    font_thickness = 1
    
    # Add hour markers and time labels (24 hours)
    for hour in range(25):  # 0 to 24 hours (include 24 for end of day)
        position = hour / 24.0  # Convert hour to percentage of day
        x_pos = int(position * width)
        
        # Adjust thickness and height based on marker type
        if hour % 6 == 0:  # More prominent markers at 6-hour intervals
            marker_height = int(height * 0.7)  # 70% of total height
            thickness = max(1, int(width/1200))
            alpha = 0.5  # 50% opacity
            
            # Get time label for this hour
            time_label = label_positions.get(hour, "")
                
            if time_label:
                # Calculate text size to center it
                text_size, baseline = cv2.getTextSize(time_label, font, font_scale, font_thickness)
                
                # Center text below marker
                text_x = x_pos - text_size[0] // 2
                # Position text below timeline with enough room for descenders
                text_y = timeline_y_end + baseline + 8  # Adjusted positioning to avoid clipping descenders
                
                # Handle edge cases
                if hour == 0:  # Leftmost label (12am)
                    text_x = max(text_x, 2)  # Keep a minimum of 2px from left edge
                elif hour == 24:  # Rightmost label (12am)
                    text_x = min(text_x, width - text_size[0] - 2)  # Keep from right edge
                
                # Draw a slightly darker rectangle behind the text
                # Make the background rectangle taller to accommodate descenders
                cv2.rectangle(timeline_img, 
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + baseline + 2),
                            (30, 30, 30), -1)
                
                # Draw the text
                cv2.putText(timeline_img, time_label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        else:
            marker_height = int(height * 0.4)  # 40% of total height
            thickness = max(1, int(width/1500))
            alpha = 0.3  # 30% opacity
            
        # Skip markers at the very edges but still draw text
        if x_pos < radius or x_pos > width - radius:
            continue
            
        # Draw line from bottom
        start_y = timeline_y_end - 1
        end_y = timeline_y_end - marker_height
        
        # Draw semi-transparent hour marker lines
        overlay = timeline_img.copy()
        cv2.line(overlay, (x_pos, start_y), (x_pos, end_y), hour_marker_color, thickness)
        cv2.addWeighted(overlay, alpha, timeline_img, 1-alpha, 0, timeline_img)

    # Add marks for timestamps
    if timestamps:
        for position in timestamps:
            # Calculate pixel position
            x_pos = int(position * width)
            # Draw a red vertical line with alpha blending
            cv_thickness = max(1, int(width/1000))  # Scale thickness with width
            alpha = 0.8  # 80% opacity (slightly more visible than before)
            
            # Draw semi-transparent line
            overlay = timeline_img.copy()
            cv2.line(overlay, (x_pos, timeline_y_start), (x_pos, timeline_y_end), mark_color, cv_thickness)
            cv2.addWeighted(overlay, alpha, timeline_img, 1-alpha, 0, timeline_img)

    return timeline_img

def generate_timeline_arrow(width=1200, height=24):
    # Add special timestamp marker for current time
    if hasattr(st.session_state, 'hour') and hasattr(st.session_state, 'minute') and hasattr(st.session_state, 'second'):
        # Calculate position as percentage of day
        current_seconds = st.session_state.hour * 3600 + st.session_state.minute * 60 + st.session_state.second
        day_percentage = current_seconds / (24 * 60 * 60)
        #print(f"day_percentage: {day_percentage}")
        x_pos = int(day_percentage * width)
        radius = int(height/2)
        timeline_y_start = height -1
        timeline_y_end = 1
        timeline_img = np.zeros((height, width, 3), dtype=np.uint8)
        special_marker_color = (255, 255, 0)  # Yellow in BGR format

        # Skip if too close to edges
        if x_pos >= radius and x_pos <= width - radius:
            # Draw special marker (thicker line with yellow color)
            cv_thickness = max(2, int(width/800))  # Thicker than regular markers
            
            # Draw semi-transparent triangle ABOVE timeline (pointing down)
            triangle_height = int(height * 0.75)  # Adjusted to 75% of top margin height
            triangle_width = int(height * 0.75)  # Slightly wider triangle
            
            # Triangle points (pointing down)
            bottom_point = (x_pos, timeline_y_start - 1)  # Just above timeline
            left_point = (x_pos - triangle_width//2, timeline_y_start - triangle_height)
            right_point = (x_pos + triangle_width//2, timeline_y_start - triangle_height)
            
            # Draw the triangle
            triangle_pts = np.array([bottom_point, left_point, right_point], np.int32)
            triangle_pts = triangle_pts.reshape((-1, 1, 2))
            overlay = timeline_img.copy()
            cv2.fillPoly(overlay, [triangle_pts], special_marker_color)
            cv2.addWeighted(overlay, 0.8, timeline_img, 0.2, 0, timeline_img)

    return timeline_img

def on_timeline_click(coords):
    """Handle clicks on the timeline image."""
    set_autoplay("None")
    if coords is None or 'x' not in coords:
        return
    
    # Calculate the time position based on click position
    x = coords['x']
    width = coords.get('width', 1)
    if width <= 0:
        return  # Avoid division by zero
    
    # Calculate the position as a percentage of the day
    day_percentage = x / width
    day_percentage = max(0, min(1, day_percentage))
    
    # Convert to hours, minutes, seconds
    total_seconds = int(day_percentage * 24 * 60 * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    # Update state
    st.session_state.hour = hours
    st.session_state.minute = minutes
    st.session_state.second = seconds
    st.session_state.in_playback_mode = True
    set_autoplay("none")
    

def build_main_area():
    """Create the main UI area elements and return placeholders."""
    #print("build_main_area")
    # Apply CSS for main area styling
    st.markdown("""
    <style>
    /* General layout */

    .block-container { padding-top: 0.7rem !important; padding-bottom: 0.7rem !important; }
    
    h1, h2, h3 { margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; padding: 0 !important; }
    hr { margin: 0 !important; padding: 0 !important; }
    /* Time Display */
    .time-display { font-size: 1.6rem; font-weight: 400; text-align: center; font-family: "Source Sans Pro", sans-serif; background: transparent; color: #333; border-radius: 5px; padding: 2px; margin: 2px 0; }
    @media (prefers-color-scheme: dark) { .time-display { color: #e0e0e0 !important; } }
    /* Date Picker */
    div[data-testid="stDateInput"] { margin: 10px 0 0 0 !important; padding: 6px 0 !important; display: flex !important; flex-direction: column !important; justify-content: center !important; }
    div[data-testid="stDateInput"] > div { margin-bottom: 0 !important; display: flex !important; align-items: center !important; height: 32px !important; }
    div[data-testid="stDateInput"] input { padding: 2px 0px !important; height: 32px !important; font-size: 14px !important; background-color: #2e2e2e !important; color: #fff !important; border: 1px solid #555 !important; border-radius: 5px !important; text-align: center !important; font-weight: bold !important; margin: 0 !important; }
    div[data-testid="stDateInput"] svg { fill: #fff !important; }

    /* Live button styling */
    button[data-testid="baseButton-primary"]:has(div:contains("Live")) {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Layout ---

    # Centered controls container
    st.markdown("<div style='max-width: 600px; margin: 5px auto 0 auto;'>", unsafe_allow_html=True)

    # Date Picker
    min_date, max_date = None, datetime.now().date()
    if st.session_state.oldest_timestamp: min_date = st.session_state.oldest_timestamp.date()
    if st.session_state.newest_timestamp: max_date = max(max_date, st.session_state.newest_timestamp.date())
    st.date_input("Date", key="date",
                   # Callback no longer needs args
                   on_change=on_date_change,
                   help="Select date", label_visibility="collapsed",
                   min_value=min_date, max_value=max_date)

    # Time controls row
    time_cols = st.columns([3, 1, 1, 1, 1, 1, 0.2, 1])
    with time_cols[0]: # Time Display Placeholder
        time_display = st.empty()
        # Set initial display value
        #time_display.markdown(f'<div class="time-display">{format_time_12h(st.session_state.hour, st.session_state.minute, st.session_state.second)}</div>', unsafe_allow_html=True)
    with time_cols[1]: # Fast Reverse Button
        st.button("⏪", key="fast_reverse_button", use_container_width=True, help="Fast Reverse",
                  on_click=on_fast_reverse_button)
    with time_cols[2]: # Prev Button
        st.button("◀️", key="prev_button", use_container_width=True, help="Previous",
                  # Callback no longer needs args
                  on_click=on_prev_button)
    with time_cols[3]: # Pause Button
        st.button("⏸️", key="pause_button", use_container_width=True, help="Pause",
                  on_click=on_pause_button)
    with time_cols[4]: # Next Button
        st.button("▶️", key="next_button", use_container_width=True, help="Next",
                  # Callback no longer needs args
                  on_click=on_next_button)
    with time_cols[5]: # Fast Forward Button
        st.button("⏩", key="fast_forward_button", use_container_width=True, help="Fast Forward",
                  on_click=on_fast_forward_button)
    with time_cols[6]: # Separator
        st.markdown('<div style="width:1px;background-color:#555;height:32px;margin:0 auto;"></div>', unsafe_allow_html=True)
    with time_cols[7]: # Live/Pause Button
        is_capturing = st.session_state.video_capture.is_capturing()
        in_playback = st.session_state.in_playback_mode
        
        # Always display "Live" text, but with different styles
        button_text = "Live"
        
        if in_playback:
            # Not in live mode - unfilled button
            button_help = "Switch to live view"
            button_type = "secondary"  # Gray (unfilled) button
        else:
            # In live mode - filled red button
            button_help = "Pause live view"
            button_type = "primary"    # Red (filled) button via CSS
        
        st.button(button_text, key="live_btn", use_container_width=True, help=button_help,
                  on_click=toggle_live_pause, disabled=not is_capturing, type=button_type)
    
    # Time Arrow above timeline
    timearrow_placeholder = st.empty()
    #timearrow_img = generate_timeline_arrow()
    timearrow_img = np.zeros((24, 1200, 3), dtype=np.uint8)
    timearrow_placeholder.image(timearrow_img, channels="RGB", use_container_width=True)

    # Generate timeline image with appropriate width and increased height (50% taller)
    timeline_img = generate_timeline_image(width=1200, height=60)  # Increased from 40 to 60
    
    # Store previous click coordinates to avoid duplicate processing
    prev_coords = st.session_state.get('prev_timeline_coords', None)
    
    # Display the clickable image
    clicked_coords = streamlit_image_coordinates(
        timeline_img, 
        key="timeline_bar",
        use_column_width=True
    )
    #st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle timeline click only if it's a new click (different from previous)
    if clicked_coords and 'x' in clicked_coords:
        # Convert to tuple for comparison (dictionaries aren't hashable)
        current_click = (clicked_coords.get('x'), clicked_coords.get('y'))
        previous_click = prev_coords if prev_coords else (-1, -1)
        
        if current_click != previous_click:
            # This is a new click
            on_timeline_click(clicked_coords)
            # Save current click to avoid reprocessing
            st.session_state.prev_timeline_coords = current_click

    st.markdown("</div>", unsafe_allow_html=True) # Close centered container

    # Create Video Placeholder *after* the controls container
    video_placeholder = st.empty() 
     # reduces flicker but causes double-jump on timeline click:
    #if st.session_state.last_frame is not None and not st.session_state.in_playback_mode:
    #    video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)

    # Return placeholders needed outside this build function (by callbacks and main loop)
    return video_placeholder, time_display, timearrow_placeholder

# --- Main UI Update Loop (runs in jiffycam.py) ---
# This function is moved from jiffycam.py
def run_ui_update_loop():
    """The main loop to update the UI based on capture state and interactions."""
    # Fetch placeholders from session state at the start of the loop
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    error_placeholder = st.session_state.error_placeholder
    time_display = st.session_state.time_display

    # --- Initial Image Display --- 
    if st.session_state.need_to_display_recent and not st.session_state.video_capture.is_capturing():
        display_most_recent_image() # Fetches placeholders from session_state
    elif st.session_state.last_frame is not None:
        #print(f"last_frame is not None: {st.session_state.step_direction}")
        update_image_display(st.session_state.step_direction)

    while True:
        heartbeat()

        # Check for errors first
        if st.session_state.video_capture.error_event.is_set():
            st.session_state.video_capture.stop_capture() # Ensure cleanup
            error_placeholder.error(st.session_state.video_capture.last_error)
            status_placeholder.text("Status: Error")
            st.session_state.status_message = "Status: Error"
            # Keep last frame visible on error if possible
            #if st.session_state.last_frame is not None:
            #      new_image_display(st.session_state.last_frame)
            break # Exit loop

        is_capturing = st.session_state.video_capture.is_capturing()
        current_status = st.session_state.get("status_message", "")
        new_status = current_status # Default to no change

        if is_capturing:
            if st.session_state.in_playback_mode:
                # Playback Mode (Capture Running): Drain queue, show paused frame
                while True:
                    frame_data = st.session_state.video_capture.get_frame()
                    if frame_data[0] is None: break

                # Ensure paused frame is displayed
                if st.session_state.last_frame is not None:
                    #new_image_display(st.session_state.last_frame)
                    ts = st.session_state.actual_timestamp
                    new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback Mode"
                else:
                    new_status = "Playback Mode (No Frame)"
                    video_placeholder.empty() # Clear if no frame to show

            else: # Live View Mode (Capture Running)
                frame, fps, width, height = st.session_state.video_capture.get_frame()
                if(False and st.session_state.video_capture.image_just_saved):
                    st.session_state.video_capture.image_just_saved = False
                    status_placeholder.text(st.session_state.video_capture.save_status)
                    st.session_state.last_frame = frame.copy()
                    st.rerun()          # rebuild timeline  (causese flicker and strange behavior)

                if frame is not None:
                    # Update time display and slider for live view
                    current_time = datetime.now()
                    time_display.markdown(f'<div class="time-display">{format_time_12h(current_time.hour, current_time.minute, current_time.second)}</div>', unsafe_allow_html=True)
                    st.session_state.hour = current_time.hour
                    st.session_state.minute = current_time.minute
                    st.session_state.second = current_time.second
                    # Store & display live frame
                    st.session_state.last_frame = frame.copy()
                    new_image_display(frame)

                    # Update internal stats
                    st.session_state.video_capture.current_fps = fps
                    st.session_state.video_capture.current_width = width
                    st.session_state.video_capture.current_height = height
                    
                    # Update status (Save msg or FPS)
                    if st.session_state.video_capture.image_just_saved:
                        new_status = st.session_state.video_capture.save_status
                    else:
                        new_status = f"Live View - FPS: {fps}"

        else: # Capture Stopped
            if st.session_state.in_playback_mode:
                # Playback Mode (Capture Stopped): Show paused frame
                if st.session_state.last_frame is not None:
                    #video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    #new_image_display(st.session_state.last_frame)

                    ts = st.session_state.actual_timestamp
                    new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback (Stopped)"
                    #st.session_state.last_frame = None  # mjm
                else:
                    new_status = "Playback (Stopped - No Frame)"
                    video_placeholder.empty()
            else:
                print("Odd state: capture stopped")     # this shouldnt happen if jiffycam.py inits in playback mode
                video_placeholder.info("Capture stopped. Select date/time or Start Capture.")
                new_status = "Status: Idle"

        # Update status placeholder only if message changed
        if True: #new_status != current_status:
            status_placeholder.text(new_status)
            st.session_state.status_message = new_status # Store new status

        if(st.session_state.video_capture.image_just_saved):
            st.session_state.video_capture.image_just_saved = False
            st.rerun()

        # Main loop delay
        time.sleep(0.01)

    print("exit run_ui_update_loop!")
