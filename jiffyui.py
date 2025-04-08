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
autoplay_direction = "None"

# --- UI Helper Functions ---
def heartbeat():
    """Heartbeat to check if the UI is still running.""" 
    global autoplay_direction
    #print(f"autoplay_direction: {autoplay_direction}")
    if(autoplay_direction == "forward"):
        on_next_button(False)
    elif(autoplay_direction == "reverse"):
        on_prev_button(False)    

def set_autoplay(direction):
    """Set autoplay direction."""
    global autoplay_direction
    autoplay_direction = direction

def sync_time_slider():
    """Sync the time slider to the current time.""" 
    return
 
    if(autoplay_direction != "None"):
          return
    
    try:    #some async issues with time slider
        #print(f"sync_time_slider: {st.session_state.hour}, {st.session_state.minute}, {st.session_state.second}")
        st.session_state.time_slider = datetime_time(
            st.session_state.hour,
            st.session_state.minute,
            st.session_state.second
        )  
        st.session_state.date = st.session_state.browsing_date
    except Exception as e:
        #print(f"Error syncing time slider: {e}")
        pass

def rt_time_slider():
    """Update the time slider to the current time."""
    st.session_state.time_slider = datetime_time(datetime.now().hour, datetime.now().minute, datetime.now().second) # Reset slider to current time

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
        rt_time_slider()

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
        #st.session_state.time_slider = datetime_time(0, 0, 0) # Reset slider to midnight

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

    # Fetch placeholders from session_state inside the update function
    #update_image_display(direction="down")

def on_date_change():
    """Handle date picker change."""
    set_autoplay("None")

    st.session_state.browsing_date = st.session_state.date
    st.session_state.in_playback_mode = True 

    # Sync slider to current time state
    #sync_time_slider()
 #   toggle_live_pause()

    # Fetch placeholders from session_state inside the update function
    update_image_display(direction="down")

def on_prev_button(stopAuto=True):
    """Handle previous image button click."""
    if(stopAuto):
        set_autoplay("None")
    st.session_state.in_playback_mode = True
    st.session_state.slider_currently_being_dragged = False

    # Decrement time logic
    dt = datetime.combine(st.session_state.date, datetime_time(st.session_state.hour, st.session_state.minute, st.session_state.second))
    dt -= timedelta(seconds=1)
    st.session_state.browsing_date = dt.date()
    st.session_state.hour = dt.hour
    st.session_state.minute = dt.minute
    st.session_state.second = dt.second

    # Fetch placeholders from session_state inside the update function
    update_image_display(direction="down")
    sync_time_slider()

def on_next_button(stopAuto=True):
    """Handle next image button click."""
    if(stopAuto):
        set_autoplay("None")
    st.session_state.in_playback_mode = True
    st.session_state.slider_currently_being_dragged = False

    # Increment time logic
    dt = datetime.combine(st.session_state.date, datetime_time(st.session_state.hour, st.session_state.minute, st.session_state.second))
    dt += timedelta(seconds=1)
    st.session_state.browsing_date = dt.date()
    st.session_state.hour = dt.hour
    st.session_state.minute = dt.minute
    st.session_state.second = dt.second

    # Fetch placeholders from session_state inside the update function
    update_image_display(direction="up")
    sync_time_slider()

def toggle_live_pause():
    """Handle Live/Pause button click."""    
    if st.session_state.in_playback_mode:
        # Go Live
        set_autoplay("None")
        st.session_state.live_button_clicked = True
        # Update browsing date/time to current (will be reflected in loop)
        current_time = datetime.now()
        st.session_state.browsing_date = current_time.date()
        rt_time_slider()

        st.session_state.in_playback_mode = False
        # Let the main loop update hour/min/sec/slider when live
    else:
        # Pause
        st.session_state.live_button_clicked = False

        # Store the timestamp of the *currently displayed* frame if available
        # If last_frame exists, its timestamp should be in actual_timestamp
        # If not, store current time as a fallback?
        if st.session_state.last_frame is None:
            st.session_state.actual_timestamp = datetime.now() # Fallback
        # No need to explicitly set hour/min/sec here, they reflect the paused frame
        st.session_state.in_playback_mode = True

def on_pause_button():
    """Handle pause button click."""
    set_autoplay("None")
    st.session_state.in_playback_mode = True
    
def on_time_slider_change():
    """Handle time slider change."""
    set_autoplay("None")

    if(st.session_state.live_button_clicked):
        st.session_state.live_button_clicked = False
        return
    
    time_obj = st.session_state.time_slider
    hours, minutes, seconds = time_obj.hour, time_obj.minute, time_obj.second

    # Check if time actually changed
    if (hours != st.session_state.hour or
        minutes != st.session_state.minute or
        seconds != st.session_state.second):

        st.session_state.hour = hours
        st.session_state.minute = minutes
        st.session_state.second = seconds
        st.session_state.slider_currently_being_dragged = True

        # Fetch placeholders from session_state inside the update function
        update_image_display(direction="closest")
        st.session_state.slider_currently_being_dragged = False

        st.session_state.in_playback_mode = True

def on_fast_reverse_button():
    """Handle fast reverse button click."""
    set_autoplay("reverse")

def on_fast_forward_button():
    """Handle fast forward button click."""
    set_autoplay("forward")

# --- UI Update Functions ---
def new_image_display(frame):
    """Display a new image based on the current date and time."""
    video_placeholder = st.session_state.video_placeholder
    video_placeholder.image(frame, channels="BGR", use_container_width=True)
    #video_placeholder.image(streamlit_image_coordinates(frame, key="image_display"), channels="BGR", use_container_width=True)
    #print(f"video_placeholder.coords: {video_placeholder.coords}")

def update_image_display(direction=None):
    """Update the image display based on the current date and time."""
    # Fetch placeholders from session state
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    time_display = st.session_state.time_display

    #st.session_state.in_playback_mode = True

    # --- Get necessary state for jiffyget ---
    session = st.session_state.get('session', 'Default') # Safely get session
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData') # Get data_dir from config
    browse_date = st.session_state.browsing_date # Get the date being browsed
    # --- End state retrieval ---

    # Find the closest image using jiffyget directly
    time_posix = datetime.combine(browse_date, datetime.min.time()) + timedelta(hours=st.session_state.hour, minutes=st.session_state.minute, seconds=st.session_state.second)
    time_posix = float(time_posix.timestamp())

    closest_image, timestamp = jiffyget(
        time_posix,
        st.session_state.cam_name,
        session,       # Pass session variable
        data_dir,      # Pass data_dir variable
        direction
    )

    success = False
    if closest_image is not None:
        try:
            # Store frame and actual timestamp
            st.session_state.last_frame = closest_image.copy()
            dts = datetime.fromtimestamp(timestamp/1000)
            st.session_state.actual_timestamp = dts

            # Update UI placeholders
            #video_placeholder.image(frame, channels="BGR", use_container_width=True)
            new_image_display(closest_image)

            # Update time state to match the found image
            st.session_state.hour = dts.hour
            st.session_state.minute = dts.minute
            st.session_state.second = dts.second
            st.session_state.browsing_date = dts.date() # Keep browsing date in sync with found image

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

    # Ensure playback mode remains true
    #st.session_state.in_playback_mode = True
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
    #sync_time_slider()

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

def generate_timeline_image(width=800, height=32):
    """Generate an image for the timeline bar based on available data."""
    session = st.session_state.get('session', 'Default')
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')

    # Construct base path safely
    browse_date_posix = int(time.mktime(st.session_state.date.timetuple()))
    timestamps = get_locations(st.session_state.cam_name, session, data_dir, browse_date_posix*1000)

    # Create a blank image (dark gray background)
    background_color = (51, 51, 51)  # Dark gray
    mark_color = (255, 0, 0)  # Red in BGR format (Blue = 0, Green = 0, Red = 255) 
    hour_marker_color = (180, 180, 180)  # Light gray for hour markers
    special_marker_color = (255, 255, 0)  # Yellow in BGR format
    text_color = (220, 220, 220)  # Brighter light gray for text
    
    # Add padding for text labels - increase to accommodate descenders
    text_padding = 20  # Pixels below timeline for text (increased from 15)
    label_height = height + text_padding
    
    # Start with a blank transparent image (including space for labels)
    rounded_img = np.zeros((label_height, width, 3), dtype=np.uint8)
    
    # Add radius for rounded corner effect (small radius)
    radius = int(height/2)
    
    # Draw a rounded rectangle for the timeline (top portion only)
    cv2.rectangle(rounded_img, (radius, 0), (width-radius, height), background_color, -1)
    cv2.rectangle(rounded_img, (0, radius), (width, height-radius), background_color, -1)
    
    # Draw the four corners
    cv2.circle(rounded_img, (radius, radius), radius, background_color, -1)
    cv2.circle(rounded_img, (width-radius, radius), radius, background_color, -1)
    cv2.circle(rounded_img, (radius, height-radius), radius, background_color, -1)
    cv2.circle(rounded_img, (width-radius, height-radius), radius, background_color, -1)
    
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
                text_y = height + baseline + 8  # Adjusted positioning to avoid clipping descenders
                
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
        start_y = height - 1
        end_y = height - marker_height
        
        # Draw semi-transparent hour marker lines
        overlay = timeline_img.copy()
        cv2.line(overlay, (x_pos, start_y), (x_pos, end_y), hour_marker_color, thickness)
        cv2.addWeighted(overlay, alpha, timeline_img, 1-alpha, 0, timeline_img)

    # Add special timestamp marker for current time
    if hasattr(st.session_state, 'hour') and hasattr(st.session_state, 'minute') and hasattr(st.session_state, 'second'):
        # Calculate position as percentage of day
        current_seconds = st.session_state.hour * 3600 + st.session_state.minute * 60 + st.session_state.second
        day_percentage = current_seconds / (24 * 60 * 60)
        x_pos = int(day_percentage * width)
        
        # Skip if too close to edges
        if x_pos >= radius and x_pos <= width - radius:
            # Draw special marker (thicker line with yellow color)
            cv_thickness = max(2, int(width/800))  # Thicker than regular markers
            
            # Draw semi-transparent triangle at top of timeline
            triangle_height = int(height * 0.6)
            triangle_width = int(height * 0.4)
            
            # Triangle points
            top_point = (x_pos, 0)
            left_point = (x_pos - triangle_width//2, triangle_height)
            right_point = (x_pos + triangle_width//2, triangle_height)
            
            # Draw the triangle
            triangle_pts = np.array([top_point, left_point, right_point], np.int32)
            triangle_pts = triangle_pts.reshape((-1, 1, 2))
            overlay = timeline_img.copy()
            cv2.fillPoly(overlay, [triangle_pts], special_marker_color)
            cv2.addWeighted(overlay, 0.7, timeline_img, 0.3, 0, timeline_img)

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
            cv2.line(overlay, (x_pos, 0), (x_pos, height), mark_color, cv_thickness)
            cv2.addWeighted(overlay, alpha, timeline_img, 1-alpha, 0, timeline_img)

    return timeline_img

def on_timeline_click(coords):
    """Handle clicks on the timeline image."""
    set_autoplay("None")
    
    if coords is None or 'x' not in coords:
        return
    
    # Calculate the time position based on click position
    x = coords['x']
    width = coords.get('width', 1)
    
    # Calculate the position as a percentage of the day
    if width <= 0:
        return  # Avoid division by zero
        
    day_percentage = x / width
    
    # Clamp percentage to valid range [0,1]
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

    # Update display and mark that a new time was selected
    update_image_display(direction="closest")
    sync_time_slider()

    #set_autoplay("None")

def build_main_area():
    """Create the main UI area elements and return placeholders."""
    # Apply CSS for main area styling
    st.markdown("""
    <style>
    /* General layout */
    .block-container { padding-top: 0.7rem !important; padding-bottom: 0.7rem !important; }
    div[data-testid="stVerticalBlock"] > div { padding-top: 1px !important; padding-bottom: 1px !important; }
    div[data-testid="element-container"] { margin-top: 1px !important; margin-bottom: 1px !important; }
    h1, h2, h3 { margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; padding: 0 !important; }
    hr { margin: 0 !important; padding: 0 !important; }
    /* Video */
    div[data-testid="stImage"] { margin-top: 2px !important; margin-bottom: 2px !important; }
    div[data-testid="stImage"] > img { margin: 2px 0 !important; padding: 2px 0 !important; }
    /* Time Display */
    .time-display { font-size: 1.6rem; font-weight: 400; text-align: center; font-family: "Source Sans Pro", sans-serif; background: transparent; color: #333; border-radius: 5px; padding: 2px; margin: 2px 0; }
    @media (prefers-color-scheme: dark) { .time-display { color: #e0e0e0 !important; } }
    [data-testid="stAppViewContainer"][data-theme="dark"] .time-display { color: #e0e0e0 !important; }
    /* Date Picker */
    div[data-testid="stDateInput"] { margin: 10px 0 0 0 !important; padding: 1px 0 !important; display: flex !important; flex-direction: column !important; justify-content: center !important; }
    div[data-testid="stDateInput"] > div { margin-bottom: 0 !important; display: flex !important; align-items: center !important; height: 32px !important; }
    div[data-testid="stDateInput"] input { padding: 2px 8px !important; height: 32px !important; font-size: 14px !important; background-color: #2e2e2e !important; color: #fff !important; border: 1px solid #555 !important; border-radius: 5px !important; text-align: center !important; font-weight: bold !important; margin: 0 !important; }
    div[data-testid="stDateInput"] svg { fill: #fff !important; }
    /* Slider */
    div[data-testid="stSlider"] { padding: 0 !important; margin: -10px 0 0 0 !important; }
    div[data-testid="stSlider"] > label { text-align: center !important; width: 100% !important; display: block !important; font-weight: bold !important; margin-bottom: 0 !important; height: 0 !important; overflow: hidden !important; }
    div[data-testid="stSlider"] > div > div > div { background-color: #555 !important; } /* Track */
    div[data-testid="stSlider"] > div > div > div > div { background-color: #0f0f0f !important; border-color: #fff !important; } /* Thumb */
    div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { display: none !important; }
    div[data-testid="stSlider"] > div { padding-top: 0 !important; margin-top: -5px !important; }
    div[data-testid="element-container"]:has(div[data-testid="stSlider"]) { margin-top: -5px !important; padding-top: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # --- Layout ---
    # MOVED: Video placeholder now created *after* controls
    # video_placeholder = st.empty()

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
        time_display.markdown(f'<div class="time-display">{format_time_12h(st.session_state.hour, st.session_state.minute, st.session_state.second)}</div>', unsafe_allow_html=True)
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
        button_text = "Live" if in_playback else "⏸"
        button_help = "Go Live" if in_playback else "Pause"
        st.button(button_text, key="live_btn", use_container_width=True, help=button_help,
                  on_click=toggle_live_pause, disabled=not is_capturing)

    # Timeline Bar - Replace HTML approach with image
    st.markdown('<div style="margin-top:-5px; margin-bottom:2px;">', unsafe_allow_html=True)
    
    # Generate timeline image with appropriate width and increased height for time labels
    timeline_img = generate_timeline_image(width=1200, height=40)
    
    # Store previous click coordinates to avoid duplicate processing
    prev_coords = st.session_state.get('prev_timeline_coords', None)
    
    # Display the clickable image
    clicked_coords = streamlit_image_coordinates(
        timeline_img, 
        key="timeline_bar",
        use_column_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
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

    # Time Slider
    #date_str = st.session_state.date.strftime("%Y-%m-%d")
    #st.slider(date_str, min_value=datetime_time(0,0,0), max_value=datetime_time(23,59,59),
    #          key="time_slider", step=timedelta(seconds=1), label_visibility="hidden",
    #          # Callback no longer needs args
    #          on_change=on_time_slider_change)

    st.markdown("</div>", unsafe_allow_html=True) # Close centered container

    # Create Video Placeholder *after* the controls container
    video_placeholder = st.empty() 

    # Return placeholders needed outside this build function (by callbacks and main loop)
    return video_placeholder, time_display


# --- Main UI Update Loop (runs in jiffycam.py) ---
# This function is moved from jiffycam.py
#@st.fragment(run_every=0.1)
def run_ui_update_loop():
    """The main loop to update the UI based on capture state and interactions."""
    # Fetch placeholders from session state at the start of the loop
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    error_placeholder = st.session_state.error_placeholder
    time_display = st.session_state.time_display

    while True:
        heartbeat()

        # Check for errors first
        if st.session_state.video_capture.error_event.is_set():
            st.session_state.video_capture.stop_capture() # Ensure cleanup
            error_placeholder.error(st.session_state.video_capture.last_error)
            status_placeholder.text("Status: Error")
            st.session_state.status_message = "Status: Error"
            # Keep last frame visible on error if possible
            if st.session_state.last_frame is not None:
                #video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                new_image_display(st.session_state.last_frame)
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
                    #video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    new_image_display(st.session_state.last_frame)
                    ts = st.session_state.actual_timestamp
                    new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback Mode"
                else:
                    new_status = "Playback Mode (No Frame)"
                    video_placeholder.empty() # Clear if no frame to show

            else: # Live View Mode (Capture Running)
                frame, fps, width, height = st.session_state.video_capture.get_frame()
                if frame is not None:
                    # Update time display and slider for live view
                    current_time = datetime.now()
                    time_display.markdown(f'<div class="time-display">{format_time_12h(current_time.hour, current_time.minute, current_time.second)}</div>', unsafe_allow_html=True)
                    if not st.session_state.get('slider_currently_being_dragged', False):
                        st.session_state.hour = current_time.hour
                        st.session_state.minute = current_time.minute
                        st.session_state.second = current_time.second

                    # Store & display live frame
                    st.session_state.last_frame = frame.copy()
                    #video_placeholder.image(frame, channels="BGR", use_container_width=True)
                    new_image_display(frame)
                    # Update internal stats
                    st.session_state.video_capture.current_fps = fps
                    st.session_state.video_capture.current_width = width
                    st.session_state.video_capture.current_height = height
                    
                    # Update status (Save msg or FPS)
                    if st.session_state.video_capture.image_just_saved:
                        new_status = st.session_state.video_capture.save_status
                        # Clear flag after timeout
                        if time.time() - st.session_state.video_capture.image_saved_time > 3:
                            st.session_state.video_capture.image_just_saved = False
                    else:
                        new_status = f"Live View - FPS: {fps}"
                # else: Keep last frame if get_frame returns None briefly?

        else: # Capture Stopped
            if st.session_state.in_playback_mode:
                # Playback Mode (Capture Stopped): Show paused frame
                if st.session_state.last_frame is not None:
                    #video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    new_image_display(st.session_state.last_frame)

                    ts = st.session_state.actual_timestamp
                    new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback (Stopped)"
                else:
                    new_status = "Playback (Stopped - No Frame)"
                    video_placeholder.empty()
            else:
                 # Not capturing, not in playback - Show last frame or empty
                if st.session_state.last_frame is not None:
                    #video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    new_image_display(st.session_state.last_frame)
                    new_status = "Capture Stopped"
                else:
                    video_placeholder.info("Capture stopped. Select date/time or Start Capture.")
                    new_status = "Status: Idle"

        # Update status placeholder only if message changed
        if True: #new_status != current_status:
            status_placeholder.text(new_status)
            st.session_state.status_message = new_status # Store new status

        # Main loop delay
        time.sleep(0.01)
