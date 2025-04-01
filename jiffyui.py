"""
jiffyui.py: UI components and logic for JiffyCam

This module contains functions for building and updating the Streamlit UI.
"""

from datetime import datetime, timedelta, time as datetime_time
import time
import os
import streamlit as st

from jiffyconfig import RESOLUTIONS   # Import necessary components from other modules
from jiffyget import jiffyget

# --- UI Helper Functions ---

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
        st.session_state.time_slider = datetime_time(0, 0, 0) # Reset slider to midnight

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

    # Fetch placeholders from session_state inside the update function
    update_image_display(direction="down")

def on_date_change():
    """Handle date picker change."""
    st.session_state.browsing_date = st.session_state.date
    st.session_state.in_playback_mode = True

    # Sync slider to current time state
    st.session_state.time_slider = datetime_time(
        st.session_state.hour,
        st.session_state.minute,
        st.session_state.second
    )
    # Fetch placeholders from session_state inside the update function
    update_image_display(direction="down")

def on_prev_button():
    """Handle previous image button click."""
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

def on_next_button():
    """Handle next image button click."""
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

def toggle_live_pause():
    """Handle Live/Pause button click."""
    if st.session_state.in_playback_mode:
        # Go Live
        st.session_state.in_playback_mode = False
        st.session_state.live_button_clicked = True
        # Update browsing date/time to current (will be reflected in loop)
        current_time = datetime.now()
        st.session_state.browsing_date = current_time.date()
        # Let the main loop update hour/min/sec/slider when live
    else:
        # Pause
        st.session_state.in_playback_mode = True
        # Store the timestamp of the *currently displayed* frame if available
        # If last_frame exists, its timestamp should be in actual_timestamp
        # If not, store current time as a fallback?
        if st.session_state.last_frame is None:
            st.session_state.actual_timestamp = datetime.now() # Fallback
        # No need to explicitly set hour/min/sec here, they reflect the paused frame

def on_time_slider_change():
    """Handle time slider change."""
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
        st.session_state.in_playback_mode = True

        # Fetch placeholders from session_state inside the update function
        update_image_display(direction="down")

        st.session_state.slider_currently_being_dragged = False

# --- UI Update Functions ---

def update_image_display(direction=None):
    """Update the image display based on the current date and time."""
    # Fetch placeholders from session state
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    time_display = st.session_state.time_display

    st.session_state.in_playback_mode = True

    # --- Get necessary state for jiffyget ---
    session = st.session_state.get('session', 'Default') # Safely get session
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData') # Get data_dir from config
    browse_date = st.session_state.browsing_date # Get the date being browsed
    # --- End state retrieval ---

    # Find the closest image using jiffyget directly
    closest_image = jiffyget(
        st.session_state.hour, st.session_state.minute, st.session_state.second,
        st.session_state.cam_name,
        session,       # Pass session variable
        data_dir,      # Pass data_dir variable
        browse_date,   # Pass browse_date variable
        direction
    )

    success = False
    if closest_image:
        frame, timestamp = closest_image
        if frame is not None:
            try:
                # Store frame and actual timestamp
                st.session_state.last_frame = frame.copy()
                st.session_state.actual_timestamp = timestamp

                # Update UI placeholders
                video_placeholder.image(frame, channels="BGR", use_container_width=True)

                # Update time state to match the found image
                st.session_state.hour = timestamp.hour
                st.session_state.minute = timestamp.minute
                st.session_state.second = timestamp.second
                st.session_state.browsing_date = timestamp.date() # Keep browsing date in sync with found image

                # Update time display text
                time_display.markdown(
                    f'<div class="time-display">{format_time_12h(timestamp.hour, timestamp.minute, timestamp.second)}</div>',
                    unsafe_allow_html=True
                )

                # Update status text
                status_placeholder.text(f"Viewing: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                st.session_state.status_message = f"Viewing: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}" # Update internal status too
                success = True

            except Exception as e:
                status_placeholder.error(f"Error displaying image: {str(e)}")
                st.session_state.status_message = f"Error displaying image: {str(e)}"

    if not success:
        # No image found or error displaying
        status_placeholder.text(f"No image found near specified time")
        st.session_state.status_message = f"No image found near specified time"
        # Keep showing the last valid frame if one exists
        if st.session_state.last_frame is not None:
            video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
        else: # No previous frame, clear the placeholder
            video_placeholder.empty()

    # Ensure playback mode remains true
    st.session_state.in_playback_mode = True
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

def generate_timeline_bar_html():
    """Generate HTML for the timeline bar based on available data."""
    session = st.session_state.get('session', 'Default')
    data_dir = st.session_state.video_capture.config.get('data_dir', 'JiffyData')

    # Construct base path safely
    cam_name_parent = os.path.dirname(st.session_state.cam_name)
    base_dir = os.path.join(data_dir, session, cam_name_parent) if cam_name_parent else os.path.join(data_dir, session)

    timestamps = []
    if os.path.exists(base_dir):
        try:
            current_date = st.session_state.date # Assumes it's a date object
            for dir_name in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, dir_name)):
                    try:
                        timestamp_ms = int(dir_name)
                        dt = datetime.fromtimestamp(timestamp_ms / 1000)
                        if dt.date() == current_date:
                            seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
                            position = (seconds_since_midnight / (24 * 60 * 60)) * 100
                            timestamps.append(position)
                    except (ValueError, TypeError):
                        continue # Ignore invalid directory names or date issues
        except FileNotFoundError:
            pass # Ignore if base_dir disappears

    # Base HTML structure (always include CSS)
    html = """
    <style>
    .timeline-container{width:100%;padding:0;margin:0;margin-top:-5px;margin-bottom:2px;}
    .timeline-bar{position:relative;width:100%;height:8px;background-color:#333333;border-radius:2px;}
    .timeline-mark{position:absolute;width:2px;height:8px;background-color:#ffffff;opacity:0.7;}
    </style>
    <div class="timeline-container"><div class="timeline-bar">
    """
    # Add marks if timestamps were found
    if timestamps:
        for position in timestamps:
            html += f'<div class="timeline-mark" style="left: {position}%;"></div>'
    # Close HTML
    html += "</div></div>"
    return html

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
    time_cols = st.columns([3, 1, 1, 0.2, 1])
    with time_cols[0]: # Time Display Placeholder
        time_display = st.empty()
        # Set initial display value
        time_display.markdown(f'<div class="time-display">{format_time_12h(st.session_state.hour, st.session_state.minute, st.session_state.second)}</div>', unsafe_allow_html=True)
    with time_cols[1]: # Prev Button
        st.button("◀", key="prev_button", use_container_width=True, help="Previous",
                  # Callback no longer needs args
                  on_click=on_prev_button)
    with time_cols[2]: # Next Button
        st.button("▶", key="next_button", use_container_width=True, help="Next",
                  # Callback no longer needs args
                  on_click=on_next_button)
    with time_cols[3]: # Separator
        st.markdown('<div style="width:1px;background-color:#555;height:32px;margin:0 auto;"></div>', unsafe_allow_html=True)
    with time_cols[4]: # Live/Pause Button
        is_capturing = st.session_state.video_capture.is_capturing()
        in_playback = st.session_state.in_playback_mode
        button_text = "Live" if in_playback else "⏸"
        button_help = "Go Live" if in_playback else "Pause"
        st.button(button_text, key="live_btn", use_container_width=True, help=button_help,
                  on_click=toggle_live_pause, disabled=not is_capturing)

    # Timeline Bar
    timeline_html = generate_timeline_bar_html()
    st.markdown(timeline_html, unsafe_allow_html=True)

    # Time Slider
    date_str = st.session_state.date.strftime("%Y-%m-%d")
    st.slider(date_str, min_value=datetime_time(0,0,0), max_value=datetime_time(23,59,59),
              key="time_slider", step=timedelta(seconds=1), label_visibility="hidden",
              # Callback no longer needs args
              on_change=on_time_slider_change)

    st.markdown("</div>", unsafe_allow_html=True) # Close centered container

    # Create Video Placeholder *after* the controls container
    video_placeholder = st.empty()

    # Return placeholders needed outside this build function (by callbacks and main loop)
    return video_placeholder, time_display


# --- Main UI Update Loop (runs in jiffycam.py) ---
# This function is moved from jiffycam.py
def run_ui_update_loop():
    """The main loop to update the UI based on capture state and interactions."""
    # Fetch placeholders from session state at the start of the loop
    video_placeholder = st.session_state.video_placeholder
    status_placeholder = st.session_state.status_placeholder
    error_placeholder = st.session_state.error_placeholder
    time_display = st.session_state.time_display

    while True:
        # Check for errors first
        if st.session_state.video_capture.error_event.is_set():
            st.session_state.video_capture.stop_capture() # Ensure cleanup
            error_placeholder.error(st.session_state.video_capture.last_error)
            status_placeholder.text("Status: Error")
            st.session_state.status_message = "Status: Error"
            # Keep last frame visible on error if possible
            if st.session_state.last_frame is not None:
                video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
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
                    # Update stats silently (optional)
                    # _, fps, width, height = frame_data
                    # st.session_state.video_capture.current_fps = fps ... etc

                # Ensure paused frame is displayed
                if st.session_state.last_frame is not None:
                    video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
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
                    video_placeholder.image(frame, channels="BGR", use_container_width=True)
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
                    video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    ts = st.session_state.actual_timestamp
                    new_status = f"Viewing: {ts.strftime('%Y-%m-%d %H:%M:%S')}" if ts else "Playback (Stopped)"
                else:
                    new_status = "Playback (Stopped - No Frame)"
                    video_placeholder.empty()
            else:
                 # Not capturing, not in playback - Show last frame or empty
                if st.session_state.last_frame is not None:
                    video_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    new_status = "Capture Stopped"
                else:
                    video_placeholder.info("Capture stopped. Select date/time or Start Capture.")
                    new_status = "Status: Idle"

        # Update status placeholder only if message changed
        if new_status != current_status:
            status_placeholder.text(new_status)
            st.session_state.status_message = new_status # Store new status

        # Main loop delay
        time.sleep(0.05)
