"""
jiffyui_components.py: UI component building functions for JiffyCam

This module contains reusable UI component builders and styling for Streamlit UI.
These are extracted from jiffyui.py to promote code reuse and maintainability.
"""

import streamlit as st
from datetime import datetime
import numpy as np

def apply_general_css():
    """Apply general CSS styling for the application."""
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
    
    /* Day navigation buttons styling */
    button[data-testid="baseButton-secondary"]:has(div:contains("◀")),
    button[data-testid="baseButton-secondary"]:has(div:contains("▶")) {
        padding: 0 !important;
    }
    
    /* Centered controls container with margin for better positioning */
    .top-controls {
        max-width: 600px;
        margin: 20px 0 0 0;
    }
    </style>
    """, unsafe_allow_html=True)

def apply_metrics_css():
    """Apply CSS styling specifically for metrics display."""
    st.markdown("""
    <style>
    /* Make metric labels and values more compact */
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding-bottom: 0.1rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Reduce padding around metric containers */
    [data-testid="metric-container"] {
        padding: 5px 0px !important;
        margin-bottom: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_navigation_button(label, key, help_text, on_click_handler, disabled=False, use_container_width=True):
    """Create a standardized navigation button.
    
    Args:
        label: The text/symbol to display on the button
        key: Unique identifier for the button
        help_text: Tooltip text for the button
        on_click_handler: Function to call when button is clicked
        disabled: Whether the button should be disabled
        use_container_width: Whether the button should use the full container width
    
    Returns:
        The created button
    """
    return st.button(
        label, 
        key=key, 
        use_container_width=use_container_width,
        help=help_text,
        on_click=on_click_handler,
        disabled=disabled
    )

def create_playback_button(label, key, help_text, on_click_handler, disabled=False, button_type="secondary"):
    """Create a standardized playback control button.
    
    Args:
        label: The text/symbol to display on the button
        key: Unique identifier for the button
        help_text: Tooltip text for the button
        on_click_handler: Function to call when button is clicked
        disabled: Whether the button should be disabled
        button_type: The button style ("primary" or "secondary")
    
    Returns:
        The created button
    """
    return st.button(
        label, 
        key=key, 
        use_container_width=True,
        help=help_text,
        on_click=on_click_handler,
        disabled=disabled,
        type=button_type
    )

def create_metric_display(label, value, help_text="", delta=None, border=True):
    """Create a standardized metric display.
    
    Args:
        label: The label for the metric
        value: The value to display
        help_text: Tooltip text for the metric
        delta: Optional delta value to display
        border: Whether to show a border around the metric
    
    Returns:
        The created metric widget
    """
    return st.metric(
        label,
        value,
        delta=delta,
        delta_color="normal",
        help=help_text,
        label_visibility="visible",
        border=border
    )

def create_fps_metrics_row():
    """Create a row with FPS Camera and FPS Display metrics.
    
    Returns:
        Tuple of (capture_fps_placeholder, display_fps_placeholder)
    """
    fps_cols = st.columns(2)
    
    with fps_cols[0]:
        capture_fps_placeholder = create_metric_display(
            "FPS Camera", 
            "---", 
            "Frames per second being captured by the camera"
        )
        
    with fps_cols[1]:
        display_fps_placeholder = create_metric_display(
            "FPS Display", 
            "---", 
            "Frames per second being displayed"
        )
        
    return capture_fps_placeholder, display_fps_placeholder

def create_detection_metrics():
    """Create metrics for detection counts and timestamps.
    
    Returns:
        Tuple of (frames_detected_placeholder, last_save_time_placeholder)
    """
    frames_detected_placeholder = create_metric_display(
        "Detections",
        0,
        "Number of frames detected for current date"
    )
    
    last_save_time_placeholder = create_metric_display(
        "Last Detection",
        "---",
        "Time of last image detection and save"
    )
    
    return frames_detected_placeholder, last_save_time_placeholder

def create_date_navigation(prev_handler, next_handler, single_day_mode=False):
    """Create date navigation buttons and picker in a row.
    
    Args:
        prev_handler: Function to handle previous day button click
        next_handler: Function to handle next day button click
        single_day_mode: Whether there's only one day available (disables buttons)
    
    Returns:
        Tuple of (date_cols, date_picker_placeholder)
        where date_cols is a list of column objects and
        date_picker_placeholder is the center column placeholder for the date picker
    """
    # Date Picker with navigation buttons in a row
    date_cols = st.columns([0.7, 5, 0.7])
    
    # Previous day button
    with date_cols[0]:
        st.write("")  # Space for alignment
        create_navigation_button(
            "◀", 
            "prev_day_button",
            "Previous Day" if not single_day_mode else "No previous days available",
            prev_handler,
            disabled=single_day_mode
        )
    
    # Date picker placeholder
    with date_cols[1]:
        date_picker_placeholder = st.empty()
    
    # Next day button
    with date_cols[2]:
        st.write("")  # Space for alignment
        create_navigation_button(
            "▶", 
            "next_day_button",
            "Next Day" if not single_day_mode else "No more days available",
            next_handler,
            disabled=single_day_mode
        )
    
    return date_cols, date_picker_placeholder

def create_playback_controls(handlers):
    """Create a row of playback control buttons.
    
    Args:
        handlers: Dictionary with keys 'fast_reverse', 'prev', 'pause', 'next', 
                 'fast_forward', and 'live' pointing to the respective handler functions
    
    Returns:
        Tuple of (time_cols, time_display) where time_cols is a list of column objects
        and time_display is the placeholder for the time display
    """
    # Create row for playback controls
    time_cols = st.columns([3, 1, 1, 1, 1, 1, 0.2, 1])
    
    # Time display
    with time_cols[0]:
        time_display = st.empty()
    
    # Fast reverse button
    with time_cols[1]:
        create_playback_button(
            "⏪", 
            "fast_reverse_button", 
            "Fast Reverse",
            handlers['fast_reverse']
        )
    
    # Previous button
    with time_cols[2]:
        create_playback_button(
            "◀️", 
            "prev_button", 
            "Previous",
            handlers['prev']
        )
    
    # Pause button
    with time_cols[3]:
        create_playback_button(
            "⏸️", 
            "pause_button", 
            "Pause",
            handlers['pause']
        )
    
    # Next button
    with time_cols[4]:
        create_playback_button(
            "▶️", 
            "next_button", 
            "Next",
            handlers['next']
        )
    
    # Fast forward button
    with time_cols[5]:
        create_playback_button(
            "⏩", 
            "fast_forward_button", 
            "Fast Forward",
            handlers['fast_forward']
        )
    
    # Separator
    with time_cols[6]:
        st.markdown('<div style="width:1px;background-color:#555;height:32px;margin:0 auto;"></div>', unsafe_allow_html=True)
    
    # Live button - handled separately as it may have conditional styling
    
    return time_cols, time_display

def create_live_button(handler, is_enabled, is_live_mode, help_text, session_name=None):
    """Create the Live button with appropriate styling based on state.
    
    Args:
        handler: Function to handle button click
        is_enabled: Whether the button should be enabled
        is_live_mode: Whether we're currently in live mode (affects styling)
        help_text: Default help text (may be overridden based on state)
        session_name: Optional session name for additional context in help text
    
    Returns:
        The created button
    """
    button_text = "Live"
    
    if not is_live_mode:
        # Not in live mode - use secondary (gray) styling
        if not help_text:
            if is_enabled:
                button_help = "Switch to live view"
            else:
                button_help = f"Server not capturing session '{session_name}'" if session_name else "Live view not available"
        else:
            button_help = help_text
        
        button_type = "secondary"
    else:
        # In live mode - use primary (red) styling
        button_help = "Pause live view" if not help_text else help_text
        button_type = "primary"
    
    return create_playback_button(
        button_text,
        "live_btn",
        button_help,
        handler,
        disabled=not is_enabled,
        button_type=button_type
    )

def create_empty_timeline_arrow(width=1200, height=24):
    """Create an empty timeline arrow image placeholder.
    
    Args:
        width: Width of the timeline arrow image
        height: Height of the timeline arrow image
    
    Returns:
        Tuple of (placeholder, initial_image) where placeholder is the Streamlit
        empty element and initial_image is the empty numpy array
    """
    placeholder = st.empty()
    initial_image = np.zeros((height, width, 3), dtype=np.uint8)
    placeholder.image(initial_image, channels="RGB", use_container_width=True)
    return placeholder, initial_image

def create_recording_selector(options, current_selection, on_change_handler, help_text="Select the recording session to view.", server_session=None):
    """Create a selector for recordings with formatting for the active recording.
    
    Args:
        options: List of recording options
        current_selection: Currently selected recording
        on_change_handler: Function to call when selection changes
        help_text: Tooltip text for the selector
        server_session: Current active server session (to highlight)
    
    Returns:
        The created selectbox
    """
    format_func = lambda x: (x+" (Active)") if (x==server_session) else x
    
    return st.selectbox(
        "Select Recording",
        options=options,
        key="selected_recording_key",
        on_change=on_change_handler,
        help=help_text,
        label_visibility="collapsed",
        format_func=format_func
    )

def create_date_picker(value, min_value, max_value, on_change_handler, key="date", help_text="Select date", single_day=False):
    """Create a date picker with appropriate range constraints.
    
    Args:
        value: Current date value
        min_value: Minimum selectable date
        max_value: Maximum selectable date
        on_change_handler: Function to call when date changes
        key: Unique key for the date picker
        help_text: Tooltip text for the date picker
        single_day: Whether this is a single-day picker (fixed to one date)
    
    Returns:
        The created date picker
    """
    try:
        if single_day:
            return st.date_input(
                "Date",
                value=value,
                key=key,
                on_change=on_change_handler,
                help=help_text,
                label_visibility="collapsed",
                min_value=value,  # Lock to the specified date
                max_value=value
            )
        else:
            return st.date_input(
                "Date",
                value=value,
                key=key,
                on_change=on_change_handler,
                help=help_text,
                label_visibility="collapsed",
                min_value=min_value,
                max_value=max_value
            )
    except Exception as e:
        # Fallback for date picker errors
        print(f"Date picker error: {str(e)}")
        return st.date_input(
            "Date",
            value=value,
            key=f"{key}_fallback",
            on_change=on_change_handler,
            label_visibility="collapsed"
        ) 