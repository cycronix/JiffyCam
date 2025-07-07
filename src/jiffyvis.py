"""
jiffyvis.py: Timeline visualization functions for JiffyCam

This module provides visualization functions for creating timeline images,
markers and other visual elements used in the JiffyCam UI.
"""

import time
import numpy as np
import cv2
import streamlit as st
from jiffyget import get_locations
#import inspect

def format_time_12h(hour, minute, second):
    """Format time in 12-hour format with AM/PM indicator."""
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12}:{minute:02d}:{second:02d} {period}"

#import inspect
def generate_timeline_image(useTimestamps=True, width=1200, height=48):
    #print(f"generate_timeline_image: {inspect.stack()[1].function}")
    """Generate an image for the timeline bar based on available data."""
    session = st.session_state.get('session', 'Default')
    data_dir = st.session_state.get('data_dir', 'JiffyData')

    # Construct base path safely
    browse_date = st.session_state.browsing_date
    browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
    
    timestamps = None
    if(useTimestamps):
        timestamps = get_locations(st.session_state.cam_name, session, data_dir, browse_date_posix)
        #if(timestamps is None):
        #    return np.zeros((height, width, 3), dtype=np.uint8)

    # Create a blank image (dark gray background)
    background_color = (51, 51, 51)  # Dark gray
    mark_color = (255, 20, 20)  # Red in BGR format (Blue = 0, Green = 0, Red = 255) 
    hour_marker_color = (180, 180, 180)  # Light gray for hour markers
    special_marker_color = (255, 255, 0)  # Yellow in BGR format
    text_color = (220, 220, 220)  # Brighter light gray for text
    
    # Add equal padding for top and bottom margins
    text_padding = 15  # Pixels below timeline for text (reduced from 20)
    top_margin = 0  # Match top margin to text padding (was 12)
    timeline_y_start = top_margin  # Timeline now starts below top margin
    timeline_y_end = timeline_y_start + height
    total_height = timeline_y_end + text_padding  # Total image height
    #print(f"total_height: {total_height}")

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
    #print(f"timeline_img: {timeline_img.shape}")

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
    #print(f"timestamps: {timestamps}")
    if timestamps:
        for position in timestamps:
            # Calculate pixel position
            x_pos = int(position * width)
            cv2.line(timeline_img, (x_pos, timeline_y_start), (x_pos, timeline_y_end), mark_color, 1)

    return timeline_img

def generate_timeline_arrow(width=1200, height=24, with_markers=False):
    """
    Generate an arrow image for the timeline position indicator.
    This shows the current time position on the timeline.
    
    Args:
        width: Width of the arrow image
        height: Height of the arrow image
        with_markers: Whether to include timeline markers (red vertical lines)
    """
    #print(f"generate_timeline_arrow: {inspect.stack()[1].function}")
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
        
        # Add red timeline markers if requested
        if with_markers:
            session = st.session_state.get('session', 'Default')
            data_dir = st.session_state.get('data_dir', 'JiffyData')
            
            # Get the timestamps from the browsing date
            try:
                browse_date = st.session_state.browsing_date
                browse_date_posix = int(time.mktime(browse_date.timetuple()) * 1000)
                timestamps = get_locations(st.session_state.cam_name, session, data_dir, browse_date_posix)
                
                if timestamps:
                    mark_color = (255, 20, 20)  # Red in BGR format (same as in timeline)
                    
                    # Add marks for timestamps
                    for position in timestamps:
                        # Calculate pixel position
                        marker_x_pos = int(position * width)
                        # Draw a red vertical line with alpha blending
                        cv_thickness = max(1, int(width/1000))  # Scale thickness with width
                        alpha = 0.8  # 80% opacity
                        
                        # Draw semi-transparent line across full height of arrow image
                        overlay = timeline_img.copy()
                        cv2.line(overlay, (marker_x_pos, 0), (marker_x_pos, height), mark_color, cv_thickness)
                        cv2.addWeighted(overlay, alpha, timeline_img, 1-alpha, 0, timeline_img)
            except Exception as e:
                print(f"Error adding markers to timeline arrow: {str(e)}")

        # Skip if too close to edges
        if x_pos >= radius and x_pos <= width - radius:
            # Draw special marker (thicker line with yellow color)
            cv_thickness = max(2, int(width/800))  # Thicker than regular markers
            
            # Draw semi-transparent triangle ABOVE timeline (pointing down)
            triangle_height = int(height * 0.75)  # Reduced to 25% of timearrow height
            triangle_width = int(height * 0.6)  # Maintain proportional width
            
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