import cv2
import numpy as np

def draw_bboxes(frame, bboxes, ids):
    """
    Draws bounding boxes and IDs on the frame.

    Args:
        frame (np.array): The image frame (BGR).
        bboxes (list): List of bounding boxes in (x, y, w, h) format.
        ids (list): List of corresponding IDs for each bounding box.

    Returns:
        np.array: The frame with bounding boxes and IDs drawn.
    """
    output_frame = frame.copy()
    
    # Generate a unique color for each ID based on a hash of the ID
    # This ensures consistent colors for the same ID across frames without relying on random seed
    # and handles IDs beyond a small fixed range.
    def get_color_from_id(player_id):
        np.random.seed(player_id) # Seed with the ID to get a consistent color
        return tuple(map(int, np.random.randint(0, 255, size=3)))

    for i, bbox in enumerate(bboxes):
        # Ensure valid index and ID
        if i >= len(ids) or ids[i] is None or ids[i] == -1:
            continue
            
        x, y, w, h = map(int, bbox)
        player_id = ids[i]

        # Get color based on player_id
        color = get_color_from_id(player_id) # BGR
        thickness = 2

        # Draw rectangle
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, thickness)

        # Put text (ID) above the bounding box
        text = f"ID: {player_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255) # White text for visibility

        # Calculate text size to position it correctly
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Position text slightly above the top-left corner of the bbox
        text_x = x
        text_y = y - 10 if y - 10 > text_height else y + text_height + 5 # Adjust if too close to top edge

        # Draw text background for better readability
        # Ensure the background rectangle doesn't go out of bounds
        bg_x1 = text_x
        bg_y1 = text_y - text_height - baseline
        bg_x2 = text_x + text_width
        bg_y2 = text_y + baseline

        # Clamp coordinates to frame boundaries
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(output_frame.shape[1], bg_x2)
        bg_y2 = min(output_frame.shape[0], bg_y2)

        cv2.rectangle(output_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1) # -1 for filled rectangle

        # Draw text
        cv2.putText(output_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
    return output_frame

