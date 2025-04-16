import cv2
import numpy as np

def detect_document(frame):
    """
    Detect ID document in a frame and return the document contour
    
    Args:
        frame: Camera frame (numpy array)
        
    Returns:
        processed_frame: Frame with detected document highlighted
        document_detected: Boolean indicating if document was detected
        document_coords: Coordinates of the document if detected, None otherwise
    """
    # Make a copy of the frame to avoid modifying the original
    processed_frame = frame.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 75, 200)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    document_detected = False
    document_coords = None
    
    # Loop through the contours
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If the contour has 4 points, it's likely to be the document
        if len(approx) == 4:
            # Additional check: make sure it's large enough to be an ID
            area = cv2.contourArea(approx)
            frame_area = frame.shape[0] * frame.shape[1]
            min_area_ratio = 0.05  # Document should be at least 5% of the frame
            max_area_ratio = 0.9   # Document should not be more than 90% of the frame
            
            if min_area_ratio * frame_area < area < max_area_ratio * frame_area:
                # Draw the contour on the processed frame
                cv2.drawContours(processed_frame, [approx], -1, (0, 255, 0), 3)
                
                # Add corner markers
                for point in approx:
                    x, y = point[0]
                    cv2.circle(processed_frame, (x, y), 5, (0, 0, 255), -1)
                
                # Add "ID DETECTED" text
                cv2.putText(processed_frame, "ID DETECTED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                document_detected = True
                document_coords = approx
                break
    
    # If no document is detected, add helper rectangle
    if not document_detected:
        h, w = frame.shape[:2]
        # Draw guide rectangle
        cv2.rectangle(processed_frame, (int(w*0.2), int(h*0.2)), 
                      (int(w*0.8), int(h*0.8)), (0, 0, 255), 2)
        
        # Add text guide
        cv2.putText(processed_frame, "ALIGN ID WITHIN RECTANGLE", (int(w*0.2), int(h*0.15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return processed_frame, document_detected, document_coords

def extract_document(frame, coords):
    """
    Extract and rectify the document from the frame
    
    Args:
        frame: Camera frame (numpy array)
        coords: Coordinates of the document
        
    Returns:
        warped: Rectified document image
    """
    if coords is None:
        return None
    
    # Order the points in the contour
    rect = order_points(coords.reshape(4, 2))
    (top_left, top_right, bottom_right, bottom_left) = rect
    
    # Compute the width of the new image
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Compute the height of the new image
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Set of destination points for the birds-eye view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply the perspective transformation
    warped = cv2.warpPerspective(frame, M, (max_width, max_height))
    
    return warped

def order_points(pts):
    """
    Order points in the following order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect 