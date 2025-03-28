import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners, display_frame, preview_frame = param
        corners.append((x, y))
        
        # Draw selected point
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw lines between selected points
        if len(corners) > 1:
            cv2.line(display_frame, corners[-2], corners[-1], (0, 255, 0), 2)
        
        # Draw preview of calibration
        preview_frame = display_frame.copy()
        if len(corners) == 4:
            # Draw complete field outline
            for i in range(4):
                cv2.line(preview_frame, corners[i], corners[(i+1)%4], (0, 255, 0), 2)
            
            # Add preview text
            cv2.putText(preview_frame, "Preview of Calibration", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview_frame, "Press 'Enter' to confirm, 'r' to reset", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibration Preview', preview_frame)
        
        cv2.imshow('Select Field Corners', display_frame)

class FieldCornerSelector:
    def __init__(self):
        self.corners = []
        self.display_frame = None
        self.preview_frame = None

    def get_instructions(self):
        return [
            "Field Corner Selection Instructions:",
            "1. Click the corners in this specific order:",
            "   - Top Left: The leftmost corner of the field",
            "   - Top Right: The rightmost corner of the field",
            "   - Bottom Right: The rightmost corner at the bottom",
            "   - Bottom Left: The leftmost corner at the bottom",
            "",
            "Tips for accurate selection:",
            "- Select the actual field corners, not the video frame",
            "- Choose points that are clearly visible",
            "- Try to select points where the field lines meet",
            "",
            "Controls:",
            "- Left click: Select corner",
            "- 'r': Reset selection",
            "- 'q': Quit and use default corners",
            "- 'Enter': Confirm selection"
        ]

    def draw_instructions(self):
        y_pos = 30
        for line in self.get_instructions():
            cv2.putText(self.display_frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20

    def get_default_corners(self, frame):
        height, width = frame.shape[:2]
        return [
            (width//4, height//4),      # top_left
            (3*width//4, height//4),    # top_right
            (3*width//4, 3*height//4),  # bottom_right
            (width//4, 3*height//4)     # bottom_left
        ]

    def select_corners(self, frame):
        """
        Manually select the four corners of the football field using mouse clicks
        Returns: List of 4 points [top_left, top_right, bottom_right, bottom_left]
        """
        self.corners = []
        self.display_frame = frame.copy()
        self.preview_frame = frame.copy()
        
        # Draw instructions
        self.draw_instructions()
        
        cv2.namedWindow('Select Field Corners')
        cv2.setMouseCallback('Select Field Corners', mouse_callback, 
                           [self.corners, self.display_frame, self.preview_frame])
        
        while True:
            cv2.imshow('Select Field Corners', self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.corners = []
                self.display_frame = frame.copy()
                self.draw_instructions()
            elif key == 13 and len(self.corners) == 4:  # Enter key
                break
        
        cv2.destroyAllWindows()
        
        if len(self.corners) != 4:
            print("Warning: Not all corners selected. Using default corners.")
            return self.get_default_corners(frame)
        
        return self.corners 