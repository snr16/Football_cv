import cv2
import numpy as np
import os
import json

class ViewTransformer():
    def __init__(self):
        # Standard football field dimensions in meters
        self.court_width = 68  # standard football field width
        self.court_length = 23.32  # visible football field length
        
        # Hardcoded field corners in pixel coordinates
        self.pixel_vertices = np.array([[110, 1015], 
                               [230, 335], 
                               [930, 315], 
                               [1704, 895]], dtype=np.float32)
        
        # Target vertices in real-world meters
        self.target_vertices = np.array([
            [0, self.court_width],          # Bottom-left
            [0, 0],                         # Top-left
            [self.court_length, 0],         # Top-right
            [self.court_length, self.court_width]  # Bottom-right
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
        self.is_calibrated = True

    def transform_point(self, point):
        """
        Transform a point from image coordinates to top-down view coordinates in meters
        Args:
            point: (x, y) coordinates in image space
        Returns:
            Transformed point in meters or None if invalid
        """
        if point is None or len(point) != 2:
            return None
            
        p = (int(point[0]), int(point[1]))
        
        # Check if point is inside the field with some tolerance
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= -10  # 10 pixel tolerance
        if not is_inside:
            return None

        # Transform the point
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        # Ensure the transformed point is in meters
        transformed_point = transformed_point.reshape(-1, 2)
        
        # Validate the transformed coordinates are within the field bounds with tolerance
        x, y = transformed_point[0]
        tolerance = 1.0  # 1 meter tolerance
        if not (-tolerance <= x <= self.court_length + tolerance and 
                -tolerance <= y <= self.court_width + tolerance):
            return None
            
        return transformed_point

    def add_transformed_position_to_tracks(self, tracks):
        """
        Add transformed positions to all tracks
        Args:
            tracks: Dictionary containing track data
        Returns:
            Updated tracks with transformed positions
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if object == 'ball':
                    # Ball track is a dictionary, not a list of dictionaries
                    position = track.get('position')
                    if position is not None:
                        position = np.array(position)
                        position_transformed = self.transform_point(position)
                        if position_transformed is not None:
                            position_transformed = position_transformed.squeeze().tolist()
                        tracks[object][frame_num]['position_transformed'] = position_transformed
                else:
                    # Players and referees are dictionaries with track IDs
                    for track_id, track_info in track.items():
                        position = track_info.get('position')
                        if position is not None:
                            position = np.array(position)
                            position_transformed = self.transform_point(position)
                            if position_transformed is not None:
                                position_transformed = position_transformed.squeeze().tolist()
                            tracks[object][frame_num][track_id]['position_transformed'] = position_transformed

        return tracks

    def create_visualization(self, frame, calibration_dir):
        """
        Create and save visualization of the field transformation
        Args:
            frame: Frame for visualization
            calibration_dir: Directory to save visualization results
        """
        # Create a copy of the frame for visualization
        validation_frame = frame.copy()
        
        # Draw original field corners
        for i in range(4):
            cv2.line(validation_frame, 
                    tuple(map(int, self.pixel_vertices[i])), 
                    tuple(map(int, self.pixel_vertices[(i+1)%4])), 
                    (0, 255, 0), 2)
            cv2.circle(validation_frame, tuple(map(int, self.pixel_vertices[i])), 5, (0, 0, 255), -1)
        
        # Transform the frame to top-down view
        height, width = frame.shape[:2]
        transformed_frame = cv2.warpPerspective(frame, self.perspective_transformer, 
                                              (int(self.court_length), int(self.court_width)))
        
        # Draw transformed field lines
        for i in range(4):
            cv2.line(transformed_frame, 
                    tuple(map(int, self.target_vertices[i])), 
                    tuple(map(int, self.target_vertices[(i+1)%4])), 
                    (255, 0, 0), 2)
        
        # Create field grid visualization
        grid_frame = transformed_frame.copy()
        self._draw_field_grid(grid_frame)
        
        # Create field zones visualization
        zones_frame = transformed_frame.copy()
        self._draw_field_zones(zones_frame)
        
        # Save validation images
        cv2.imwrite(os.path.join(calibration_dir, 'view_transform_original.jpg'), validation_frame)
        cv2.imwrite(os.path.join(calibration_dir, 'view_transform_transformed.jpg'), transformed_frame)
        cv2.imwrite(os.path.join(calibration_dir, 'view_transform_grid.jpg'), grid_frame)
        cv2.imwrite(os.path.join(calibration_dir, 'view_transform_zones.jpg'), zones_frame)
        
        # Save transformation data
        transform_data = {
            'pixel_vertices': self.pixel_vertices.tolist(),
            'target_vertices': self.target_vertices.tolist(),
            'field_dimensions': {
                'length': self.court_length,
                'width': self.court_width
            }
        }
        
        with open(os.path.join(calibration_dir, 'view_transform_data.json'), 'w') as f:
            json.dump(transform_data, f, indent=4)
        
        return validation_frame, transformed_frame

    def _draw_field_grid(self, frame):
        """
        Draw grid lines on the field
        """
        # Draw vertical lines every 10 meters
        for x in range(0, int(self.court_length) + 1, 10):
            cv2.line(frame, (x, 0), (x, int(self.court_width)), (255, 255, 255), 1)
            cv2.putText(frame, f"{x}m", (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw horizontal lines every 10 meters
        for y in range(0, int(self.court_width) + 1, 10):
            cv2.line(frame, (0, y), (int(self.court_length), y), (255, 255, 255), 1)
            cv2.putText(frame, f"{y}m", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_field_zones(self, frame):
        """
        Draw field zones (defensive, midfield, attacking)
        """
        # Define zones
        zones = {
            'defensive': [(0, 0), (self.court_length/3, 0), 
                         (self.court_length/3, self.court_width), (0, self.court_width)],
            'midfield': [(self.court_length/3, 0), (2*self.court_length/3, 0),
                        (2*self.court_length/3, self.court_width), (self.court_length/3, self.court_width)],
            'attacking': [(2*self.court_length/3, 0), (self.court_length, 0),
                         (self.court_length, self.court_width), (2*self.court_length/3, self.court_width)]
        }
        
        # Draw zones with different colors and transparency
        colors = {
            'defensive': (0, 0, 255, 0.2),    # Red
            'midfield': (0, 255, 0, 0.2),     # Green
            'attacking': (255, 0, 0, 0.2)     # Blue
        }
        
        for zone_name, points in zones.items():
            overlay = frame.copy()
            points = np.array(points, np.int32)
            cv2.fillPoly(overlay, [points], colors[zone_name][:3])
            cv2.addWeighted(overlay, colors[zone_name][3], frame, 1 - colors[zone_name][3], 0, frame)
            cv2.polylines(frame, [points], True, colors[zone_name][:3], 2)

    def visualize_trajectory(self, tracks, frame, calibration_dir):
        """
        Visualize player trajectories on the field
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate before visualizing trajectories")

        # Create a blank top-down view
        trajectory_frame = np.zeros((int(self.court_length), int(self.court_width), 3), dtype=np.uint8)
        
        # Draw field grid
        self._draw_field_grid(trajectory_frame)
        
        # Draw trajectories for each player
        colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)  # Generate random colors for players
        
        for obj_name, obj in tracks.items():
            if obj_name != 'players':
                continue
                
            for track_id, track_info in obj[0].items():
                color = colors[track_id % len(colors)]
                positions = []
                
                # Collect all positions for this player
                for frame_data in obj:
                    if track_id in frame_data:
                        pos = frame_data[track_id]['position_transformed']
                        if pos is not None:
                            positions.append(pos)
                
                # Draw trajectory
                if len(positions) > 1:
                    positions = np.array(positions, np.int32)
                    cv2.polylines(trajectory_frame, [positions], False, color.tolist(), 2)
                    
                    # Draw start and end points
                    cv2.circle(trajectory_frame, tuple(positions[0]), 5, (0, 255, 0), -1)  # Start: Green
                    cv2.circle(trajectory_frame, tuple(positions[-1]), 5, (0, 0, 255), -1)  # End: Red
        
        # Save trajectory visualization
        cv2.imwrite(os.path.join(calibration_dir, 'player_trajectories.jpg'), trajectory_frame)

    def visualize_heatmap(self, tracks, frame, calibration_dir):
        """
        Create a heatmap of player positions
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate before creating heatmap")

        # Create a blank top-down view
        heatmap = np.zeros((int(self.court_length), int(self.court_width)), dtype=np.float32)
        
        # Collect all player positions
        positions = []
        for obj_name, obj in tracks.items():
            if obj_name != 'players':
                continue
                
            for frame_data in obj:
                for track_id, track_info in frame_data.items():
                    pos = track_info['position_transformed']
                    if pos is not None:
                        positions.append(pos)
        
        if positions:
            positions = np.array(positions, np.int32)
            
            # Create heatmap using Gaussian kernel
            for pos in positions:
                cv2.circle(heatmap, tuple(pos), 5, 1, -1)
            
            # Apply Gaussian blur
            heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
            
            # Normalize and convert to color
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Draw field grid on top
            self._draw_field_grid(heatmap)
            
            # Save heatmap
            cv2.imwrite(os.path.join(calibration_dir, 'player_heatmap.jpg'), heatmap) 