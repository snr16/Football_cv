import cv2
import numpy as np
import sys
import os
import json
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5  # Using 5 frames for smoother speed calculation
        self.frame_rate = 24
        self.total_distance_covered = {}
        self.max_speed = 40  # Maximum reasonable speed in km/h
        self.min_speed = 0.1  # Minimum reasonable speed in km/h
        self.speed_history = {}  # Store speed history for smoothing
        self.debug_log = []  # Store debug information

    def add_speed_and_distance_to_tracks(self, tracks, frames=None):
        """
        Add speed and distance measurements to all tracks
        Args:
            tracks: Dictionary containing track data with transformed positions
            frames: Optional list of video frames for visualization
        """
        total_distance_covered = {}
        for obj_name, obj in tracks.items():
            if obj_name == 'ball' or obj_name == 'referee':
                continue

            number_of_frames = len(obj)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, track_info in obj[frame_num].items():
                    if track_id not in obj[last_frame]:
                        continue
                    
                    # Get both original and transformed positions
                    start_position = track_info['position']
                    end_position = obj[last_frame][track_id]['position']
                    start_position_transformed = track_info['position_transformed']
                    end_position_transformed = obj[last_frame][track_id]['position_transformed']

                    if start_position_transformed is None or end_position_transformed is None:
                        continue

                    # Calculate distance in meters using transformed positions
                    distance_covered = np.linalg.norm(np.array(end_position_transformed) - np.array(start_position_transformed))

                    
                    # Calculate time in seconds
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    
                    # Calculate speed in km/h
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # # Log positions for debugging (every 30 frames)
                    # # if frame_num % 30 == 0:
                    # debug_info = {
                    #     'frame': frame_num,
                    #     'track_id': track_id,
                    #     'start_pos': start_position,
                    #     'end_pos': end_position,
                    #     'start_pos_transformed': start_position_transformed,
                    #     'end_pos_transformed': end_position_transformed
                    # }
                    # self.debug_log.append(debug_info)
                    # print(f"\nDebug Info for Frame {frame_num}, Track {track_id}:")
                    # print(f"Original Start: {start_position}")
                    # print(f"Original End: {end_position}")
                    # print(f"Transformed Start: {start_position_transformed}")
                    # print(f"Transformed End: {end_position_transformed}")
                    # print(f"Distance Covered: {distance_covered}")
                    # print(f"Time Elapsed: {time_elapsed}")
                    # print(f"Speed: {speed_km_per_hour}")

                    # Initialize speed history for this track if not exists
                    if track_id not in self.speed_history:
                        self.speed_history[track_id] = []

                    # Add current speed to history
                    self.speed_history[track_id].append(speed_km_per_hour)
                    
                    # Keep only last 5 speed measurements
                    if len(self.speed_history[track_id]) > 5:
                        self.speed_history[track_id].pop(0)
                    
                    # Calculate smoothed speed using moving average
                    smoothed_speed = np.mean(self.speed_history[track_id])

                    # Validate speed
                    if smoothed_speed > self.max_speed:
                        print(f"Warning: Unrealistic speed detected: {smoothed_speed:.2f} km/h")
                        # Cap the speed and adjust distance accordingly
                        smoothed_speed = self.max_speed
                        distance_covered = (smoothed_speed / 3.6) * time_elapsed
                    elif smoothed_speed < self.min_speed:
                        # Ignore very slow movements (likely noise)
                        continue

                    if obj_name not in total_distance_covered:
                        total_distance_covered[obj_name] = {}

                    if track_id not in total_distance_covered[obj_name]:
                        total_distance_covered[obj_name][track_id] = 0

                    total_distance_covered[obj_name][track_id] += distance_covered

                    # Add measurements to all frames in the window
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[obj_name][frame_num_batch]:
                            continue
                        tracks[obj_name][frame_num_batch][track_id]['speed'] = smoothed_speed
                        tracks[obj_name][frame_num_batch][track_id]['distance'] = total_distance_covered[obj_name][track_id]
                        
                        # Visualize distance and speed for the first frame in the window
                        if frame_num_batch == frame_num and frames is not None:
                            frame = frames[frame_num_batch].copy()
                            self._visualize_distance_between_frames(
                                frame,
                                start_position_transformed,
                                end_position_transformed,
                                distance_covered,
                                smoothed_speed
                            )
                            frames[frame_num_batch] = frame

        # Save debug log to file
        if self.debug_log:
            with open('debug_positions.json', 'w') as f:
                json.dump(self.debug_log, f, indent=4)

    def _visualize_distance_between_frames(self, frame, start_pos, end_pos, distance, speed):
        """
        Visualize the distance covered between two frames
        """
        # Draw line between start and end positions
        cv2.line(frame, 
                tuple(map(int, start_pos)), 
                tuple(map(int, end_pos)), 
                (0, 255, 0), 2)
        
        # Draw start and end points
        cv2.circle(frame, tuple(map(int, start_pos)), 5, (0, 255, 0), -1)  # Green for start
        cv2.circle(frame, tuple(map(int, end_pos)), 5, (255, 0, 0), -1)    # Red for end
        
        # Add distance and speed text
        mid_point = tuple(map(int, (np.array(start_pos) + np.array(end_pos)) / 2))
        cv2.putText(frame, f"{distance:.1f}m", 
                   (mid_point[0], mid_point[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"{speed:.1f} km/h", 
                   (mid_point[0], mid_point[1] + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_speed_and_distance(self, frames, tracks):
        """
        Draw speed and distance measurements on frames
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for obj_name, obj in tracks.items():
                if obj_name == 'ball' or obj_name == 'referee':
                    continue

                for _, track_info in obj[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)

                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = list(get_foot_position(bbox))
                        position[1] += 40

                        position = tuple(map(int, position))
                        # Draw speed with background for better visibility
                        cv2.putText(frame, f"{speed:.2f} km/h", position, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                        cv2.putText(frame, f"{speed:.2f} km/h", position, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        # Draw distance with background
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)

        return output_frames
