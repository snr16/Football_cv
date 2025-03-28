from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssinger
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
import os

def setup_calibration_dir():
    """
    Create calibration directory and return its path
    """
    calibration_dir = 'calibration_results'
    os.makedirs(calibration_dir, exist_ok=True)
    return calibration_dir

def process_video(input_path, output_path):
    """
    Process a football video to track players, ball, and generate analytics
    Args:
        input_path: Path to input video file
        output_path: Path to save processed video
    """
    try:
        # Setup calibration directory
        calibration_dir = setup_calibration_dir()
        
        # Read video frames
        print("Reading video...")
        video_frames = read_video(input_path)
        if not video_frames:
            raise ValueError("No frames read from video")

        # Initialize and run object tracking
        print("Initializing tracker...")
        tracker = Tracker("models/best.pt")
        tracks = tracker.get_object_tracks(video_frames,
                                         read_from_stub=True,
                                         stub_path='stubs/track_stubs.pkl')
        
        print("Adjusting tracks...")
        tracks = tracker.adjust_tracks(tracks)
        tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
        tracker.add_position_to_tracks(tracks)

        # Process camera movement
        print("Processing camera movement...")
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames,
            True,
            'stubs/camera_movement_stub.pkl'
        )
        camera_movement_estimator.adjust_position_to_tracks(tracks, camera_movement_per_frame)

        # Initialize view transformer with hardcoded values
        print("Setting up view transformation...")
        view_transformer = ViewTransformer()
        view_transformer.create_visualization(video_frames[0], calibration_dir)
        
        # Transform tracks to top-down view
        tracks = view_transformer.add_transformed_position_to_tracks(tracks)
        
        # Initialize speed and distance estimator
        print("Setting up speed and distance measurements...")
        speed_distance_estimator = SpeedAndDistanceEstimator()
        speed_distance_estimator.add_speed_and_distance_to_tracks(tracks, video_frames)

        # Generate additional visualizations
        print("Generating additional visualizations...")
        view_transformer.visualize_trajectory(tracks, video_frames[0], calibration_dir)
        view_transformer.visualize_heatmap(tracks, video_frames[0], calibration_dir)

        # Process team assignments
        print("Processing team assignments...")
        team_assigner = TeamAssinger()
        team_assigner.assign_team_color(video_frames[60], tracks['players'][60])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                player_team_id = team_assigner.assign_player_team(
                    video_frames[frame_num],
                    player_id,
                    track['bbox']
                )
                player_color = team_assigner.team_colors[player_team_id]
                tracks['players'][frame_num][player_id]['team'] = player_team_id
                tracks['players'][frame_num][player_id]['team_color'] = player_color

        # Process ball possession
        print("Processing ball possession...")
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        team_ball_control = np.array(team_ball_control)

        # Generate output video with annotations
        print("Generating output video...")
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        output_video_frames = camera_movement_estimator.draw_camera_movement(
            output_video_frames, 
            camera_movement_per_frame
        )
        output_video_frames = speed_distance_estimator.draw_speed_and_distance(
            output_video_frames, 
            tracks
        )

        # Save the processed video
        print(f"Saving video to {output_path}...")
        save_video(output_video_frames, output_path)
        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

def main():
    # Define input and output paths
    input_video = 'input_videos/bundesliga.mp4'
    output_video = 'output_videos/output_video.mp4'

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # Process the video
    process_video(input_video, output_video)

if __name__ == "__main__":
    main()