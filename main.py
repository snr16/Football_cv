from utils import read_video,save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssinger
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
   
   #read the video file
    video_frames = read_video('input_videos/bundesliga.mp4')

    # Initialize tracker
    tracker = Tracker("models/best.pt")

    # Get tracker objects
    tracks = tracker.get_object_tracks(video_frames,
                              read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl')
    

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])


    # Add positions to tracks 
    tracker.add_position_to_tracks(tracks)

    # Get camera movement estimate
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                True,
                                                'stubs/camera_movement_stub.pkl')
    

    # Adjust player camera movement
    camera_movement_estimator.adjust_position_to_tracks(tracks,camera_movement_per_frame)

    # Perspective homography or transformation
    view_transformer = ViewTransformer()
    tracks = view_transformer.add_transformed_position_to_tracks(tracks)

    #Speed Distance Estimator
    speed_distance_estimator = SpeedAndDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign player Teams
    team_assigner = TeamAssinger()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num,player_track in enumerate(tracks['players']):
        for player_id,track in player_track.items():
            player_team_id = team_assigner.assign_player_team(video_frames[frame_num],
                                                              player_id,
                                                              track['bbox'])
            player_color = team_assigner.team_colors[player_team_id]
            tracks['players'][frame_num][player_id]['team']=player_team_id
            tracks['players'][frame_num][player_id]['team_color']=player_color

    # Assign Ball Acquistion
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # Draw annotaions
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Draw speed and distance
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    #save the video file
    save_video(output_video_frames,'output_videos/output_video.mp4')

if __name__ == "__main__":
    main()