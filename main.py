from utils import read_video,save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssinger
from player_ball_assigner import PlayerBallAssigner

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
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

    # Draw annotaions
    video_frames = tracker.draw_annotations(video_frames,tracks)


    #save the video file
    save_video(video_frames,'output_videos/output_video.mp4')

if __name__ == "__main__":
    main()