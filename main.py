from utils import read_video,save_video
from trackers import Tracker

def main():
   
   #read the video file
    video_frames = read_video('input_videos/bundesliga.mp4')

    # Initialize tracker
    tracker = Tracker("models/best.pt")

    # Get tracker objects
    tracks = tracker.get_object_tracks(video_frames,
                              read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl')
    
    # Draw annotaions
    video_frames = tracker.draw_annotations(video_frames,tracks)

    #save the video file
    save_video(video_frames,'output_videos/output_video.mp4')

if __name__ == "__main__":
    main()