from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center_bbox,get_bbox_width
import cv2
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        #Interpolate ball  positions
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() #edge case for first frames

        ball_positions = [{'bbox':x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self,frames):
        detections = []
        batch_size = 20
        for idx in range(0,len(frames),batch_size):
            batch_detections = self.model.predict(frames[idx:idx+batch_size],conf=0.1)
            detections+=batch_detections
        return detections

    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players':[],
            'referees':[],
            'ball':[]
        }

        for frame_num,detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for (k,v) in class_names.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection) #return detection obj

            # Convert goalkeeper to player object
            for obj_idx,class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_idx] = class_names_inv['player']

            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            bboxes = detection_with_tracks.xyxy.tolist() #bboxes from xyxy attribute
            class_ids = detection_with_tracks.class_id.tolist() #class_id
            tracker_ids = detection_with_tracks.tracker_id.tolist()  #tracker id
            class_names= detection_with_tracks.data['class_name'].tolist()  #class names


            for bbox,class_id,track_id,class_name in zip(bboxes,class_ids,tracker_ids,class_names):
                if str(class_name) == 'player': #if detected object is player
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}

                if str(class_name) == 'referee': #if detected object is referee
                    tracks['referees'][frame_num][track_id] = {'bbox':bbox}

            for obj_in_frame in detection_supervision:
                if str(obj_in_frame[5]['class_name']) == 'ball': #if detected object is ball
                    tracks['ball'][frame_num]={'bbox': obj_in_frame[0].tolist()}

            if stub_path is not None:
                with open(stub_path,'wb') as f:
                    pickle.dump(tracks,f)


        return tracks
    
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2=int(bbox[3])
        x_center,_ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_center - rectangle_width//2)
        x2_rect = int(x_center + rectangle_width//2)
        y1_rect = int((y2 - rectangle_height//2) +15)
        y2_rect = int((y2 + rectangle_height//2) +15)

        if track_id:
            cv2.rectangle(frame,
                        (x1_rect,y1_rect),
                        (x2_rect,y2_rect),
                        color,
                        cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text-=10

            text = str(track_id)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            x_center = (x1_rect + x2_rect) // 2 - (text_width // 2)
            y_center = (y1_rect + y2_rect) // 2 + (text_height // 2)

            cv2.putText(frame, 
                        text, 
                        (int(x_center), int(y_center)),  
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
                        fontScale=0.5,
                        color=tuple(255 - c for c in color), 
                        thickness=2)

        return frame


    def draw_traingle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_ = get_center_bbox(bbox)
        traingle_points=np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])

        cv2.drawContours(frame,[traingle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[traingle_points],0,(0,0,0),2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw semi transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,1000),(255,255,255),cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #get number of time each team had ball control
        team_1_num_frames = (team_ball_control_till_frame[team_ball_control_till_frame==1]).shape[0]
        team_2_num_frames = len(team_ball_control_till_frame) - team_1_num_frames
        
        team_1 = (team_1_num_frames)/(team_1_num_frames+team_2_num_frames)
        team_2 = 1-team_1

        cv2.putText(frame,f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 1 Ball Control: {team_2*100:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame

    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video_frames = []

        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            #Draw players
            for track_id,player in player_dict.items():
                color = player.get('team_color',(0,0,255))
                frame = self.draw_ellipse(frame,player['bbox'],color,track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame,player['bbox'],(0,0,255))
        
            #Draw referees
            for track_id,referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee['bbox'],(0,0,0),None)

            #Draw ball
            if ball_dict:
                frame = self.draw_traingle(frame,ball_dict['bbox'],(0,255,0))

            #Draw Team ball control
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)
            

            output_video_frames.append(frame)
            
        return output_video_frames
