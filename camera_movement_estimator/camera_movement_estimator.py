import cv2
import numpy as np
import pickle 
import sys,os
sys.path.append('../')
from utils import measure_xy_distance,measure_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def adjust_position_to_tracks(self,tracks,camera_movement_per_frame):
         
         for obj_name,obj in tracks.items():
                for frame_num,frame in enumerate(obj):
                    if obj_name!='ball':
                        for track_id,track_info in frame.items():
                           position = track_info['position']
                           camera_movement = camera_movement_per_frame[frame_num]
                           adjusted_position = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                           tracks[obj_name][frame_num][track_id]['adjusted_position'] = adjusted_position
                    else:
                        position = frame['position']
                        camera_movement = camera_movement_per_frame[frame_num]
                        adjusted_position = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                        tracks[obj_name][frame_num]['adjusted_position'] = adjusted_position

    def get_camera_movement(self,frames,read_from_stub=False,stub_path=None):

        #read the camera movement from stub path
        if stub_path and read_from_stub and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
            
        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY) #old gray iamge
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num,frame in enumerate(frames[1:],1):
            
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            new_features,_,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_distance = 0
            camera_x_movement,camera_y_movement = (0,0)

            for (old,new) in zip(old_features,new_features):
                old = old.ravel()
                new = new.ravel()

                distance = measure_distance(old,new)

                if distance > max_distance:
                    camera_x_movement,camera_y_movement = measure_xy_distance(old,new)
                    max_distance = distance

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_x_movement,camera_y_movement]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()

        if stub_path:     
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement


    def draw_camera_movement(self,frames,camera_movement_per_frame):

        output_frames = []
        for frame_num,frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()

            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha=0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            camera_x,camera_y = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement x: {camera_x:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement y: {camera_y:.2f}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame)

        return output_frames



                


