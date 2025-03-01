from ultralytics import YOLO
import supervision as sv
import pickle
import os


class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

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
    
    
    