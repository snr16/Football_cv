import cv2
import numpy as np

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
        [110,961],
        [231,243],
        [819,227],
        [1454,818]]
        )

        self.target_vertices = np.array(
           [[0,court_width],
           [0,0],
           [court_length,0],
           [court_width,0]])
        
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices,self.target_vertices)

    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))

        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >=0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)

        return transformed_point.reshape(-1,2)
       
       
    def add_transformed_position_to_tracks(self,tracks):
        for obj_name,obj in tracks.items():
          for frame_num,frame in enumerate(obj):
            if obj_name!='ball':
                for track_id,track_info in frame.items():
                    adjusted_position = track_info['adjusted_position']
                    position = np.array(adjusted_position)
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                      position_transformed = position_transformed.squeeze().tolist()

                    tracks[obj_name][frame_num][track_id]['position_transformed'] = position_transformed
            else:
                adjusted_position =frame['adjusted_position']
                position = np.array(adjusted_position)
                position_transformed = self.transform_point(position)

                if position_transformed is not None:
                    position_transformed = position_transformed.squeeze().tolist()

                tracks[obj_name][frame_num]['position_transformed'] = position_transformed

        return tracks