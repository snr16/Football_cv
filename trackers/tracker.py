from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
