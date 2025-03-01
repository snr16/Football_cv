from ultralytics import YOLO
import torch
import numpy as np
import json



model = YOLO('models/best.pt')

results = model.predict(source='input_videos/bundesliga.mp4',save=True)
print(results[0])

print("*"*30)

for box in results[0].boxes:
    print(box)
    break

def tensor_to_json(obj):

    if isinstance(obj, torch.Tensor):
        obj = obj.tolist() 
    
    if isinstance(obj, np.ndarray):
        save_name='sample_frame.npy'
        np.save(save_name, obj)  
        return save_name
    
    if isinstance(obj, (list, tuple)):
        return [tensor_to_json(item) for item in obj]
    
    if isinstance(obj, dict):
        return {k: tensor_to_json(v) for k, v in obj.items()}
    
    if hasattr(obj, "__dict__"):
        return {attr: tensor_to_json(getattr(obj, attr))
                for attr in dir(obj)
                if not attr.startswith("_") and not callable(getattr(obj, attr))}
    
    return obj

frame_boxes_json = tensor_to_json(vars(results[0]))

with open('sample_frame_detection.json', 'w') as file:
    json.dump(frame_boxes_json, file, indent=4)
