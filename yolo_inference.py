from ultralytics import YOLO
 

model = YOLO('models/best.pt')

results = model.predict(source='input_videos/bundesliga.mp4',save=True)
print(results[0])
print("*"*30)

for box in results[0].boxes:
    print(box)