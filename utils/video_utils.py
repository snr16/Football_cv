import cv2

def read_video(input_video_path):
    frames = []
    cap = cv2.VideoCapture(input_video_path)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

import cv2

def save_video(frames, output_path):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    for frame in frames:
        out.write(frame)

    out.release()
