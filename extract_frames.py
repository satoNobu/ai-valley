import cv2
import os

# パスはDocker内でマウントされたものに合わせる
video_path = "/shared/input_video.mp4"
output_dir = "/shared/frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
saved_count = 0
interval = int(frame_rate)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % interval == 0:
        frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames.")