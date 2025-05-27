import cv2
import os
from datetime import timedelta

video_path = "/shared/input_video.mp4"
output_dir = "/shared/frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
saved_count = 0
interval = int(frame_rate) // 10 # 毎秒1フレーム抽出

while cap.isOpened():
    ret, frame = cap.read()
     # 上下反転補正（必要に応じて）
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    if not ret:
        break
    if frame_count % interval == 0:
        timestamp = str(timedelta(seconds=int(frame_count / frame_rate)))
        cv2.putText(frame, f"Time: {timestamp}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        fname = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(fname, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"✅ 完了：{saved_count}枚のフレームを保存しました。")