# .gitignore に shared の中身を無視させるには以下を記述：
# /shared/*
# !/shared/.gitkeep

# extract_frames.py
import cv2
import os
from datetime import timedelta
from ultralytics import YOLO

# === 入出力設定 ===
video_path = "/shared/input_video.mp4"
output_dir = "/shared/frames"
os.makedirs(output_dir, exist_ok=True)

# === YOLOモデルのロード（yolov8n を使用） ===
model = YOLO("yolov8n.pt")  # 事前にこのモデルをダウンロードしておくこと

# === 動画読み込み ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 動画ファイルが開けませんでした")
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
saved_count = 0
interval = int(frame_rate)

# === フレーム抽出・タイムスタンプ描画・YOLO推論 ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        # タイムスタンプの計算
        timestamp_sec = int(frame_count / frame_rate)
        time_str = str(timedelta(seconds=timestamp_sec))

        # タイムスタンプをフレームに描画
        cv2.putText(
            frame,
            f"Time: {time_str}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # === YOLOによる物体検出 ===
        results = model(frame)
        detections = results[0].boxes

        # sports ball のみ抽出
        for box, cls in zip(detections.xyxy, detections.cls):
            class_id = int(cls.item())
            if model.names[class_id] == "sports ball":
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "sports ball",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

        # バウンディングボックス付き画像として保存
        frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Saved {saved_count} frames with timestamps and YOLO detections (person + sports ball).")
