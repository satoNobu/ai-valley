# .gitignore に shared の中身を無視させるには以下を記述：
# /shared/*
# !/shared/.gitkeep

# extract_frames.py
import cv2
import os
import csv
from datetime import timedelta
from ultralytics import YOLO

# === 入出力設定 ===
video_path = "/shared/input_video.mp4"
output_dir = "/shared/frames"
os.makedirs(output_dir, exist_ok=True)

csv_path = "/shared/ball_positions.csv"

# === YOLOモデルのロード（yolov8n を使用） ===
model = YOLO("yolov8n.pt")

# === 動画読み込み ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 動画ファイルが開けませんでした", flush=True)
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
saved_count = 0
interval = int(frame_rate)

with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp", "x1", "y1", "x2", "y2"])

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

            frame_name = f"frame_{saved_count:04d}.jpg"
            frame_filename = os.path.join(output_dir, frame_name)

            # === YOLOによる物体検出 ===
            results = model(frame)
            boxes = results[0].boxes
            class_names = model.model.names

            print(f"[Frame {frame_name}] Detected {len(boxes)} objects.", flush=True)

            if boxes is not None:
                for box, cls in zip(boxes.xyxy, boxes.cls):
                    class_id = int(cls.item())
                    name = class_names[class_id]
                    print(f" - Detected class: {name}", flush=True)

                    if name == "sports ball":
                        x1, y1, x2, y2 = map(int, box[:4])
                        # 描画
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, "sports ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        # CSVに出力
                        print(f"   → Writing to CSV: {frame_name}, {time_str}, {x1}, {y1}, {x2}, {y2}", flush=True)
                        csv_writer.writerow([frame_name, time_str, x1, y1, x2, y2])

            # バウンディングボックス付き画像として保存
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

cap.release()
print(f"✅ Saved {saved_count} frames with timestamps and YOLO detections (person + sports ball).", flush=True)
print(f"📝 Ball positions saved to: {csv_path}", flush=True)