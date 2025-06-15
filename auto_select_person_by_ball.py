# auto_select_person_by_ball.py

import os
import cv2
import numpy as np
from ultralytics import YOLO

# 入力画像パス
INPUT_IMAGE = "/shared/frames/frame_0111.jpg"
OUTPUT_IMAGE = "/shared/selected_person_by_ball.jpg"

# モデル読み込み（YOLOv8n を使用）
model = YOLO("yolov8n.pt")

# 画像読み込み
image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise FileNotFoundError(f"画像が見つかりません: {INPUT_IMAGE}")
h, w, _ = image.shape

# 推論
results = model(image)[0]

# 人物とボールの検出結果を抽出
people = []
balls = []

for box in results.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    label = model.names[cls_id]

    if label == "person":
        people.append((xyxy, conf))
    elif "ball" in label:
        balls.append((xyxy, conf))

# ボールと人物が両方いないときは終了
if not people or not balls:
    print("🛑 ボールまたは人物が検出されませんでした。")
    exit()

# 最もスコアの高いボールを基準にする
ball_xyxy, _ = max(balls, key=lambda x: x[1])
ball_cx = (ball_xyxy[0] + ball_xyxy[2]) // 2
ball_cy = (ball_xyxy[1] + ball_xyxy[3]) // 2
ball_center = np.array([ball_cx, ball_cy])

# 各人物の中心点との距離を計算
closest_person = None
min_dist = float("inf")
for xyxy, conf in people:
    cx = (xyxy[0] + xyxy[2]) // 2
    cy = (xyxy[1] + xyxy[3]) // 2
    person_center = np.array([cx, cy])
    dist = np.linalg.norm(ball_center - person_center)

    if dist < min_dist:
        min_dist = dist
        closest_person = xyxy

# 選ばれた人物に枠線を描画
x1, y1, x2, y2 = closest_person
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(image, "Selected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# ボールの位置も描画（デバッグ用）
cv2.circle(image, tuple(ball_center), 5, (0, 0, 255), -1)
cv2.putText(image, "Ball", (ball_cx + 5, ball_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 保存
cv2.imwrite(OUTPUT_IMAGE, image)
print(f"✅ 画像を保存しました: {OUTPUT_IMAGE}")