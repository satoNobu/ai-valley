# run_deep_sort.py - boxmotを使って人物追跡（ID付き）を実行

import os
import cv2
import torch
import numpy as np
from glob import glob
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS
from ultralytics import YOLO

# ✅ 設定
FRAME_DIR = "shared/frames"
OUTPUT_DIR = "output"
MODEL_PATH = "yolov8n.pt"
TRACKER_NAME = "bytetrack"
TRACKER_CONFIG_PATH = TRACKER_CONFIGS / f"{TRACKER_NAME}.yaml"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ トラッカー準備
tracker = create_tracker(
    tracker_type=TRACKER_NAME,
    tracker_config=str(TRACKER_CONFIG_PATH),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# ✅ 検出器
yolo = YOLO(MODEL_PATH)

# ✅ フレーム取得
frame_paths = sorted(glob(os.path.join(FRAME_DIR, "*.jpg")))
total = len(frame_paths)
print(f"🔍 {total} フレームを処理します")

# ✅ メインループ
for frame_id, frame_path in enumerate(frame_paths):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"⚠️ 読み込み失敗: {frame_path}")
        continue

    results = yolo(frame)[0]

    # 検出結果を抽出
    detection_list = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        detection_list.append([x1, y1, x2, y2, conf, float(cls_id)])

    # ✅ 配列化 + 安全な形状保証
    if len(detection_list) == 0:
        detections = np.empty((0, 6), dtype=np.float32)
    else:
        detections = np.array(detection_list, dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] != 6:
            print(f"[ERROR] Invalid detections shape: {detections.shape}")
            detections = np.empty((0, 6), dtype=np.float32)
        elif np.isnan(detections).any() or np.isinf(detections).any():
            print("[ERROR] Detections contain NaN or Inf. Skipping frame.")
            detections = np.empty((0, 6), dtype=np.float32)

    # ✅ 確実な ndarray 確認ログ
    print(f"[DEBUG] {frame_id+1}/{total} detections.shape: {detections.shape}, type: {type(detections)}")

    # トラッカーに渡す
    tracks = tracker.update(detections, frame)

    for track in tracks:
        # track = [x1, y1, x2, y2, track_id, ...]
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    save_path = os.path.join(OUTPUT_DIR, os.path.basename(frame_path))
    cv2.imwrite(save_path, frame)

    print(f"🌀 処理中: {frame_id+1}/{total} → {os.path.basename(frame_path)}")

print("✅ 完了しました！トラッキング済画像は output/ に保存されました")