# run_deep_sort.py - boxmotを使って人物追跡（ID付き）を実行＋トラッキング結果をJSONに保存

import os
import cv2
import torch
import numpy as np
import json
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
TRACK_JSON_PATH = os.path.join(OUTPUT_DIR, "track_results.json")

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

# ✅ トラッキング結果保存用
tack_results_dict = {}

# ✅ メインループ
for frame_id, frame_path in enumerate(frame_paths):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"⚠️ 読み込み失敗: {frame_path}")
        continue

    results = yolo(frame)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append([x1, y1, x2, y2, conf, float(cls_id)])

    detections = np.array(detections, dtype=np.float32)
    if detections.size == 0:
        detections = np.empty((0, 6), dtype=np.float32)

    # トラッキング実行
    tracks = tracker.update(detections, frame)

    # フレーム名
    frame_name = os.path.basename(frame_path)
    tack_results_dict[frame_name] = []

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        cls_id = int(track[5])
        tack_results_dict[frame_name].append({
            "id": track_id,
            "bbox": [x1, y1, x2, y2],
            "cls": cls_id
        })

        # 可視化
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    save_path = os.path.join(OUTPUT_DIR, frame_name)
    cv2.imwrite(save_path, frame)

    print(f"🌀 処理中: {frame_id+1}/{total} → {frame_name}")

# ✅ JSON保存
with open(TRACK_JSON_PATH, "w") as f:
    json.dump(tack_results_dict, f, indent=2)

print("✅ 完了しました！トラッキング済画像とtrack_results.jsonを出力しました")