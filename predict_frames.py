# predict_frames.py - 複数モデルによるマルチ動作検出（spike, block, receive, serve）対応

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 設定
IMG_SIZE = (64, 64)
MAX_FRAMES = 10
FRAME_DIR = "/shared/frames"
OUTPUT_DIR = "/shared/action_frames"
MODELS = {
    "spike": "/shared/spike_model.h5",
    "block": "/shared/block_model.h5",
    "receive": "/shared/receive_model.h5",
    "serve": "/shared/serve_model.h5",
}
THRESHOLD = 0.4
PRE_FRAMES = 10
POST_FRAMES = 30
SKIP_AFTER_DETECTION = 30

# モデル読み込み
loaded_models = {k: load_model(v) for k, v in MODELS.items()}

# 出力先フォルダ作成
for cls in MODELS.keys():
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# フレーム取得
frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith(".jpg")])

# 検出記録
detected_frames = {cls: [] for cls in MODELS.keys()}
i = 0
while i < len(frame_files) - MAX_FRAMES:
    if i % 50 == 0:
        print(f"🌀 Processing frame {i}/{len(frame_files)} ...", flush=True)

    batch_files = frame_files[i:i + MAX_FRAMES]
    frames = []
    for fname in batch_files:
        img_path = os.path.join(FRAME_DIR, fname)
        img = load_img(img_path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        frames.append(arr)

    clip = np.expand_dims(np.stack(frames, axis=0), axis=0)

    hit = False
    for cls, model in loaded_models.items():
        preds = model.predict(clip, verbose=0)[0]
        confidence = preds[1]  # index 1 = targetクラス（2値分類 [not_class, class]）
        if confidence >= THRESHOLD:
            detected_frames[cls].append(i + MAX_FRAMES // 2)
            print(f"✅ Detected {cls} ({confidence:.2f}) at frame {i}", flush=True)
            hit = True

    i += SKIP_AFTER_DETECTION if hit else 1

# clip保存
for cls, centers in detected_frames.items():
    clip_index = 0
    used = set()
    for center in centers:
        start = max(center - PRE_FRAMES, 0)
        end = min(center + POST_FRAMES, len(frame_files))

        if any(f in used for f in range(start, end)):
            continue

        for f in range(start, end):
            used.add(f)

        clip_dir = os.path.join(OUTPUT_DIR, cls, f"clip_{clip_index:04d}")
        os.makedirs(clip_dir, exist_ok=True)
        for f in range(start, end):
            fname = frame_files[f]
            src = os.path.join(FRAME_DIR, fname)
            dst = os.path.join(clip_dir, fname)
            cv2.imwrite(dst, cv2.imread(src))
        clip_index += 1

print("✅ 全モデルによるclip抽出が完了しました")