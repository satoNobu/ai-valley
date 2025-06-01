import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 設定
IMG_SIZE = (64, 64)
MAX_FRAMES = 100
MODEL_PATH = "/shared/spike_action_model_lstm.h5"
FRAME_DIR = "/shared/frames"
OUTPUT_DIR = "/shared/spike_frames"
CLASSES = ['spike', 'receive', 'block']
THRESHOLD = 0.6
PRE_FRAMES = 10
POST_FRAMES = 30
SKIP_AFTER_DETECTION = 30  # 次の検出を抑制するフレーム数

# モデル読み込み
model = load_model(MODEL_PATH)

# 出力ディレクトリ準備
os.makedirs(OUTPUT_DIR, exist_ok=True)

# フレーム画像を取得しソート
frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])

# 各フレームを1枚ずつスライドしながら判定
detected_frames = []
i = 0
while i < len(frame_files) - MAX_FRAMES:
    batch_files = frame_files[i:i + MAX_FRAMES]
    frames = []
    for fname in batch_files:
        img_path = os.path.join(FRAME_DIR, fname)
        img = load_img(img_path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        frames.append(arr)

    clip = np.expand_dims(np.stack(frames, axis=0), axis=0)
    preds = model.predict(clip, verbose=0)[0]
    pred_class = CLASSES[np.argmax(preds)]
    confidence = np.max(preds)

    if pred_class == 'spike' and confidence >= THRESHOLD:
        detected_frames.append(i + MAX_FRAMES // 2)  # 中央フレームを記録
        i += SKIP_AFTER_DETECTION
    else:
        i += 1

# スパイクごとにclip出力
clip_index = 0
used = set()
for center in detected_frames:
    start = max(center - PRE_FRAMES, 0)
    end = min(center + POST_FRAMES, len(frame_files))

    # 重複検出を避ける
    if any(f in used for f in range(start, end)):
        continue

    for f in range(start, end):
        used.add(f)

    clip_dir = os.path.join(OUTPUT_DIR, f"clip_{clip_index:04d}")
    os.makedirs(clip_dir, exist_ok=True)
    for f in range(start, end):
        fname = frame_files[f]
        src = os.path.join(FRAME_DIR, fname)
        dst = os.path.join(clip_dir, fname)
        cv2.imwrite(dst, cv2.imread(src))
    clip_index += 1

print(f"✅ {clip_index}件のスパイク動作を検出・分割しました！")