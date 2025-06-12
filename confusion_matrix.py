# ディレクトリ構成:
# dataset/
# ├── spike/
# │   ├── 1/（← ファイル名自由でOK）
# │   │   ├── frame_0190.jpg
# │   │   ├── frame_0191.jpg
# │   └── 2/
# ├── receive/
# └── block/

# ディレクトリ構成:
# dataset/
# ├── spike/
# ├── receive/
# ├── block/
# ├── none/
# └── serve/

# confusion_matrix.py - 正例 vs 他すべて（マルチ負例）対応版

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Masking
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 共通設定
IMG_SIZE = (64, 64)
MAX_FRAMES = 20
DATASET_DIR = 'dataset'
MODEL_SAVE_DIR = '/shared'
TARGET_CLASSES = ['spike', 'block', 'receive', 'serve']

# モデル作成関数
def create_model(output_classes):
    model = Sequential([
        TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(MAX_FRAMES, *IMG_SIZE, 3)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        Masking(mask_value=0.0),
        LSTM(32),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# clip読み込み
def load_clip(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    frames = []
    for path in frame_paths[:MAX_FRAMES]:
        img = load_img(path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        frames.append(arr)
    if not frames:
        return None
    pad_frame = np.zeros_like(frames[0])
    while len(frames) < MAX_FRAMES:
        frames.append(pad_frame)
    return np.stack(frames, axis=0)

# データセット読み込み（正例 vs 他すべて）
def load_dataset_for_class(target_class):
    clips = []
    labels = []
    all_classes = os.listdir(DATASET_DIR)
    for label in all_classes:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for clip_name in os.listdir(label_dir):
            clip_path = os.path.join(label_dir, clip_name)
            clip = load_clip(clip_path)
            if clip is not None:
                label_val = 1 if label == target_class else 0
                clips.append(clip)
                labels.append(label_val)
    return np.array(clips), tf.keras.utils.to_categorical(labels, num_classes=2)

# 各モデルの学習と保存
for action in TARGET_CLASSES:
    print(f"\n🧠 モデル: {action} vs all others")
    X, y = load_dataset_for_class(action)
    model_path = os.path.join(MODEL_SAVE_DIR, f"{action}_model.h5")

    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"✅ {action}_model 読み込み済")
    else:
        model = create_model(2)
        print(f"🆕 {action}_model 新規作成")

    model.fit(X, y, epochs=10, batch_size=4)
    model.save(model_path)
    print(f"💾 {action}_model 保存済 → {model_path}")