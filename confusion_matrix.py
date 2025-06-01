# ディレクトリ構成:
# dataset/
# ├── spike/
# │   ├── 1/（← ファイル名自由でOK）
# │   │   ├── frame_0190.jpg
# │   │   ├── frame_0191.jpg
# │   └── 2/
# ├── receive/
# └── block/

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Masking
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 読み込み設定
IMG_SIZE = (64, 64)
MAX_FRAMES = 20
CLASSES = ['spike', 'receive', 'block']
DATASET_DIR = 'dataset'
MODEL_PATH = '/shared/spike_action_model_lstm.h5'

# パディング付きのclip読み込み（フレーム名自由対応）
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

# データセット読み込み
def load_dataset():
    clips = []
    labels = []
    for label_idx, label in enumerate(CLASSES):
        label_dir = os.path.join(DATASET_DIR, label)
        for clip_name in os.listdir(label_dir):
            clip_path = os.path.join(label_dir, clip_name)
            clip = load_clip(clip_path)
            if clip is not None:
                clips.append(clip)
                labels.append(label_idx)
    return np.array(clips), tf.keras.utils.to_categorical(labels, num_classes=len(CLASSES))

X, y = load_dataset()  # shape: (N, MAX_FRAMES, 64, 64, 3)

# 既存モデルがあれば読み込む、なければ新規作成
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ 既存モデルを読み込みました")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 追学習可能にするため再コンパイル
else:
    model = Sequential([
        TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(MAX_FRAMES, *IMG_SIZE, 3)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        Masking(mask_value=0.0),
        LSTM(32),
        Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("🆕 新しいモデルを作成しました")

# モデル学習（積み重ね型）
model.fit(X, y, epochs=10, batch_size=4)

# モデル保存（外部共有フォルダに出力）
model.save(MODEL_PATH)
print("💾 モデルを保存しました：", MODEL_PATH)