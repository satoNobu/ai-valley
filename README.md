# ai-valley
ソフトバレーの動画解析

# ビルド
docker build -t video-frame-yolo .

## 配置
video_frame_extractor/
├── Dockerfile
├── requirements.txt
├── extract_frames.py
└── shared/
    ├── input_video.mp4      ← 動画をここに置く
    └── frames/              ← 抽出画像がここに出力される
 dataset/　　　　　　　　　　　　← 学習データ
 ├── spike/
 │   ├── 1/（← ファイル名自由でOK）
 │   │   ├── frame_0190.jpg
 │   │   ├── frame_0191.jpg
 │   └── 2/
 ├── receive/
 └── block/

## 実行
### 1. 動画の検出
input_video.mp4という動画でshared配下に配置
以下のコマンドを実行

docker run --rm -v $(pwd)/shared:/shared video-frame-yolo


### 2. スパイクなどの学習
docker run --rm \
  -v $(pwd)/shared:/shared \
  -v $(pwd)/dataset:/app/dataset \
  video-frame-yolo \
  python confusion_matrix.py

### 3. 自動で抽出
docker run --rm \
  -v $(pwd)/shared:/shared \
  video-frame-yolo \
  python predict_frames.py