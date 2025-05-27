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

## 実行
### 1. 動画の検出
docker run --rm -v $(pwd)/shared:/shared video-frame-yolo