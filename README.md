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

### 2. 吐き出したCSVから得点した部分の動画を抜き出し
docker run --rm -v $(pwd)/shared:/shared video-frame-yolo python make_score_video.py