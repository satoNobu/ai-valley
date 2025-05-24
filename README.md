# ai-valley
ソフトバレーの動画解析

# 
docker build -t video-frame-extractor .

## 配置
video_frame_extractor/
├── Dockerfile
├── requirements.txt
├── extract_frames.py
└── shared/
    ├── input_video.mp4      ← 動画をここに置く
    └── frames/              ← 抽出画像がここに出力される