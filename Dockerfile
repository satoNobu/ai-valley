# Dockerfile
FROM python:3.10-slim

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && apt-get clean

# Pythonライブラリのインストール
RUN pip install --no-cache-dir opencv-python ultralytics

# 作業ディレクトリを指定
WORKDIR /app

# スクリプトをコピー
COPY extract_frames.py ./
COPY make_score_video.py ./

# YOLOv8モデルをダウンロード（初回のみ）
RUN curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt || echo "⬇️ モデルURLは必要に応じて更新してください"

# デフォルトの実行コマンド（あとで書き換えてもOK）
CMD ["python", "extract_frames.py"]