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

# YOLOv8モデルをダウンロード（最新版モデルURLを指定）
RUN curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt || echo "⬇️ モデルURLは必要に応じて更新してください"

# 実行コマンド
CMD ["python", "extract_frames.py"]