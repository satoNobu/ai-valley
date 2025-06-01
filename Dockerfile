FROM python:3.10-slim

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && apt-get clean

# Pythonライブラリのインストール
RUN pip install --no-cache-dir opencv-python ultralytics tensorflow

# 作業ディレクトリを指定
WORKDIR /app

# スクリプトをコピー
COPY extract_frames.py ./
COPY confusion_matrix.py ./
COPY predict_frames.py ./

# 必要に応じてデータセットもコピーしたい場合はこちら（今回は sharedマウントでOK）

# YOLOv8モデルをダウンロード（必要なら）
RUN curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt || echo "⬇️ モデルURLは必要に応じて更新してください"

# デフォルトは extract_frames.py を実行（run時に上書き可）
CMD ["python", "extract_frames.py"]