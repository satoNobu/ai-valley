FROM python:3.10-slim

# 必要なライブラリのインストール（OpenCV依存含む）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir opencv-python

# 作業ディレクトリを設定
WORKDIR /app

# スクリプトをコピー
COPY extract_frames.py ./

# 実行コマンド
CMD ["python", "extract_frames.py"]