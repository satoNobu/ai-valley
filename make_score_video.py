import cv2
import pandas as pd
import os

# パスを絶対パスに修正（Docker内でマウントされる/sharedを使う）
csv_path = "/shared/ball_positions.csv"
frames_dir = "/shared/frames"
video_out_path = "/shared/score_events.mp4"
margin = 2
fps = 5

# CSV読み込みとインデックス展開
df = pd.read_csv(csv_path)
frame_indices = df["frame"].str.extract(r"(\d+)")[0].astype(int).tolist()

expanded_indices = set()
for idx in frame_indices:
    for offset in range(-margin, margin + 1):
        expanded_indices.add(idx + offset)
expanded_indices = sorted(expanded_indices)

# 画像収集
images = []
for idx in expanded_indices:
    fname = f"frame_{idx:04d}.jpg"
    path = os.path.join(frames_dir, fname)
    if os.path.exists(path):
        img = cv2.imread(path)
        images.append(img)

# 動画書き出し
if images:
    height, width, _ = images[0].shape
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for img in images:
        out.write(img)
    out.release()
    print(f"✅ 動画出力完了: {video_out_path}")
else:
    print("❌ 対象のフレーム画像が見つかりませんでした")