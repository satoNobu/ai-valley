# .gitignore に shared の中身を無視させるには以下を記述：
# /shared/*
# !/shared/.gitkeep

# extract_frames.py
import cv2
import os
import csv
from datetime import timedelta
from ultralytics import YOLO

# === 入出力設定 ===
video_path = "/shared/input_video.mp4"
output_dir = "/shared/frames"
os.makedirs(output_dir, exist_ok=True)

csv_path = "/shared/ball_positions.csv"

# === 検出設定 ===
YOLO_MODEL = "yolov8m.pt"  # yolov8n.pt（軽量）、yolov8m.pt（中間）、yolov8l.pt（高精度）から選択可能
BALL_CONF_THRESHOLD = 0.25  # ボール検出の信頼度しきい値（低いほど多くのボールが検出される）
CLASSES = [0, 32]  # 0=person, 32=sports ball、検出対象クラスを限定して処理を高速化

# スコア判定のパラメータ
MIN_DY_FOR_SCORE = 20  # ボールの最小垂直移動量（下向き）
MIN_CONFIDENCE = 0.35  # ボールの最小信頼度スコア
MAX_Y_POS_RATIO = 0.85  # 画面の下から何%の位置までをコート面とみなすか（0.85 = 下から85%）

# === YOLOモデルのロード ===
model = YOLO(YOLO_MODEL)
print(f"🔍 YOLOモデル {YOLO_MODEL} を読み込みました", flush=True)

# === 動画読み込み ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 動画ファイルが開けませんでした", flush=True)
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
saved_count = 0
# 1秒間に何フレーム処理するか（小さくするとより多くのフレームを処理）
interval = int(frame_rate) // 2  # 0.5秒ごとにフレーム処理（より高頻度にボール検出）

rows = []

# === フレーム抽出・タイムスタンプ描画・YOLO推論 ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        # タイムスタンプの計算
        timestamp_sec = int(frame_count / frame_rate)
        time_str = str(timedelta(seconds=timestamp_sec))

        # タイムスタンプをフレームに描画
        cv2.putText(
            frame,
            f"Time: {time_str}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        frame_name = f"frame_{saved_count:04d}.jpg"
        frame_filename = os.path.join(output_dir, frame_name)

        # === YOLOによる物体検出 ===
        results = model(frame, conf=BALL_CONF_THRESHOLD, classes=CLASSES)  # 信頼度しきい値とクラス指定
        boxes = results[0].boxes
        class_names = model.model.names

        print(f"[Frame {frame_name}] Detected {len(boxes)} objects.", flush=True)
        
        # フレームの高さと幅を取得（スコア判定に使用）
        frame_height, frame_width = frame.shape[:2]
        ground_level = frame_height * MAX_Y_POS_RATIO  # 地面と判定するY座標のしきい値

        if boxes is not None:
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                class_id = int(cls.item())
                confidence = float(conf.item())
                name = class_names[class_id]
                print(f" - Detected class: {name} (conf: {confidence:.2f})", flush=True)

                if name == "sports ball":
                    x1, y1, x2, y2 = map(int, box[:4])
                    # ボールの中心座標を計算
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 描画
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"ball {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    
                    # 地面に近いかどうかの判定用の線を描画
                    cv2.line(frame, (0, int(ground_level)), (frame_width, int(ground_level)), (255, 0, 0), 1)
                    
                    # CSV用データを蓄積（地面に近いかどうかのフラグも追加）
                    is_near_ground = center_y >= ground_level
                    rows.append([frame_name, time_str, x1, y1, x2, y2, confidence, center_x, center_y, is_near_ground])

        # フレーム処理の最後にスコア判定のためのデータを一時保存（オプショナル）
        if len(rows) > 0 and rows[-1][0] == frame_name:  # 最後に追加したデータが現在のフレームの場合
            # 前のフレームと比較して仮のスコア判定を行う（視覚的確認用）
            if len(rows) > 1:
                prev_center_y = rows[-2][8]  # 前のフレームのcenter_y
                curr_center_y = rows[-1][8]  # 現在のフレームのcenter_y
                dy = curr_center_y - prev_center_y
                is_near_ground = rows[-1][9]  # is_near_ground
                confidence = rows[-1][6]  # confidence
                
                # 仮のスコア判定（CSVのスコア判定と同じロジック）
                if dy > MIN_DY_FOR_SCORE and is_near_ground and confidence > MIN_CONFIDENCE:
                    # スコア判定されたフレームには特別な表示を追加
                    cv2.putText(
                        frame,
                        "SCORE DETECTED!",
                        (frame_width // 2 - 150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3
                    )
        
        # バウンディングボックス付き画像として保存
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

# === CSV保存と is_score 列の付加 ===
import pandas as pd
columns = ["frame", "timestamp", "x1", "y1", "x2", "y2", "confidence", "center_x", "center_y", "is_near_ground"]
df = pd.DataFrame(rows, columns=columns)
if not df.empty:
    df["frame_index"] = df["frame"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("frame_index").reset_index(drop=True)
    
    # ボールの垂直方向の移動を計算
    df["y_center"] = df["center_y"]
    df["dy"] = df["y_center"].diff()
    
    # 前のフレームと比較して、ボールが下降していて（dy>0）、地面に近く（is_near_ground=True）、
    # 信頼度が一定以上（confidence>MIN_CONFIDENCE）で、垂直移動量が一定以上（dy>MIN_DY_FOR_SCORE）の場合にスコアと判定
    df["is_downward"] = df["dy"] > MIN_DY_FOR_SCORE
    
    # スコア判定: 下降移動中 AND 地面付近 AND 十分な信頼度
    df["is_score"] = df["is_downward"] & df["is_near_ground"] & (df["confidence"] > MIN_CONFIDENCE)
    
    # スコア判定の連続を防ぐ（同じスコアイベントが複数回カウントされないように）
    # 同じスコアイベントが連続している場合は、最初のものだけをTrueにする
    df["score_group"] = (df["is_score"] != df["is_score"].shift(1)).cumsum()
    df["is_first_in_group"] = ~df["is_score"].duplicated(subset=["score_group"])
    df["is_score"] = df["is_score"] & df["is_first_in_group"]
    
    # 不要な列を削除
    df.drop(columns=["frame_index", "score_group", "is_first_in_group"], inplace=True)
    
    df.to_csv(csv_path, index=False)
    print(f"📝 CSV出力完了: {csv_path}（改良したスコア判定付き）", flush=True)
    
    # スコア数をカウント
    score_count = df["is_score"].sum()
    print(f"🏆 検出されたスコア数: {score_count}", flush=True)
else:
    print("⚠️ ボールが検出されなかったため CSV は生成されません", flush=True)

print(f"✅ Saved {saved_count} frames with timestamps and YOLO detections (person + sports ball).", flush=True)
