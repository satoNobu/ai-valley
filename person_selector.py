import cv2
import os
from glob import glob
from ultralytics import YOLO

# 設定
FRAME_DIR = "shared/frames"
OUTPUT_DIR = "shared/cropped_persons"  # 保存先ルート（分類ディレクトリに分岐）
MODEL_PATH = "shared/yolov8n.pt"
MAX_CLIP = 10  # 前後合わせて10フレーム
os.makedirs(OUTPUT_DIR, exist_ok=True)

# クラス分類キー
CLASS_KEYS = {
    ord('a'): "spike",
    ord('b'): "block",
    ord('c'): "receive",
    ord('d'): "serve",
    ord('e'): "tosu",
    ord('f'): "none"
}

# モデル読み込み
model = YOLO(MODEL_PATH)

# フレーム一覧
frame_paths = sorted(glob(os.path.join(FRAME_DIR, "*.jpg")))
frame_index = 0
clip_counter = 0

while 0 <= frame_index < len(frame_paths):
    frame_path = frame_paths[frame_index]
    image = cv2.imread(frame_path)
    if image is None:
        print(f"⚠️ 読み込み失敗: {frame_path}")
        frame_index += 1
        continue

    orig = image.copy()
    results = model(image)[0]

    # 人物検出
    boxes = []
    for i, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        if class_id == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{i}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Select person (0–9 = select, ESC = skip, ↑ = prev, ↓ = next, q = quit)", image)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        frame_index += 1
        continue
    elif key == ord('q'):
        print("🛑 処理を中断しました")
        break
    elif key == 0:  # ↑
        frame_index = max(0, frame_index - 1)
        continue
    elif key == 1:  # ↓
        frame_index += 1
        continue

    selected = key - ord('0')
    if not (0 <= selected < len(boxes)):
        print("❌ 無効な人物IDでした")
        continue

    print("📌 a: spike, b: block, c: receive, d: serve, e: tosu, f: none")
    class_key = cv2.waitKey(0)
    class_name = CLASS_KEYS.get(class_key)

    if not class_name:
        print("❌ 無効なクラスキーです")
        continue

    # clip範囲を決定
    start = max(0, frame_index - MAX_CLIP // 2)
    end = min(len(frame_paths), start + MAX_CLIP)
    clip_frames = frame_paths[start:end]

    # clip保存ディレクトリ
    clip_dir = os.path.join(OUTPUT_DIR, class_name, f"clip_{clip_counter:04d}")
    os.makedirs(clip_dir, exist_ok=True)

    for i, clip_path in enumerate(clip_frames):
        img = cv2.imread(clip_path)
        if img is None:
            continue

        # 対象人物を検出（再推論）
        res = model(img)[0]
        if selected < len(res.boxes):
            x1, y1, x2, y2 = map(int, res.boxes[selected].xyxy[0])
            crop = img[y1:y2, x1:x2]
            fname = f"frame_{i:04d}.jpg"
            cv2.imwrite(os.path.join(clip_dir, fname), crop)

    print(f"✅ Clip saved to: {clip_dir}")
    clip_counter += 1
    frame_index += 1

cv2.destroyAllWindows()