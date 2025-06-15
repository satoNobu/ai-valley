import cv2
import os
from glob import glob
from ultralytics import YOLO

# 設定
FRAME_DIR = "shared/frames"
OUTPUT_DIR = "shared/cropped_persons"
MODEL_PATH = "shared/yolov8n.pt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# モデル読み込み
model = YOLO(MODEL_PATH)

# 対象画像ファイル一覧
frame_paths = sorted(glob(os.path.join(FRAME_DIR, "*.jpg")))
frame_index = 0

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
        if class_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 緑枠
            cv2.putText(image, f"{i}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # 赤数字

    cv2.imshow("Select person (0–9 = select, ESC = skip, ↑ = prev, ↓ = next, q = quit)", image)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        frame_index += 1
        continue
    elif key == ord('q'):
        print("🛑 処理を中断しました")
        break
    elif key == 0:  # ↑ arrow (KEY_LEFT on macOS/Linux)
        frame_index = max(0, frame_index - 1)
        continue
    elif key == 1:  # ↓ arrow (KEY_DOWN on macOS/Linux)
        frame_index += 1
        continue

    selected = key - ord('0')
    if 0 <= selected < len(boxes):
        x1, y1, x2, y2 = boxes[selected]
        cropped = orig[y1:y2, x1:x2]
        base_name = os.path.basename(frame_path)
        save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(base_name)[0]}_p{selected}.jpg")
        cv2.imwrite(save_path, cropped)
        print(f"✅ Saved: {save_path}")
        frame_index += 1
    else:
        print("❌ 無効なキー入力でした")
        # 再入力を促すため frame_index はそのまま

cv2.destroyAllWindows()