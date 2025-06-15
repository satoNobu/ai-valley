import cv2
import os
import json
from glob import glob
import numpy as np

# 設定
FRAME_DIR = "output"
OUTPUT_DIR = "shared/cropped_persons"
TRACK_JSON = os.path.join(FRAME_DIR, "track_results.json")
MAX_CLIP = 10
MARGIN = 20  # bboxの拡張マージン
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

# トラッキング結果を読み込む
with open(TRACK_JSON, "r") as f:
    raw_results = json.load(f)

track_data_index = {
    frame: {int(obj["id"]): obj["bbox"] for obj in objs}
    for frame, objs in raw_results.items()
}

frame_paths = sorted(glob(os.path.join(FRAME_DIR, "*.jpg")))
frame_index = 0
clip_counter = 0
selected_id = None
mouse_x, mouse_y = -1, -1
total = len(frame_paths)

# マウスクリックでID選択
def click_event(event, x, y, flags, param):
    global mouse_x, mouse_y, selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        fname = os.path.basename(frame_paths[frame_index])
        candidates = track_data_index.get(fname, {})
        for tid, bbox in candidates.items():
            x1, y1, x2, y2 = map(int, bbox)
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = tid
                print(f"[✅] ID {tid} 選択（クリック位置: {x}, {y}）")
                break

cv2.namedWindow("フレーム確認（↑↓で移動、クリック→Enterで分類）")
cv2.setMouseCallback("フレーム確認（↑↓で移動、クリック→Enterで分類）", click_event)

while 0 <= frame_index < total:
    frame_path = frame_paths[frame_index]
    frame_name = os.path.basename(frame_path)
    image = cv2.imread(frame_path)
    if image is None:
        print(f"⚠️ 読み込み失敗: {frame_path}")
        frame_index += 1
        continue

    display = image.copy()
    if selected_id is not None:
        bbox = track_data_index.get(frame_name, {}).get(selected_id)
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"ID {selected_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("フレーム確認（↑↓で移動、クリック→Enterで分類）", display)
    key = cv2.waitKey(0)
    print(f"[DEBUG] key = {key}")

    if key == ord('q'):
        print("🛑 終了します")
        break
    elif key in [2490368, 65362, 0]:  # ↑
        frame_index = max(0, frame_index - 1)
        selected_id = None
        continue
    elif key in [2621440, 65364, 1]:  # ↓
        frame_index = min(total - 1, frame_index + 1)
        selected_id = None
        continue
    elif key == 27:  # ESC
        frame_index += 1
        selected_id = None
        continue
    elif key == 13:  # Enter → 分類入力
        if selected_id is None:
            print("❌ 先にクリックで人物を選択してください")
            continue

        print("📌 a: spike, b: block, c: receive, d: serve, e: tosu, f: none")
        class_key = cv2.waitKey(0)
        class_name = CLASS_KEYS.get(class_key)
        if not class_name:
            print("❌ 無効な分類キーです")
            continue

        clip_dir = os.path.join(OUTPUT_DIR, class_name, f"clip_{clip_counter:04d}")
        os.makedirs(clip_dir, exist_ok=True)

        ## 起点から前10枚の場合
        # start = max(0, frame_index - MAX_CLIP // 2)
        # end = min(total, start + MAX_CLIP)

        ## 起点から後10枚の場合
        start = frame_index
        end = min(total, frame_index + MAX_CLIP)
        clip_paths = frame_paths[start:end]
        last_crop = None

        for i, path in enumerate(clip_paths):
            fname = os.path.basename(path)
            frame = cv2.imread(path)
            bbox = track_data_index.get(fname, {}).get(selected_id)

            if frame is None:
                print(f"⚠️ 読み込み失敗: {fname}")
                continue

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                x1 = np.clip(x1 - MARGIN, 0, w)
                y1 = np.clip(y1 - MARGIN, 0, h)
                x2 = np.clip(x2 + MARGIN, 0, w)
                y2 = np.clip(y2 + MARGIN, 0, h)
                crop = frame[y1:y2, x1:x2]
                last_crop = crop.copy()
            elif last_crop is not None:
                crop = last_crop.copy()
                print(f"🔁 補完: {fname} に ID {selected_id} が見つからず、前フレームから複製")
            else:
                print(f"⚠️ スキップ: ID {selected_id} 見つからず補完できず {fname}")
                continue

            save_path = os.path.join(clip_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(save_path, crop)

        print(f"✅ Clip 保存完了: {clip_dir}")
        clip_counter += 1
        frame_index += 1
        selected_id = None
    else:
        print("🔁 ↑↓キーで移動、クリックでID選択、Enterで分類へ")

cv2.destroyAllWindows()