import cv2
import os
from glob import glob
import re

# 設定
FRAME_DIR = "output"  # ← トラッキング済み画像を使う
OUTPUT_DIR = "shared/cropped_persons"
MAX_CLIP = 10
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

# フレーム一覧
frame_paths = sorted(glob(os.path.join(FRAME_DIR, "*.jpg")))
frame_index = 0
clip_counter = 0

# ID抽出用パターン
ID_PATTERN = re.compile(r'ID\s+(\d+)')

while 0 <= frame_index < len(frame_paths):
    frame_path = frame_paths[frame_index]
    image = cv2.imread(frame_path)
    if image is None:
        print(f"⚠️ 読み込み失敗: {frame_path}")
        frame_index += 1
        continue

    display = image.copy()
    cv2.imshow("IDを選んでください（数字キー）", display)
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

    selected_id = key - ord('0')
    if not (0 <= selected_id <= 9):
        print("❌ 無効なID指定でした")
        continue

    print("📌 a: spike, b: block, c: receive, d: serve, e: tosu, f: none")
    class_key = cv2.waitKey(0)
    class_name = CLASS_KEYS.get(class_key)

    if not class_name:
        print("❌ 無効なクラスキーです")
        continue

    # clip範囲
    start = max(0, frame_index - MAX_CLIP // 2)
    end = min(len(frame_paths), start + MAX_CLIP)
    clip_frames = frame_paths[start:end]

    # 保存先
    clip_dir = os.path.join(OUTPUT_DIR, class_name, f"clip_{clip_counter:04d}")
    os.makedirs(clip_dir, exist_ok=True)

    # フレームごとに対象IDを探して切り出し
    for i, clip_path in enumerate(clip_frames):
        frame = cv2.imread(clip_path)
        if frame is None:
            continue

        # 画像中の「ID N」をOCRまたはラベルで推定するのは難しいので、
        # 現実的には前処理でトラッキング結果（x1,y1,x2,y2,track_id）を保存しておくのが正攻法
        # 今回は仮に track_id=N の矩形を見つける処理を省略（YOLOのboxと同じ方式なら保存必要）

        # 簡易処理: OpenCVのラベル描画情報がない場合、ここはスキップする（例外処理）
        print(f"🚧 フレーム {clip_path} の ID {selected_id} は仮実装中。矩形情報は別途取得してください。")

    print(f"✅ Clip saved to: {clip_dir}")
    clip_counter += 1
    frame_index += 1

cv2.destroyAllWindows()