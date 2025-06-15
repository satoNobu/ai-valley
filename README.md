# ai-valley
ソフトバレーの動画解析

# ビルド
docker build -t video-frame-yolo .

## 配置
video_frame_extractor/
├── Dockerfile
├── requirements.txt
├── extract_frames.py
├── output 　　　　　　　　　　 ←人物IDを割り振り
├　├──　frame_0276.jpg
├　├──　track_results.json   ←人物IDのjson
└── shared/
    ├── input_video.mp4      ← 動画をここに置く
    └── frames/              ← 抽出画像がここに出力される
 dataset/　　　　　　　　　　　　← 学習データ
 ├── spike/
 │   ├── 1/（← ファイル名自由でOK）
 │   │   ├── frame_0190.jpg
 │   │   ├── frame_0191.jpg
 │   └── 2/
 ├── receive/
 └── block/
└── none/
    ├── idle/          # プレイ中に静止している（立ち止まり）
    ├── walking/       # 歩いている、移動中の選手
    ├── pre-move/      # スパイク・レシーブに入る前の準備動作（曖昧ゾーン）
    ├── background/    # 誰も映っていない、または明らかにプレイ中ではない
    └── misc/          # その他、分類不能なnone系動作

## 実行
### 1. 動画の検出
input_video.mp4という動画でshared配下に分割したframeを配置
以下のコマンドを実行

docker run --rm -v $(pwd)/shared:/shared video-frame-yolo

### 2. ID付きで画像を出力
python3 run_deep_sort.py

### 3. 学習様用データの作成（手動）
python3 person_selector.py

　L 画像を選択（上下で移動）
　L Enter をタップ
　L 分類分け
　L qで終了

### 2. スパイクなどの学習
docker run --rm \
  -v $(pwd)/shared:/shared \
  -v $(pwd)/dataset:/app/dataset \
  video-frame-yolo \
  python confusion_matrix.py

### 3. 自動で抽出
docker run --rm \
  -v $(pwd)/shared:/shared \
  video-frame-yolo \
  python predict_frames.py

### 3. 自動で抽出（学習用）
docker run --rm \
  -v $(pwd)/shared:/shared \
  video-frame-yolo \
  python predict_frames_slim.py