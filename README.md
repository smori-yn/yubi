# yubi — 指差しによる指示位置推定

Kinect v2 + MediaPipe で取った関節の 3D 座標から、プロジェクター画面上の「指している位置」を推定する。

詳しい説明・経緯・データの置き場所は wiki（`yubisasi`）が正。ここにはコードと導入の最小限だけを置く。

## 構成

| フォルダ | 中身 |
|---|---|
| `record/` | 収録。画面のドットを指でなぞってもらい、関節座標と目標点を CSV に書く |
| `train/` | 学習（Stage 1: MLP で関節座標 → 画面座標） |
| `pointer/` | 学習済みモデルで、指している位置をリアルタイムに表示する |
| `shm/` | Kinect の映像を共有メモリで配る C++ サーバと、それを読む Python（姿勢の 3D 表示） |

## 必要なもの

- Kinect v2 と libfreenect2（pylibfreenect2）
- MediaPipe のモデル `pose_landmarker*.task` / `hand_landmarker.task`（git には入れていない）
- 学習だけなら `pip install -r requirements-train.txt`

## 使いかた（概略）

```bash
python record/kinect_7mark.py            # 収録 → landmarks_YYYYMMDD_HHMMSS.csv
python train/train_mlp.py                # 学習（パスはファイル冒頭の設定で指定）
python pointer/kinect_pointer.py         # 仮想ポインタ（pointer_model.joblib が必要）

g++ -O2 -o shm/kinect_shm_server shm/kinect_shm_server.cpp `pkg-config freenect2 --libs --cflags`
./shm/kinect_shm_server & python shm/kinect_pose_3d.py
```
