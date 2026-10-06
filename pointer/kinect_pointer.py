"""
kinect_pointer.py
学習済みモデル(pointer_model.joblib)を使って、Kinect v2 + MediaPipe の関節座標から
仮想ポインタの位置を推定し、プロジェクターの Dot Grid 画面に表示する。

測定用スクリプト(kinect.py)と同条件にするため、以下は測定時と完全に同じにしている:
  - Kinect の取得方法(pylibfreenect2 / OpenCLPacketPipeline / registration)
  - MediaPipe の推論対象(registered 512x424)とモデル(pose_landmarker_lite / hand_landmarker)
  - 使う7関節と並び順(JOINT_MAPPING)
  - 特徴量の作り方(write_csv と同じ: x,y は registered_color.shape 基準の int ピクセル、
    z は get_depth_at の最小値パッチ → m 換算して小数2桁に丸め)
  - Dot Grid の画面サイズ(800x600)・グリッド点・プロジェクターへのフルスクリーン表示

操作:  q = 終了 / g = グリッド点の表示切替
"""
import os
import re
import csv
import gc
import time
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

# このマシンのGPU(NVIDIA Quadro K600, Nouveauドライバ)でMesa RusticlのOpenCLを有効にする。
# pylibfreenect2のOpenCLPacketPipeline初期化より前に設定する必要がある。
os.environ.setdefault("RUSTICL_ENABLE", "nouveau")

import cv2
import numpy as np
import joblib
import pylibfreenect2 as fn2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kinect_pointer")

# ============================================================
# ここから下の設定値は測定用 kinect.py と同じ(変更しないこと)
# ============================================================

# --- Kinect v2 解像度 ---
KINECT_COLOR_W, KINECT_COLOR_H = 1920, 1080
KINECT_DEPTH_W, KINECT_DEPTH_H = 512, 424

# Kinect v2 深度有効範囲 [mm]
DEPTH_MIN = 500
DEPTH_MAX = 4500

# --- ドットグリッド設定 ---
GRID_SIZE = 6              # 6x6グリッド
GRID_W, GRID_H = 800, 600  # 黒背景の幅・高さ
DOT_RADIUS = 10            # 点の半径
STEP_X, STEP_Y = 140, 105  # 点と点の間隔(横・縦)
MARGIN_X = (GRID_W - STEP_X * (GRID_SIZE - 1)) // 2
MARGIN_Y = (GRID_H - STEP_Y * (GRID_SIZE - 1)) // 2

GRID_POINTS = [
    (MARGIN_X + col * STEP_X, MARGIN_Y + row * STEP_Y)
    for row in range(GRID_SIZE) for col in range(GRID_SIZE)
]

DELAY_MS = 1       # waitKeyの待機時間(ms)

# Hand Landmarker を Nフレームに1回だけ実行する設定(測定時と同じ1=毎フレーム)
HAND_DETECT_EVERY_N_FRAMES = 1

MAX_CONSECUTIVE_FRAME_FAILURES = 30   # これを超えたら安全停止する

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_SEARCH_DIRS = [SCRIPT_DIR, Path("/home/hayato/mp")]


def find_model_path(filename):
    for d in MODEL_SEARCH_DIRS:
        candidate = d / filename
        if candidate.is_file():
            return candidate
    return MODEL_SEARCH_DIRS[0] / filename  # 見つからなければエラー表示用にデフォルトを返す


POSE_MODEL_PATH = find_model_path("pose_landmarker_lite.task")
HAND_MODEL_PATH = find_model_path("hand_landmarker.task")

PROJECTOR_PORT = "VGA-1-1"
MAIN_DISPLAY_PORT = "DVI-I-1"


def get_monitor_geometry(port_name):
    """xrandr --query の出力からポート名を指定してモニタの位置とサイズ(x, y, w, h)を取得する。"""
    try:
        result = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, check=True, timeout=5
        )
    except Exception as e:
        logger.warning(f"xrandrの実行に失敗しました: {e}")
        return None

    for line in result.stdout.splitlines():
        if line.startswith(port_name + " ") and " connected " in line:
            m = re.search(r"(\d+)x(\d+)\+(\d+)\+(\d+)", line)
            if m:
                w, h, x, y = map(int, m.groups())
                return x, y, w, h
    return None


_projector_geom = get_monitor_geometry(PROJECTOR_PORT)
if _projector_geom:
    SUB_DISPLAY_X, SUB_DISPLAY_Y, _proj_w, _proj_h = _projector_geom
else:
    logger.warning(f"プロジェクター({PROJECTOR_PORT})が検出できなかったため、手動座標を使用します。")
    SUB_DISPLAY_X, SUB_DISPLAY_Y = 1000, 0

_main_geom = get_monitor_geometry(MAIN_DISPLAY_PORT)
if _main_geom:
    MAIN_DISPLAY_X, MAIN_DISPLAY_Y, _, _ = _main_geom
else:
    MAIN_DISPLAY_X, MAIN_DISPLAY_Y = 0, 0

DOT_WINDOW_NAME = "Dot Grid"
CAM_WINDOW_NAME = "Kinect v2 Hand Detection "

# CSVヘッダーで使う部位名。列の並び順もこの順序になる(=特徴量の並び順)
JOINT_MAPPING = {
    "head":        ("pose", 0, "顔"),
    "shoulder":    ("pose", 11, "肩"),
    "elbow":       ("pose", 13, "肘"),
    "wrist":       ("pose", 15, "手首"),
    "hand":        ("hand", 5, "手"),
    "index_pip":   ("hand", 6, "人差し指第二関節"),
    "index_tip":   ("hand", 8, "人差指"),
}
JOINT_NAMES = list(JOINT_MAPPING.keys())

# ============================================================
# 仮想ポインタ用の設定(ここは自由に変えてよい)
# ============================================================
POINTER_MODEL_PATH = find_model_path("pointer_model.joblib")

POINTER_RADIUS = 15               # ポインタ(緑の点)の半径
POINTER_COLOR = (0, 255, 0)       # BGR

# 予測値の平滑化(指数移動平均)。1.0 = 平滑化なし(生の予測値そのまま)。
# 0.3〜0.5 くらいにするとブレが減るが、そのぶん追従が遅れる。
SMOOTHING_ALPHA = 1.0

# 関節が1つでも欠けたフレームは予測しない(学習データでは欠損行が空欄になっていたため)。
# モデルが NaN を扱える(HistGradientBoosting など)場合は True にしてもよい。
ALLOW_MISSING_JOINTS = False

# 予測結果をCSVに残す(後で精度評価に使える)
SAVE_POINTER_LOG = True


# ============================================================
# 測定用 kinect.py と同じ関数
# ============================================================
def verify_model_files():
    missing = []
    for label, path in (("Pose", POSE_MODEL_PATH), ("Hand", HAND_MODEL_PATH),
                        ("Pointer(joblib)", POINTER_MODEL_PATH)):
        if not path.is_file():
            missing.append((label, path))

    if missing:
        logger.error("必要なモデルファイルが見つかりません:")
        for label, path in missing:
            logger.error(f"  - {label} モデル: {path}")
        logger.error("スクリプトと同じフォルダにモデルファイルを置いてください。")
        return False
    return True


def valid_depth_mask(depth_arr):
    """有効な深度値(有限かつ DEPTH_MIN〜DEPTH_MAX 範囲内)を示すブールマスクを返す。"""
    return np.isfinite(depth_arr) & (depth_arr > DEPTH_MIN) & (depth_arr <= DEPTH_MAX)


def get_depth_at(depth_arr, nx, ny, patch=5):
    """正規化座標(0〜1)に対応する深度値[mm]を返す(パッチ内の最小値)。測定時と同じ。"""
    depth_h, depth_w = depth_arr.shape[:2]

    dx = int(np.clip(nx * depth_w, 0, depth_w - 1))
    dy = int(np.clip(ny * depth_h, 0, depth_h - 1))

    h = patch // 2
    x0, x1 = max(0, dx - h), min(depth_w, dx + h + 1)
    y0, y1 = max(0, dy - h), min(depth_h, dy + h + 1)

    region = depth_arr[y0:y1, x0:x1]
    valid = region[valid_depth_mask(region)]
    return float(np.min(valid)) if len(valid) > 0 else float("nan")


def draw_point_at(img, cx, cy, dist_mm):
    cv2.circle(img, (cx, cy), 20, (0, 0, 0), -1)
    cv2.circle(img, (cx, cy), 12, (0, 255, 0), -1)

    text = f"{dist_mm / 1000.0:.2f}m" if not np.isnan(dist_mm) else "--"
    ty = cy - 16
    cv2.putText(img, text, (cx + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(img, text, (cx + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)


def draw_point(img, nx, ny, depth_aligned):
    h, w = img.shape[:2]
    cx = int(nx * w)
    cy = int(ny * h)
    dist_mm = get_depth_at(depth_aligned, nx, ny)
    draw_point_at(img, cx, cy, dist_mm)


def registered_norm_to_color_px(nx, ny, color_depth_map, color_w, color_h):
    dx = int(np.clip(nx * KINECT_DEPTH_W, 0, KINECT_DEPTH_W - 1))
    dy = int(np.clip(ny * KINECT_DEPTH_H, 0, KINECT_DEPTH_H - 1))
    idx = int(color_depth_map[dy * KINECT_DEPTH_W + dx])
    if idx < 0 or idx >= color_w * color_h:
        return None
    return idx % color_w, idx // color_w


def draw_landmarks(img, pose_result, hand_result, depth_aligned):
    if pose_result and pose_result.pose_landmarks:
        pl = pose_result.pose_landmarks[0]
        for source, idx, _ in JOINT_MAPPING.values():
            if source == "pose" and idx < len(pl):
                draw_point(img, pl[idx].x, pl[idx].y, depth_aligned)

    if hand_result and hand_result.hand_landmarks:
        hl = hand_result.hand_landmarks[0]
        for source, idx, _ in JOINT_MAPPING.values():
            if source == "hand" and idx < len(hl):
                draw_point(img, hl[idx].x, hl[idx].y, depth_aligned)


def draw_landmarks_on_color(img, pose_result, hand_result, depth_aligned, color_depth_map):
    h, w = img.shape[:2]

    def emit(lm):
        px = registered_norm_to_color_px(lm.x, lm.y, color_depth_map, w, h)
        if px is None:
            return
        dist_mm = get_depth_at(depth_aligned, lm.x, lm.y)
        draw_point_at(img, px[0], px[1], dist_mm)

    if pose_result and pose_result.pose_landmarks:
        pl = pose_result.pose_landmarks[0]
        for source, idx, _ in JOINT_MAPPING.values():
            if source == "pose" and idx < len(pl):
                emit(pl[idx])

    if hand_result and hand_result.hand_landmarks:
        hl = hand_result.hand_landmarks[0]
        for source, idx, _ in JOINT_MAPPING.values():
            if source == "hand" and idx < len(hl):
                emit(hl[idx])


def grab_kinect_frame(listener, registration, undistorted, registered, color_depth_map):
    frames = None
    try:
        frames = listener.waitForNewFrame()
        color_frame_raw = frames[fn2.FrameType.Color]
        depth_frame_raw = frames[fn2.FrameType.Depth]
        registration.apply(color_frame_raw, depth_frame_raw, undistorted, registered,
                           color_depth_map=color_depth_map)

        native_color = color_frame_raw.asarray(np.uint8)[:, :, :3].copy()
        registered_color = registered.asarray(np.uint8)[:, :, :3].copy()
        undistorted_depth = undistorted.asarray(np.float32).copy()

        return native_color, registered_color, undistorted_depth, color_depth_map.copy()

    except Exception as e:
        logger.warning(f"フレーム取得に失敗しました(スキップして継続): {e}")
        return None

    finally:
        if frames is not None:
            listener.release(frames)


class FrameGrabber:
    """Kinectのフレーム取得を専用スレッドで回し、常に最新フレームだけを渡す(測定時と同じ)。"""

    def __init__(self, listener, registration, undistorted, registered):
        self._listener = listener
        self._registration = registration
        self._undistorted = undistorted
        self._registered = registered
        self._color_depth_map = np.full(KINECT_DEPTH_W * KINECT_DEPTH_H, -1, dtype=np.int32)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest = None
        self._frame_id = 0
        self._consecutive_failures = 0
        self._stop_event = threading.Event()
        self._failed_permanently = False

        self._thread = threading.Thread(target=self._run, name="KinectGrabber", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop_event.is_set():
            grabbed = grab_kinect_frame(
                self._listener, self._registration,
                self._undistorted, self._registered,
                self._color_depth_map,
            )
            with self._cond:
                if grabbed is None:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_CONSECUTIVE_FRAME_FAILURES:
                        self._failed_permanently = True
                        self._cond.notify_all()
                        break
                    continue

                self._consecutive_failures = 0
                self._latest = grabbed
                self._frame_id += 1
                self._cond.notify_all()

    def get_latest(self, last_seen_id, timeout=0.5):
        with self._cond:
            got_new = self._cond.wait_for(
                lambda: self._frame_id != last_seen_id or self._failed_permanently,
                timeout=timeout,
            )
            if self._failed_permanently:
                return None, None, None, None, self._frame_id, True
            if not got_new:
                return None, None, None, None, last_seen_id, False
            native_color, registered_color, undistorted_depth, color_depth_map = self._latest
            return (native_color, registered_color, undistorted_depth, color_depth_map,
                    self._frame_id, False)


def make_depth_display(depth_aligned):
    mask = valid_depth_mask(depth_aligned)
    depth_norm = np.zeros(depth_aligned.shape, dtype=np.uint8)
    depth_norm[mask] = ((DEPTH_MAX - depth_aligned[mask]) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)
    return cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)


# ============================================================
# 仮想ポインタ用に追加した部分
# ============================================================
def feature_column_names():
    """測定CSVの列名(joint_head_x, joint_head_y, joint_head_z, ...)をその順で返す。"""
    cols = []
    for name in JOINT_NAMES:
        cols += [f"joint_{name}_x", f"joint_{name}_y", f"joint_{name}_z"]
    return cols


def extract_features(pose_result, hand_result, depth_aligned, img_shape):
    """
    測定用 write_csv と全く同じ計算で特徴量を作り、{列名: 値} の辞書で返す。
      - x, y: int(lm.x * w), int(lm.y * h)   (img_shape = registered_color.shape = 424x512)
      - z   : get_depth_at の値[mm] → m に換算し小数2桁に丸め(CSVの "%.2f" と同じ)
    取れなかった値は NaN。
    """
    h, w = img_shape[:2]
    feats = {c: float("nan") for c in feature_column_names()}

    def put(name, lm):
        dist_mm = get_depth_at(depth_aligned, lm.x, lm.y)
        feats[f"joint_{name}_x"] = float(int(lm.x * w))
        feats[f"joint_{name}_y"] = float(int(lm.y * h))
        feats[f"joint_{name}_z"] = (round(dist_mm / 1000.0, 2)
                                    if not np.isnan(dist_mm) else float("nan"))

    if pose_result and pose_result.pose_landmarks:
        pl = pose_result.pose_landmarks[0]
        for name, (source, idx, _) in JOINT_MAPPING.items():
            if source == "pose" and idx < len(pl):
                put(name, pl[idx])

    if hand_result and hand_result.hand_landmarks:
        hl = hand_result.hand_landmarks[0]
        for name, (source, idx, _) in JOINT_MAPPING.items():
            if source == "hand" and idx < len(hl):
                put(name, hl[idx])

    return feats


class PointerModel:
    """
    pointer_model.joblib を読み込み、特徴量の辞書から (x, y) を予測する。
    - モデル単体 / Pipeline / {"model": ..., "scaler": ...} 形式の辞書 のどれでも動くようにしている。
    - 学習時に DataFrame で学習していれば(feature_names_in_ がある)、その列名・列順に合わせて渡す。
      無ければ測定CSVの列順(head→shoulder→elbow→wrist→hand→index_pip→index_tip, 各x,y,z)で渡す。
    """

    def __init__(self, path):
        obj = joblib.load(path)
        self.scaler = None
        self.feature_names = None

        if isinstance(obj, dict):
            logger.info(f"joblib は辞書形式でした。キー: {list(obj.keys())}")
            self.model = next((obj[k] for k in ("model", "reg", "estimator", "regressor", "clf")
                               if k in obj), None)
            if self.model is None:
                self.model = next(v for v in obj.values() if hasattr(v, "predict"))
            self.scaler = next((obj[k] for k in ("scaler", "x_scaler", "sc") if k in obj), None)
            for k in ("feature_names", "features", "feature_cols", "columns", "x_cols"):
                if k in obj:
                    self.feature_names = list(obj[k])
                    break
        else:
            self.model = obj

        if self.feature_names is None:
            src = self.scaler if (self.scaler is not None and hasattr(self.scaler, "feature_names_in_")) \
                else self.model
            if hasattr(src, "feature_names_in_"):
                self.feature_names = list(src.feature_names_in_)

        self.use_dataframe = self.feature_names is not None
        if self.feature_names is None:
            self.feature_names = feature_column_names()

        n_in = getattr(self.model, "n_features_in_", None)
        if n_in is not None and self.scaler is None and n_in != len(self.feature_names):
            raise ValueError(
                f"モデルの入力次元({n_in})と特徴量の数({len(self.feature_names)})が一致しません。"
                "学習時に使った列を確認してください。"
            )

        missing = [c for c in self.feature_names if c not in feature_column_names()]
        if missing:
            raise ValueError(f"モデルが要求する列をリアルタイムで作れません: {missing}")

        logger.info(f"ポインタモデル: {type(self.model).__name__}"
                    + (f" + scaler({type(self.scaler).__name__})" if self.scaler is not None else ""))
        logger.info(f"入力特徴量({len(self.feature_names)}): {self.feature_names}")

        if self.use_dataframe:
            import pandas as pd
            self._pd = pd

    def predict(self, feats):
        row = [feats[c] for c in self.feature_names]
        if self.use_dataframe:
            X = self._pd.DataFrame([row], columns=self.feature_names)
        else:
            X = np.array([row], dtype=np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        y = np.asarray(self.model.predict(X)).reshape(-1)
        return float(y[0]), float(y[1])


def draw_pointer_screen(pointer_xy, show_grid, status_text=None):
    """Dot Grid と同じ 800x600 の黒画面に、グリッド点(任意)と仮想ポインタを描く。"""
    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    if show_grid:
        for pt in GRID_POINTS:
            cv2.circle(img, pt, DOT_RADIUS, (255, 0, 0), -1)

    if pointer_xy is not None:
        px = int(np.clip(pointer_xy[0], 0, GRID_W - 1))
        py = int(np.clip(pointer_xy[1], 0, GRID_H - 1))
        cv2.circle(img, (px, py), POINTER_RADIUS + 3, (0, 0, 0), -1)
        cv2.circle(img, (px, py), POINTER_RADIUS, POINTER_COLOR, -1, cv2.LINE_AA)

    if status_text:
        cv2.putText(img, status_text, (10, GRID_H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (128, 128, 128), 1, cv2.LINE_AA)
    return img


def main():
    if not verify_model_files():
        return

    pointer_model = PointerModel(POINTER_MODEL_PATH)

    fi = fn2.Freenect2()
    if fi.enumerateDevices() == 0:
        logger.error("Kinect v2 デバイスが検出されませんでした。接続を確認してください。")
        return

    pipeline = fn2.OpenCLPacketPipeline()
    device = fi.openDevice(fi.getDeviceSerialNumber(0), pipeline=pipeline)

    listener = fn2.SyncMultiFrameListener(
        fn2.FrameType.Color | fn2.FrameType.Depth
    )
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.start()

    registration = fn2.Registration(device.getIrCameraParams(), device.getColorCameraParams())

    undistorted = fn2.Frame(KINECT_DEPTH_W, KINECT_DEPTH_H, 4)
    registered = fn2.Frame(KINECT_DEPTH_W, KINECT_DEPTH_H, 4)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    grabber = FrameGrabber(listener, registration, undistorted, registered)
    grabber.start()
    last_frame_id = 0

    # ドットグリッド画面はプロジェクターへ移動してからフルスクリーン化(測定時と同じ順序)
    cv2.namedWindow(DOT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(DOT_WINDOW_NAME, SUB_DISPLAY_X, SUB_DISPLAY_Y)
    cv2.setWindowProperty(DOT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.namedWindow(CAM_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(CAM_WINDOW_NAME, MAIN_DISPLAY_X, MAIN_DISPLAY_Y)

    log_file = None
    log_writer = None
    if SAVE_POINTER_LOG:
        os.makedirs("pointer_logs", exist_ok=True)
        log_name = f"pointer_logs/pointer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_file = open(log_name, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["timestamp_ms"] + feature_column_names()
                            + ["pred_x", "pred_y", "pointer_x", "pointer_y"])
        logger.info(f"予測ログの保存先: {log_name}")

    logger.info("仮想ポインタを開始します。q=終了 / g=グリッド点の表示切替")

    pose_landmarker_lite = None
    hand_landmarker = None
    frame_counter = 0
    last_hand_result = None

    fps_window_start = time.time()
    fps_frame_count = 0
    fps_display = 0.0

    pointer_xy = None        # 平滑化後のポインタ位置(Noneなら非表示)
    show_grid = True

    executor = ThreadPoolExecutor(max_workers=1)

    try:
        pose_landmarker_lite = vision.PoseLandmarker.create_from_options(pose_options)
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        while True:
            (native_color, registered_color, undistorted_depth, color_depth_map,
             last_frame_id, failed) = grabber.get_latest(last_frame_id, timeout=0.5)

            if failed:
                logger.error(f"フレーム取得が{MAX_CONSECUTIVE_FRAME_FAILURES}回連続で失敗したため終了します。")
                break

            if native_color is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            depth_colored = make_depth_display(undistorted_depth)

            # MediaPipe推論は registered(512×424)で行う(測定時と同じ)
            mp_image = Image(image_format=ImageFormat.SRGB,
                             data=cv2.cvtColor(registered_color, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(time.time() * 1000)

            pose_future = executor.submit(
                pose_landmarker_lite.detect_for_video, mp_image, timestamp_ms
            )

            frame_counter += 1
            if frame_counter % HAND_DETECT_EVERY_N_FRAMES == 0 or last_hand_result is None:
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                last_hand_result = hand_result
            else:
                hand_result = last_hand_result

            pose_result = pose_future.result()

            # ---- 特徴量 → 仮想ポインタ ----
            # img_shape は測定時の write_csv と同じく registered_color.shape を渡す
            feats = extract_features(pose_result, hand_result, undistorted_depth,
                                     registered_color.shape)
            has_missing = any(np.isnan(feats[c]) for c in pointer_model.feature_names)

            pred = None
            if not has_missing or ALLOW_MISSING_JOINTS:
                try:
                    pred = pointer_model.predict(feats)
                except Exception as e:
                    logger.warning(f"予測に失敗しました: {e}")

            if pred is None:
                pointer_xy = None
            elif pointer_xy is None or SMOOTHING_ALPHA >= 1.0:
                pointer_xy = pred
            else:
                a = SMOOTHING_ALPHA
                pointer_xy = (a * pred[0] + (1 - a) * pointer_xy[0],
                              a * pred[1] + (1 - a) * pointer_xy[1])

            if log_writer is not None:
                log_writer.writerow(
                    [timestamp_ms] + [feats[c] for c in feature_column_names()]
                    + ([f"{pred[0]:.1f}", f"{pred[1]:.1f}"] if pred else ["", ""])
                    + ([f"{pointer_xy[0]:.1f}", f"{pointer_xy[1]:.1f}"] if pointer_xy else ["", ""])
                )

            # ---- プロジェクター画面(800x600, 測定時と同じサイズ) ----
            status = None if pointer_xy is not None else "tracking lost"
            dot_img = draw_pointer_screen(pointer_xy, show_grid, status)

            # ---- 手元モニタのカメラ画面(測定時と同じ表示) ----
            annotated_frame = native_color.copy()
            draw_landmarks_on_color(annotated_frame, pose_result, hand_result,
                                    undistorted_depth, color_depth_map)

            depth_h_target = annotated_frame.shape[0]
            depth_w_target = int(depth_colored.shape[1] * depth_h_target / depth_colored.shape[0])
            depth_disp = cv2.resize(depth_colored, (depth_w_target, depth_h_target),
                                    interpolation=cv2.INTER_NEAREST)
            draw_landmarks(depth_disp, pose_result, hand_result, undistorted_depth)

            info = (f"pointer=({pointer_xy[0]:.0f}, {pointer_xy[1]:.0f})"
                    if pointer_xy is not None else "pointer=-- (joint missing)")
            info += f"   FPS={fps_display:.1f}"
            cv2.putText(annotated_frame, info, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(annotated_frame, info, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(DOT_WINDOW_NAME, dot_img)
            cv2.imshow(CAM_WINDOW_NAME, np.hstack([annotated_frame, depth_disp]))

            fps_frame_count += 1
            now = time.time()
            if now - fps_window_start >= 1.0:
                fps_display = fps_frame_count / (now - fps_window_start)
                fps_window_start = now
                fps_frame_count = 0

            key = cv2.waitKey(DELAY_MS) & 0xFF
            if key == ord('q'):
                break
            if key == ord('g'):
                show_grid = not show_grid

    finally:
        grabber.stop()
        executor.shutdown(wait=True)
        if pose_landmarker_lite:
            pose_landmarker_lite.close()
        if hand_landmarker:
            hand_landmarker.close()
        if log_file is not None:
            log_file.close()
        device.stop()
        # OpenCLPacketPipeline使用時、listenerはdevice.close()より先に破棄する(測定時と同じ)
        del grabber
        del listener
        gc.collect()
        device.close()
        cv2.destroyAllWindows()
        logger.info("安全に終了しました。")


if __name__ == "__main__":
    main()
