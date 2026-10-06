import os
import csv
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pylibfreenect2 as fn2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

try:
    from screeninfo import get_monitors
    _monitors = get_monitors()
except Exception:
    _monitors = []

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kinect_tracking")



# --- Kinect v2 解像度 ---
KINECT_COLOR_W, KINECT_COLOR_H = 1920, 1080
KINECT_DEPTH_W, KINECT_DEPTH_H = 512, 424
# libfreenect2の仕様上、bigdepthは高さ1082pxで返るが末尾2行は無効領域のため使わない
KINECT_BIGDEPTH_H = 1082

# カラー(1920x1080)をDepth(512x424)のアスペクト比にクロップするための定数
TARGET_W = int(KINECT_COLOR_H * KINECT_DEPTH_W / KINECT_DEPTH_H)   # 1304px
CROP_X = (KINECT_COLOR_W - TARGET_W) // 2                          # 308px(左右均等にカット)

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

COUNTDOWN_SEC = 5     # 開始前のカウントダウン秒数
DELAY_MS = 1       # waitKeyの待機時間(ms)

# Hand Landmarker Nフレームに1回だけ実行する。
HAND_DETECT_EVERY_N_FRAMES = 1

MAX_CONSECUTIVE_FRAME_FAILURES = 30   #これを超えたら安全停止する

# 環境変数で上書き可能、既定はスクリプトと同じ場所を探す
SCRIPT_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = Path(os.environ.get("POSE_MODEL_PATH", SCRIPT_DIR / "pose_landmarker.task"))
HAND_MODEL_PATH = Path(os.environ.get("HAND_MODEL_PATH", SCRIPT_DIR / "hand_landmarker.task"))

if len(_monitors) >= 2:
    SUB_DISPLAY_X = _monitors[1].x
    SUB_DISPLAY_Y = _monitors[1].y
else:
    # screeninfoが使えない/サブディスプレイが検出できない場合は手動
    SUB_DISPLAY_X = 1000
    SUB_DISPLAY_Y = 0

DOT_WINDOW_NAME = "Dot Grid"
CAM_WINDOW_NAME = "Kinect v2 Hand Detection "

# CSVヘッダーで使う部位名。列の並び順もこの順序になる
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


def verify_model_files():
    missing = []
    for label, path in (("Pose", POSE_MODEL_PATH), ("Hand", HAND_MODEL_PATH)):
        if not path.is_file():
            missing.append((label, path))

    if missing:
        logger.error("必要なモデルファイルが見つかりません:")
        for label, path in missing:
            logger.error(f"  - {label} Landmarker モデル: {path}")
        logger.error(
            "スクリプトと同じフォルダにモデルファイルを置くか、"
            "環境変数 POSE_MODEL_PATH / HAND_MODEL_PATH でパスを指定してください。"
        )
        return False
    return True



def valid_depth_mask(depth_arr):
    """有効な深度値(有限かつ DEPTH_MIN〜DEPTH_MAX 範囲内)を示すブールマスクを返す。"""
    return np.isfinite(depth_arr) & (depth_arr > DEPTH_MIN) & (depth_arr <= DEPTH_MAX)


def crop_to_depth_ratio(img):
    """
    画像をDepth(512x424)のアスペクト比にクロップして512x424に。
    uint8・float32どちらにも対応。
    """
    h, w = img.shape[:2]
    x0 = int(CROP_X * w / KINECT_COLOR_W)
    x1 = w - x0
    cropped = img[:, x0:x1]
    return cv2.resize(cropped, (KINECT_DEPTH_W, KINECT_DEPTH_H), interpolation=cv2.INTER_NEAREST)


def spiral_order(n):
    order = []
    top, bottom, left, right = 0, n - 1, 0, n - 1
    while top <= bottom and left <= right:
        for row in range(top, bottom + 1):
            order.append(row * n + left)
        left += 1
        for col in range(left, right + 1):
            order.append(bottom * n + col)
        bottom -= 1
        for row in range(bottom, top - 1, -1):
            order.append(row * n + right)
        right -= 1
        for col in range(right, left - 1, -1):
            order.append(top * n + col)
        top += 1
    return order


def get_depth_at(depth_arr, nx, ny, patch=5):
    """
    入力された深度マップの実際のサイズから、正規化座標(0〜1)に対応する深度値[mm]を返す。
    patch サイズの近傍ピクセルの中央値を使う。
    nx, ny は depth_arr と同じ座標系(=同じアスペクト比)の正規化座標である前提。
    """
    depth_h, depth_w = depth_arr.shape[:2]

    dx = int(np.clip(nx * depth_w, 0, depth_w - 1))
    dy = int(np.clip(ny * depth_h, 0, depth_h - 1))

    h = patch // 2
    x0, x1 = max(0, dx - h), min(depth_w, dx + h + 1)
    y0, y1 = max(0, dy - h), min(depth_h, dy + h + 1)

    region = depth_arr[y0:y1, x0:x1]
    valid = region[valid_depth_mask(region)]
    return float(np.median(valid)) if len(valid) > 0 else float("nan")


# 描画関連
def draw_point(img, nx, ny, label, depth_aligned=None, show_dist=False, show_label=True):
    """関節点・ラベル・距離を描画する。label は改善前は未使用だったが、ここで実際に描画する。"""
    h, w = img.shape[:2]
    cx = int(nx * w)
    cy = int(ny * h)
    cv2.circle(img, (cx, cy), 7, (0, 0, 0), -1)
    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

    lines = []
    if show_label and label:
        lines.append(label)

    if show_dist and depth_aligned is not None:
        dist_mm = get_depth_at(depth_aligned, nx, ny)
        if not np.isnan(dist_mm):
            lines.append(f"{dist_mm / 1000.0:.2f}m")
        else:
            lines.append("--")

    for i, text in enumerate(lines):
        ty = cy - 10 - i * 18
        cv2.putText(img, text, (cx + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (cx + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def draw_landmarks(img, pose_result, hand_result, depth_aligned):
    """Pose・Hand のランドマークを描画する(7点で深度表示)。"""
    if pose_result and pose_result.pose_landmarks:
        pl = pose_result.pose_landmarks[0]
        for name, (source, idx, label) in JOINT_MAPPING.items():
            if source == "pose" and idx < len(pl):
                draw_point(img, pl[idx].x, pl[idx].y, label, depth_aligned=depth_aligned, show_dist=True, show_label=False)

    if hand_result and hand_result.hand_landmarks:
        hl = hand_result.hand_landmarks[0]
        for name, (source, idx, label) in JOINT_MAPPING.items():
            if source == "hand" and idx < len(hl):
                draw_point(img, hl[idx].x, hl[idx].y, label, depth_aligned=depth_aligned, show_dist=True, show_label=False)


def draw_dot_grid(grid_points, red_pt, radius, countdown=None):
    """
    指先追従用画像
    黒背景に青い点と移動する点を描画した画像を返す。
    grid_points / radius を引数として明示的に受け取る。
    countdown に整数(1〜COUNTDOWN_SEC)を渡すとカウントダウン数字を中央に重ねて表示する。
    """
    dot_img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    for pt in grid_points:
        cv2.circle(dot_img, pt, radius, (255, 0, 0), -1)
    cv2.circle(dot_img, red_pt, radius, (0, 255, 255), -1)

    if countdown is not None:
        text = str(countdown)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 6.0
        thickness = 12
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = (GRID_W - tw) // 2
        ty = (GRID_H + th) // 2
        cv2.putText(dot_img, text, (tx + 4, ty + 4), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
        cv2.putText(dot_img, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return dot_img

def update_spiral(spiral_idx, last_move_time, spiral, grid_points):
    """スパイラルの赤い点座標を返し、必要なら次の点に進める。"""
    elapsed = time.time() - last_move_time
    if spiral_idx < len(spiral) - 1:
        t = min(elapsed, 1.0)
        pt_a = grid_points[spiral[spiral_idx]]
        pt_b = grid_points[spiral[spiral_idx + 1]]
        red_pt = (
            int(pt_a[0] * (1.0 - t) + pt_b[0] * t),
            int(pt_a[1] * (1.0 - t) + pt_b[1] * t),
        )
        if elapsed >= 1.0:
            spiral_idx += 1
            last_move_time = time.time()
    else:
        red_pt = grid_points[spiral[spiral_idx]]
    return red_pt, spiral_idx, last_move_time


# CSV関連
def build_csv_header():
    header = ["timestamp_ms", "spiral_idx"]
    for name in JOINT_NAMES:
        header += [f"joint_{name}_x", f"joint_{name}_y", f"joint_{name}_z"]
    header += ["target_x", "target_y"]
    return header


def write_csv(writer, timestamp_ms, spiral_idx, red_pt, pose_result, hand_result, depth_aligned, img_shape):
    """
    CSV書き込み
    joint_*_x/y は img_shape(=カラー推論画像 1304x1080)基準のピクセル座標
    joint_*_z は depth_aligned(512x424)から取得したmm→m変換済みの深度
    """
    h, w = img_shape[:2]
    values = {name: ("", "", "") for name in JOINT_NAMES}

    def landmark_xyz(lm):
        dist_mm = get_depth_at(depth_aligned, lm.x, lm.y)
        z_str = f"{dist_mm / 1000.0:.2f}" if not np.isnan(dist_mm) else ""
        return (int(lm.x * w), int(lm.y * h), z_str)

    if pose_result and pose_result.pose_landmarks:
        pl = pose_result.pose_landmarks[0]
        for name, (source, idx, _) in JOINT_MAPPING.items():
            if source == "pose" and idx < len(pl):
                values[name] = landmark_xyz(pl[idx])

    if hand_result and hand_result.hand_landmarks:
        hl = hand_result.hand_landmarks[0]
        for name, (source, idx, _) in JOINT_MAPPING.items():
            if source == "hand" and idx < len(hl):
                values[name] = landmark_xyz(hl[idx])

    row = [timestamp_ms, spiral_idx]
    for name in JOINT_NAMES:
        row += list(values[name])
    row += [red_pt[0], red_pt[1]]

    writer.writerow(row)


def grab_kinect_frame(listener, registration, bigdepth, undistorted, registered):
    """
    Kinectから1フレーム分のカラー・深度データを取得する。
    失敗した場合は None を返す(呼び出し側で連続失敗をカウントする)。
    """
    frames = None
    try:
        frames = listener.waitForNewFrame()
        color_frame_raw = frames[fn2.FrameType.Color]
        depth_frame_raw = frames[fn2.FrameType.Depth]
        #ここで画角合わせしている
        registration.apply(color_frame_raw, depth_frame_raw, undistorted, registered, bigdepth=bigdepth)

        frame_raw = color_frame_raw.asarray(np.uint8)[:, :, :3]
        frame_full = frame_raw[:, CROP_X:KINECT_COLOR_W - CROP_X]

        # bigdepthは高さ1082pxで返るが有効なのは先頭1080行のみ(KINECT_BIGDEPTH_Hの注記参照)
        depth_reg_raw = bigdepth.asarray(np.float32)[0:KINECT_COLOR_H, :]
        depth_aligned = crop_to_depth_ratio(depth_reg_raw)

        return frame_full, depth_aligned

    except Exception as e:
        logger.warning(f"フレーム取得に失敗しました(スキップして継続): {e}")
        return None

    finally:
        if frames is not None:
            listener.release(frames)


class FrameGrabber:
    """
    Kinectからのフレーム取得(waitForNewFrame〜release)を専用スレッドで回し続け、
    メインスレッドは常に「その時点で一番新しいフレーム」だけを受け取れるようにする。

    こうすることで、Pose/Hand推論や描画が重くてメインループが遅くても、
    Kinect側の取得(USB受信)自体は滞りなく回り続け、
    表示される映像が処理待ちで古いまま溜まっていく(遅延が蓄積する)ことを防ぐ。
    """

    def __init__(self, listener, registration, bigdepth, undistorted, registered):
        self._listener = listener
        self._registration = registration
        self._bigdepth = bigdepth
        self._undistorted = undistorted
        self._registered = registered

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest = None       # (frame_full, depth_aligned)
        self._frame_id = 0        # 新しいフレームが来るたびに増える
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
                self._listener, self._registration, self._bigdepth,
                self._undistorted, self._registered,
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
        """
        last_seen_id より新しいフレームが来るまで(timeout秒を上限に)待って返す。
        戻り値: (frame_full, depth_aligned, frame_id, failed_permanently)
        タイムアウトした場合は frame_full=None, depth_aligned=None のまま last_seen_id を返す。
        """
        with self._cond:
            got_new = self._cond.wait_for(
                lambda: self._frame_id != last_seen_id or self._failed_permanently,
                timeout=timeout,
            )
            if self._failed_permanently:
                return None, None, self._frame_id, True
            if not got_new:
                return None, None, last_seen_id, False
            frame_full, depth_aligned = self._latest
            return frame_full, depth_aligned, self._frame_id, False


def make_depth_display(depth_aligned):
    #表示用深度グレースケール画像 範囲外=0(真っ黒)。
    mask = valid_depth_mask(depth_aligned)
    depth_norm = np.zeros(depth_aligned.shape, dtype=np.uint8)
    depth_norm[mask] = ((DEPTH_MAX - depth_aligned[mask]) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)
    return cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)


# メイン処理
def main():
    if not verify_model_files():
        return

    fi = fn2.Freenect2()
    if fi.enumerateDevices() == 0:
        logger.error("Kinect v2 デバイスが検出されませんでした。接続を確認してください。")
        return

    pipeline = fn2.CpuPacketPipeline()
    device = fi.openDevice(fi.getDeviceSerialNumber(0), pipeline=pipeline)

    listener = fn2.SyncMultiFrameListener(
        fn2.FrameType.Color | fn2.FrameType.Depth | fn2.FrameType.Ir
    )
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.start()

    registration = fn2.Registration(device.getIrCameraParams(), device.getColorCameraParams())

    bigdepth = fn2.Frame(KINECT_COLOR_W, KINECT_BIGDEPTH_H, 4)
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

    # Kinectのフレーム取得を専用スレッドで回し、メインループの推論負荷が取得タイミング(USB受信)をブロックしないようにする
    grabber = FrameGrabber(listener, registration, bigdepth, undistorted, registered)
    grabber.start()
    last_frame_id = 0

    spiral = spiral_order(GRID_SIZE)
    spiral_idx = 0

    cv2.namedWindow(DOT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(DOT_WINDOW_NAME, SUB_DISPLAY_X, SUB_DISPLAY_Y)
    cv2.resizeWindow(DOT_WINDOW_NAME, GRID_W, GRID_H)

    cv2.namedWindow(CAM_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(CAM_WINDOW_NAME, SUB_DISPLAY_X, SUB_DISPLAY_Y + GRID_H + 40)

    countdown_start = time.time()
    last_move_time = time.time()
    red_pt_fixed = GRID_POINTS[spiral[0]]

    os.makedirs("landmarks", exist_ok=True)
    _csv_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"landmarks/landmarks_{_csv_ts}.csv"

    logger.info(f"保存先CSVファイル: {csv_filename}")
    logger.info("検出を開始します。終了するには 'q' を押してください。")

    pose_landmarker = None
    hand_landmarker = None

    # 推論間引き用の状態
    frame_counter = 0
    last_hand_result = None

    # FPS計測用
    fps_window_start = time.time()
    fps_frame_count = 0

    try:
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(build_csv_header())

            while True:
                frame_full, depth_aligned, last_frame_id, failed = grabber.get_latest(
                    last_frame_id, timeout=0.5
                )

                if failed:
                    logger.error(
                        f"フレーム取得が{MAX_CONSECUTIVE_FRAME_FAILURES}回連続で失敗したため、"
                        "安全のため終了します。"
                    )
                    break

                if frame_full is None:
                    # timeout秒以内に新しいフレームが来なかった。
                    # 'q' 判定だけ行いつつ次のフレームを待つ(取得スレッドは裏で回り続けている)。
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                depth_colored = make_depth_display(depth_aligned)

                # 先にDepthと同じ512x424へリサイズしてから推論する(処理を軽くするための変更)
                frame_disp = cv2.resize(frame_full, (KINECT_DEPTH_W, KINECT_DEPTH_H), interpolation=cv2.INTER_LINEAR)

                # MediaPipe推論: リサイズ後(512x424)で実行
                mp_image = Image(image_format=ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
                timestamp_ms = int(time.time() * 1000)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

                # Hand Landmarkerは間引いて実行し、負荷を下げる
                frame_counter += 1
                if frame_counter % HAND_DETECT_EVERY_N_FRAMES == 0 or last_hand_result is None:
                    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                    last_hand_result = hand_result
                else:
                    hand_result = last_hand_result

                # カウントダウン判定
                elapsed_total = time.time() - countdown_start
                remaining = COUNTDOWN_SEC - elapsed_total

                if remaining > 0:
                    countdown_num = int(remaining) + 1
                    last_move_time = time.time()
                    dot_img = draw_dot_grid(GRID_POINTS, red_pt_fixed, DOT_RADIUS, countdown=countdown_num)
                    red_pt = red_pt_fixed
                else:
                    red_pt, spiral_idx, last_move_time = update_spiral(
                        spiral_idx, last_move_time, spiral, GRID_POINTS
                    )
                    dot_img = draw_dot_grid(GRID_POINTS, red_pt, DOT_RADIUS)

                annotated_frame = frame_disp.copy()
                draw_landmarks(annotated_frame, pose_result, hand_result, depth_aligned)
                draw_landmarks(depth_colored, pose_result, hand_result, depth_aligned)

                cv2.imshow(DOT_WINDOW_NAME, dot_img)
                cv2.imshow(CAM_WINDOW_NAME, np.hstack([annotated_frame, depth_colored]))

                if remaining <= 0:
                    write_csv(writer, timestamp_ms, spiral_idx, red_pt,
                              pose_result, hand_result, depth_aligned, frame_full.shape)

                # 実測FPSを定期的にログ出力
                fps_frame_count += 1
                now = time.time()
                if now - fps_window_start >= 5.0:
                    fps = fps_frame_count / (now - fps_window_start)
                    logger.info(f"実測FPS: {fps:.1f}")
                    fps_window_start = now
                    fps_frame_count = 0

                if cv2.waitKey(DELAY_MS) & 0xFF == ord('q'):
                    break
    finally:
        grabber.stop()
        if pose_landmarker:
            pose_landmarker.close()
        if hand_landmarker:
            hand_landmarker.close()
        device.stop()
        device.close()
        cv2.destroyAllWindows()
        logger.info(f"安全に終了し、CSVを保存しました: {csv_filename}")


if __name__ == "__main__":
    main()