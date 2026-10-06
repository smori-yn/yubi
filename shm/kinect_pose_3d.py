#!/usr/bin/env python3
"""
kinect_pose_3d.py  ─  Tasks API 版 (mediapipe >= 0.10.30)

共有メモリ経由で C++ サーバーから RGB + Depth を受け取り、
MediaPipe Tasks PoseLandmarker で 2D 推定 → Depth 融合 → 3D 骨格をリアルタイム表示する。

起動前に一度だけモデルをダウンロードしておく:
  wget -O pose_landmarker_full.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

起動順:
  1. $ ./build/kinect_shm_server        # C++ サーバーを先に起動
  2. $ python3 kinect_pose_3d.py        # 本スクリプトを起動
     (モデルパスを変えたい場合)
     $ python3 kinect_pose_3d.py --model /path/to/pose_landmarker_full.task
"""

import argparse
import mmap
import os
import struct
import sys
import time

import cv2
import numpy as np

# -- Tasks API インポート -------------------------------------------------------
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

BaseOptions       = mp_python.BaseOptions
VisionRunningMode = mp_vision.RunningMode

# -- 共有メモリ定数 (C++ 側と完全一致) ----------------------------------------
SHM_NAME       = "/kinect2_frames"
RGB_W, RGB_H   = 1920, 1080
DEP_W, DEP_H   = 512,  424
RGB_BYTES      = RGB_W * RGB_H * 3           # uint8 BGR
DEP_BYTES      = DEP_W * DEP_H * 4           # float32 mm
HDR_BYTES      = 8                            # seq(4) + flag(4)
SHM_SIZE       = HDR_BYTES + RGB_BYTES + DEP_BYTES
OFF_SEQ        = 0
OFF_FLAG       = 4
OFF_RGB        = HDR_BYTES
OFF_DEP        = HDR_BYTES + RGB_BYTES

IR_PARAMS_FILE = "/tmp/kinect2_ir_params.txt"
MODEL_DEFAULT  = "pose_landmarker_full.task"

# -- MediaPipe 骨格ライン定義 --------------------------------------------------
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(29,31),(28,30),(30,32),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
]
# Z距離を表示する主要関節 ID
KEY_JOINTS = {0, 11, 12, 23, 24}


# -----------------------------------------------------------------------------
# IR パラメータ読み込み
# -----------------------------------------------------------------------------
def load_ir_params(path: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            params = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        params[k.strip()] = float(v.strip())
            return params["fx"], params["fy"], params["cx"], params["cy"]
        print("[Python] Waiting for IR params ...", end="\r", flush=True)
        time.sleep(0.3)
    raise TimeoutError(
        f"IR params not found: {path}\n"
        "kinect_shm_server が起動しているか確認してください。"
    )


# -----------------------------------------------------------------------------
# 共有メモリリーダー
# -----------------------------------------------------------------------------
def _open_shm(timeout: float = 20.0) -> mmap.mmap:
    deadline = time.time() + timeout
    while time.time() < deadline:
        shm_path = f"/dev/shm{SHM_NAME}"
        if os.path.exists(shm_path):
            fd = os.open(shm_path, os.O_RDONLY)
            mm = mmap.mmap(fd, SHM_SIZE, access=mmap.ACCESS_READ)
            os.close(fd)
            return mm
        print("[Python] Waiting for kinect_shm_server ...", end="\r", flush=True)
        time.sleep(0.5)
    raise TimeoutError("Shared memory not available. kinect_shm_server を先に起動してください。")


class ShmReader:
    def __init__(self):
        self._mm       = _open_shm()
        self._last_seq = -1

    def wait_new_frame(self, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            seq  = struct.unpack_from("<I", self._mm, OFF_SEQ)[0]
            flag = struct.unpack_from("<I", self._mm, OFF_FLAG)[0]
            if flag == 1 and seq != self._last_seq:
                self._last_seq = seq
                return True
            time.sleep(0.001)
        return False

    def read_rgb(self) -> np.ndarray:
        """BGR uint8 shape=(1080, 1920, 3)"""
        raw = self._mm[OFF_RGB: OFF_RGB + RGB_BYTES]
        return np.frombuffer(raw, dtype=np.uint8).reshape(RGB_H, RGB_W, 3).copy()

    def read_depth(self) -> np.ndarray:
        """float32 mm shape=(424, 512)"""
        raw = self._mm[OFF_DEP: OFF_DEP + DEP_BYTES]
        return np.frombuffer(raw, dtype=np.float32).reshape(DEP_H, DEP_W).copy()

    def close(self):
        self._mm.close()


# -----------------------------------------------------------------------------
# Depth 融合 → 3D 座標変換
# -----------------------------------------------------------------------------
class DepthFusion:
    """
    MediaPipe Tasks API が返す NormalizedLandmark (x, y: 0–1) と
    Kinect Depth マップ (512×424, mm) から 3D 座標 [X, Y, Z] mm を求める。
    """
    _SCALE_X = DEP_W / RGB_W   # 512/1920
    _SCALE_Y = DEP_H / RGB_H   # 424/1080

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def landmark_to_3d(self, lm: NormalizedLandmark, depth: np.ndarray) -> np.ndarray:
        """lm.x, lm.y は 0–1 正規化座標。戻り値 mm。"""
        u = int(np.clip(lm.x * RGB_W * self._SCALE_X, 0, DEP_W - 1))
        v = int(np.clip(lm.y * RGB_H * self._SCALE_Y, 0, DEP_H - 1))

        Z = float(depth[v, u])
        # libfreenect2 は無効ピクセルを 0 だけでなく NaN で返すことがあるため
        # isfinite でも弾く
        if not np.isfinite(Z) or Z <= 0.0:
            Z = self._nn_depth(depth, u, v)
        if not np.isfinite(Z) or Z <= 0.0:
            return np.zeros(3)

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        xyz = np.array([X, Y, Z])
        # 念のため最終チェック(NaN/Infが残っていたら0扱い)
        if not np.all(np.isfinite(xyz)):
            return np.zeros(3)
        return xyz

    def landmarks_to_3d(self, landmarks: list, depth: np.ndarray) -> list:
        """33 点すべてを変換して list[np.ndarray] で返す。"""
        return [self.landmark_to_3d(lm, depth) for lm in landmarks]

    @staticmethod
    def _nn_depth(depth: np.ndarray, u: int, v: int, r: int = 5) -> float:
        roi   = depth[max(0, v-r): v+r+1, max(0, u-r): u+r+1]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0


# -----------------------------------------------------------------------------
# 描画ユーティリティ
# -----------------------------------------------------------------------------
def draw_skeleton_2d(img: np.ndarray,
                     landmarks: list,      # list[NormalizedLandmark]  33点
                     joints_3d: list,
                     disp_w: int, disp_h: int):
    """縮小後の表示画像上に 2D 骨格 + Z ラベルを描画する。"""
    def pt(lm):
        return (int(lm.x * disp_w), int(lm.y * disp_h))

    for a, b in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            cv2.line(img, pt(landmarks[a]), pt(landmarks[b]),
                     (0, 220, 0), 2, cv2.LINE_AA)

    for i, lm in enumerate(landmarks):
        p   = pt(lm)
        xyz = joints_3d[i] if i < len(joints_3d) else np.zeros(3)
        col = (0, 255, 255) if (lm.visibility or 0) > 0.5 else (100, 100, 0)
        cv2.circle(img, p, 4, col, -1)
        # Z が有効な値のときだけラベルを描画(NaN/0はスキップ)
        if xyz[2] > 0 and np.isfinite(xyz[2]) and i in KEY_JOINTS:
            cv2.putText(img, f"{xyz[2]/1000:.2f}m",
                        (p[0]+6, p[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)


def make_depth_colormap(depth: np.ndarray, max_mm: float = 5000.0) -> np.ndarray:
    # NaN/Inf があると colormap 変換で壊れるので事前に潰しておく
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = (np.clip(depth, 0, max_mm) / max_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)


def overlay_on_depth(dep_img: np.ndarray, landmarks: list):
    """Depth カラー画像上に関節点を打つ。"""
    sx = DEP_W / RGB_W
    sy = DEP_H / RGB_H
    for lm in landmarks:
        u = int(np.clip(lm.x * RGB_W * sx, 0, DEP_W - 1))
        v = int(np.clip(lm.y * RGB_H * sy, 0, DEP_H - 1))
        cv2.circle(dep_img, (u, v), 3, (255, 255, 255), -1)


def fmt_mm(v: float) -> str:
    """NaN/Infを安全に表示用文字列にする(A/N落ち防止)。"""
    if not np.isfinite(v):
        return "  N/A "
    return f"{v:+7.0f}"


# -----------------------------------------------------------------------------
# メインループ
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT,
                        help=f"pose_landmarker_*.task のパス (default: {MODEL_DEFAULT})")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] モデルファイルが見つかりません: {args.model}")
        print("以下のコマンドでダウンロードしてください:")
        print("  wget -O pose_landmarker_full.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        sys.exit(1)

    # -- IR パラメータ & 共有メモリ --
    print("[Python] Loading IR camera parameters ...")
    fx, fy, cx, cy = load_ir_params(IR_PARAMS_FILE)
    print(f"[Python] fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    print("[Python] Opening shared memory ...")
    reader = ShmReader()
    print("[Python] Shared memory OK.\n")

    fusion = DepthFusion(fx, fy, cx, cy)

    # -- Tasks API: PoseLandmarker (VIDEO モード) ------------------------------
    #
    # VIDEO モードを選ぶ理由:
    #   - LIVE_STREAM はコールバック非同期のため Depth との同期が難しい
    #   - IMAGE モードは毎フレーム検出器を走らせ低速
    #   - VIDEO モードはトラッキングを維持しつつ同期的に結果を返す → 最適
    #
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    DISP_W, DISP_H = 960, 540
    fps_counter = 0
    fps_val     = 0.0
    fps_t0      = time.time()
    last_print  = -1

    print("[Python] Starting main loop. Press 'q' to quit.\n")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            # -- フレーム取得 --
            if not reader.wait_new_frame(timeout=2.0):
                print("[Python] Frame timeout – サーバーが停止しましたか?")
                continue

            rgb   = reader.read_rgb()    # BGR (1080, 1920, 3)
            depth = reader.read_depth()  # float32 mm (424, 512)

            # -- MediaPipe Tasks API 推論 --------------------------------------
            # mp.Image は RGB 順で渡す
            rgb_for_mp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_for_mp)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # -- Depth 融合 & 描画 ---------------------------------------------
            disp = cv2.resize(rgb, (DISP_W, DISP_H))
            cv2.imshow("test", disp)
            print(result.pose_landmarks)

            if result.pose_landmarks:
                # Tasks API: result.pose_landmarks は list[list[NormalizedLandmark]]
                # 1人トラッキングなので [0] を使用
                landmarks = result.pose_landmarks[0]   # list[NormalizedLandmark] 33点

                joints_3d = fusion.landmarks_to_3d(landmarks, depth)

                draw_skeleton_2d(disp, landmarks, joints_3d, DISP_W, DISP_H)
                cv2.imshow("test2", disp)
                # 右上に Depth カラーマップを合成
                dep_col = make_depth_colormap(depth)
                overlay_on_depth(dep_col, landmarks)
                dep_small = cv2.resize(dep_col, (320, 266))
                disp[0:266, DISP_W-320:DISP_W] = dep_small

                # コンソール出力 (毎秒 1 回)
                now_s = int(time.time())
                if now_s != last_print:
                    last_print = now_s
                    nose   = joints_3d[0]
                    l_sh   = joints_3d[11]
                    r_sh   = joints_3d[12]
                    l_hip  = joints_3d[23]
                    r_hip  = joints_3d[24]
                    print(f"  Nose    X={fmt_mm(nose[0])}  Y={fmt_mm(nose[1])}  Z={fmt_mm(nose[2])} mm")
                    print(f"  L.Shldr X={fmt_mm(l_sh[0])}  Y={fmt_mm(l_sh[1])}  Z={fmt_mm(l_sh[2])} mm")
                    print(f"  R.Shldr X={fmt_mm(r_sh[0])}  Y={fmt_mm(r_sh[1])}  Z={fmt_mm(r_sh[2])} mm")
                    print(f"  L.Hip   X={fmt_mm(l_hip[0])}  Y={fmt_mm(l_hip[1])}  Z={fmt_mm(l_hip[2])} mm")
                    print(f"  R.Hip   X={fmt_mm(r_hip[0])}  Y={fmt_mm(r_hip[1])}  Z={fmt_mm(r_hip[2])} mm")
            else:
                # 右上に Depth のみ表示
                dep_small = cv2.resize(make_depth_colormap(depth), (320, 266))
                disp[0:266, DISP_W-320:DISP_W] = dep_small

            # -- FPS 表示 --
            fps_counter += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                fps_val     = fps_counter / elapsed
                fps_counter = 0
                fps_t0      = time.time()
            cv2.putText(disp, f"FPS: {fps_val:.1f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Kinect v2 – 3D Skeleton (Tasks API)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    reader.close()
    cv2.destroyAllWindows()
    print("[Python] Done.")


if __name__ == "__main__":
    main()
