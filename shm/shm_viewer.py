#!/usr/bin/env python3
"""
shm_viewer.py  ─  共有メモリ確認用ビューワー（MediaPipe なし）

kinect_shm_server が書き出した RGB + Depth を読んで表示するだけ。
映像が出れば共有メモリ経由のパイプラインは正常。

起動:
  python3 shm_viewer.py
終了:
  q キー
"""

import mmap, struct, os, time
import cv2
import numpy as np

# ── 共有メモリ定数（C++ 側と一致） ──────────────────────────────────────────
SHM_NAME     = "/kinect2_frames"
RGB_W, RGB_H = 1920, 1080
DEP_W, DEP_H = 512,  424
RGB_BYTES    = RGB_W * RGB_H * 3
DEP_BYTES    = DEP_W * DEP_H * 4   # float32
HDR_BYTES    = 8
SHM_SIZE     = HDR_BYTES + RGB_BYTES + DEP_BYTES
OFF_SEQ      = 0
OFF_FLAG     = 4
OFF_RGB      = HDR_BYTES
OFF_DEP      = HDR_BYTES + RGB_BYTES

# ── 共有メモリを開く ──────────────────────────────────────────────────────────
shm_path = f"/dev/shm{SHM_NAME}"
print(f"[viewer] 共有メモリを待機中: {shm_path}")
while not os.path.exists(shm_path):
    time.sleep(0.5)
    print(".", end="", flush=True)
print("\n[viewer] 接続完了")

fd = os.open(shm_path, os.O_RDONLY)
mm = mmap.mmap(fd, SHM_SIZE, access=mmap.ACCESS_READ)
os.close(fd)

last_seq = -1
fps_counter = 0
fps_val = 0.0
fps_t0  = time.time()

print("[viewer] 表示開始。q で終了。")

while True:
    # ── 新フレーム待ち ────────────────────────────────────────────────────────
    seq  = struct.unpack_from("<I", mm, OFF_SEQ)[0]
    flag = struct.unpack_from("<I", mm, OFF_FLAG)[0]

    if flag != 1 or seq == last_seq:
        time.sleep(0.002)
        continue
    last_seq = seq

    # ── RGB 読み取り ──────────────────────────────────────────────────────────
    raw_rgb = mm[OFF_RGB: OFF_RGB + RGB_BYTES]
    rgb = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(RGB_H, RGB_W, 3).copy()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # ── Depth 読み取り ────────────────────────────────────────────────────────
    raw_dep = mm[OFF_DEP: OFF_DEP + DEP_BYTES]
    depth = np.frombuffer(raw_dep, dtype=np.float32).reshape(DEP_H, DEP_W).copy()

    # ── Depth → カラーマップ変換 ──────────────────────────────────────────────
    d_norm = (np.clip(depth, 0, 5000) / 5000 * 255).astype(np.uint8)
    dep_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

    # ── 統計（コンソール、毎秒1回）────────────────────────────────────────────
    fps_counter += 1
    elapsed = time.time() - fps_t0
    if elapsed >= 1.0:
        fps_val     = fps_counter / elapsed
        fps_counter = 0
        fps_t0      = time.time()

        valid = depth[depth > 0]
        d_min = valid.min() if len(valid) else 0
        d_max = valid.max() if len(valid) else 0
        d_med = float(np.median(valid)) if len(valid) else 0
        d_ctr = float(depth[DEP_H//2, DEP_W//2])
        print(f"[seq={seq:06d}] FPS={fps_val:.1f}  "
              f"Depth min={d_min:.0f} med={d_med:.0f} max={d_max:.0f} mm  "
              f"center={d_ctr:.0f} mm  "
              f"valid={len(valid)}/{DEP_W*DEP_H} px")

    # ── 表示（RGB を半分に縮小、Depth を右に並べる）──────────────────────────
    rgb_small = cv2.resize(rgb, (960, 540))
    dep_small = cv2.resize(dep_color, (424, 374))   # 縦横比を合わせて調整

    # FPS をオーバーレイ
    cv2.putText(rgb_small, f"Color  FPS:{fps_val:.1f}  seq:{seq}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(dep_small, f"Depth  {depth[DEP_H//2,DEP_W//2]:.0f}mm (center)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Color (960x540)", rgb_small)
    cv2.imshow("Depth colormap (TURBO, 0-5m)", dep_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

mm.close()
cv2.destroyAllWindows()
print("[viewer] 終了")
