"""Stage 1: MLP指示位置推定(ローカル版)

Colab版からの変更点:
  - files.upload() を glob によるフォルダ読み込みに置き換え
  - %matplotlib inline / google.colab の依存を除去
  - GroupKFold の n_splits をグループ数に合わせて自動調整
  - 収録失敗ファイルの除外リストを追加

使い方:  python train_mlp.py
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

# ============================================================
# 設定(ここだけ書き換えれば実験条件を変えられる)
# ============================================================
BASE = "/home/sota/land_test"

# 学習に使うファイルのglobパターン(複数指定可)
TRAIN_PATTERNS = [f"{BASE}/goto/landmarks_20260722_*.csv"]

# テストに使うファイル(1ファイル)
TEST_FILE = f"{BASE}/mori/landmarks_20260728_143529.csv"

# 収録失敗ファイル(指が動いておらず target と相関しない。詳細は下のコメント)
#   105618: index_tip の可動域 65px / 相関 -0.25
#   110030: index_tip の可動域 24px / 相関 -0.19
#   いずれも正常ファイルより tip_y が 400〜500px 下にあり、指を下ろしたままの収録とみられる
EXCLUDE = {
    "landmarks_20260722_105618.csv",
    "landmarks_20260722_110030.csv",
}

# px→cm換算係数(画面: 水平144cm/800px, 垂直107cm/600px)
CM_PER_PX_X = 144 / 800    # 0.18 cm/px
CM_PER_PX_Y = 107 / 600    # 約0.1783 cm/px

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# グリッドサーチの探索範囲
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(64, 64), (128, 128), (384, 384)],
    "mlp__alpha": [0.001, 0.01, 0.1],
}
MAX_ITER = 10000
EARLY_STOPPING = False


# ============================================================
# ① 学習用CSV読み込み(ファイル単位で session 付与)
# ============================================================
train_files = []
for pat in TRAIN_PATTERNS:
    train_files.extend(glob.glob(pat))
train_files = sorted(f for f in train_files if os.path.basename(f) not in EXCLUDE)

if not train_files:
    raise SystemExit(f"学習ファイルが見つかりません: {TRAIN_PATTERNS}")

train_dfs = []
for file_id, path in enumerate(train_files):
    df = pd.read_csv(path)
    df["session"] = file_id
    print(f"  session {file_id:2d}: {os.path.basename(path)}  {df.shape}")
    train_dfs.append(df)

df_train = pd.concat(train_dfs, ignore_index=True)
print("\ntrain合計:", df_train.shape)
print("ファイル数:", df_train["session"].nunique())

n_excluded = sum(
    1 for pat in TRAIN_PATTERNS for f in glob.glob(pat)
    if os.path.basename(f) in EXCLUDE
)
if n_excluded:
    print(f"除外した収録失敗ファイル: {n_excluded}本")

# ============================================================
# ② テスト用CSV
# ============================================================
df_test = pd.read_csv(TEST_FILE)
print("test:", os.path.basename(TEST_FILE), df_test.shape)

# ============================================================
# ③ 列の定義
# ============================================================
feature_cols = [col for col in df_train.columns if col.startswith("joint_")]
target_cols = ["target_x", "target_y"]

# ============================================================
# ④ 欠損値(NaN)の処理
# ============================================================
print("train NaN行数:", df_train[feature_cols + target_cols].isna().any(axis=1).sum(), "/", len(df_train))
print("test  NaN行数:", df_test[feature_cols + target_cols].isna().any(axis=1).sum(), "/", len(df_test))

df_train = df_train.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
df_test = df_test.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

print("処理後 train:", df_train.shape, " test:", df_test.shape)

# ============================================================
# ⑤ X, yに分割(groups は必ず dropna 後に取得)
# ============================================================
X_train = df_train[feature_cols]
Y_train = df_train[target_cols]
X_test = df_test[feature_cols]
Y_test = df_test[target_cols]

groups = df_train["session"].values

# ============================================================
# ⑥ グリッドサーチ(GroupKFold: ファイル単位)
# ============================================================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(activation="relu", solver="adam",
                         max_iter=MAX_ITER, early_stopping=EARLY_STOPPING)),
])

# n_splits はグループ数を超えられない。ファイル単位 Leave-One-Group-Out にする
n_groups = len(np.unique(groups))
n_splits = n_groups
print(f"GroupKFold: n_splits={n_splits}(グループ数={n_groups})")

grid = GridSearchCV(
    pipe,
    PARAM_GRID,
    cv=GroupKFold(n_splits=n_splits),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2,
)

grid.fit(X_train, Y_train, groups=groups)

print("ベストパラメータ:", grid.best_params_)
print("ベストCV MAE:", -grid.best_score_, "px")

# ============================================================
# ⑦ 全組み合わせの結果一覧
# ============================================================
results = pd.DataFrame(grid.cv_results_)
summary = results[["param_mlp__hidden_layer_sizes",
                   "param_mlp__alpha",
                   "mean_test_score",
                   "std_test_score",
                   "rank_test_score"]].copy()
summary["CV_MAE_px"] = -summary["mean_test_score"]
summary["CV_STD_px"] = summary["std_test_score"]
summary = summary.drop(columns=["mean_test_score", "std_test_score"])
summary = summary.sort_values("CV_MAE_px").reset_index(drop=True)
print(summary.to_string())

best_model = grid.best_estimator_

# ============================================================
# ⑧ 採点(test評価: MAE/MAX/STD + cm換算)
# ============================================================
pred = best_model.predict(X_test)

err_x = np.abs(pred[:, 0] - Y_test.values[:, 0])
err_y = np.abs(pred[:, 1] - Y_test.values[:, 1])
err_x_cm = err_x * CM_PER_PX_X
err_y_cm = err_y * CM_PER_PX_Y

print("=== test評価 ===")
print(f"MAE  x: {err_x.mean():6.1f}px ({err_x_cm.mean():5.2f}cm)   y: {err_y.mean():6.1f}px ({err_y_cm.mean():5.2f}cm)")
print(f"MAX  x: {err_x.max():6.1f}px ({err_x_cm.max():5.2f}cm)   y: {err_y.max():6.1f}px ({err_y_cm.max():5.2f}cm)")
print(f"STD  x: {err_x.std():6.1f}px ({err_x_cm.std():5.2f}cm)   y: {err_y.std():6.1f}px ({err_y_cm.std():5.2f}cm)")

# train側(過学習診断用)
pred_train = best_model.predict(X_train)
err_tx = np.abs(pred_train[:, 0] - Y_train.values[:, 0])
err_ty = np.abs(pred_train[:, 1] - Y_train.values[:, 1])
print("=== train評価(参考) ===")
print(f"MAE  x: {err_tx.mean():6.1f}px   y: {err_ty.mean():6.1f}px")

# ============================================================
# グラフ(GUIが無い環境では画像保存のみ)
# ============================================================
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

plt.figure(figsize=(8, 6))
plt.plot(Y_test.values[:, 0], Y_test.values[:, 1],
         c="blue", linewidth=1, marker="o", markersize=2, label="target")
plt.plot(pred[:, 0], pred[:, 1],
         c="red", linewidth=1, marker="o", markersize=2, alpha=0.6, label="pred")
plt.xlim(0, 800)
plt.ylim(600, 0)
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.legend()
plt.grid(True)
plt.title("target vs prediction")

fig_path = os.path.join(OUT_DIR, "pred_plot.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
print("プロット保存:", fig_path)
if os.environ.get("DISPLAY"):
    plt.show()
