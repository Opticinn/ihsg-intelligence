"""
diagnose.py
===========
Jalankan ini untuk diagnosa kenapa accuracy rendah.

Usage:
    python diagnose.py

Output: analisis label, korelasi fitur, dan rekomendasi fix.
"""

import sys
import numpy as np
import pandas as pd
from app.services.features import build_features, FEATURE_COLS, LABEL_COL
from app.services.ingestion import IHSG_TICKERS

# ── 1. Ambil data satu ticker dulu ──
print("=" * 60)
print("DIAGNOSA: IHSG ML Pipeline")
print("=" * 60)

ticker = "BBCA.JK"
print(f"\n[1] Ambil fitur untuk {ticker}...")
df = build_features(ticker)

print(f"    Total baris: {len(df)}")
print(f"    Rentang tanggal: {df['date'].min().date()} → {df['date'].max().date()}")

# ── 2. Cek distribusi label ──
print(f"\n[2] Distribusi Label (lookahead=3 hari, threshold=1%):")
label_counts = df[LABEL_COL].value_counts().sort_index()
label_names  = {0: "SELL", 1: "HOLD", 2: "BUY"}
for k, n in label_counts.items():
    pct = n / len(df) * 100
    print(f"    {label_names[k]} ({k}): {n} baris ({pct:.1f}%)")

# ── 3. Cek apakah label terlalu "flat" ──
print(f"\n[3] Cek volatilitas harga (apakah pasar sideways?):")
close = df["close_price"]
returns = close.pct_change(3).dropna()
print(f"    Return 3-hari — mean: {returns.mean():.4f}, std: {returns.std():.4f}")
above_1pct = (returns.abs() > 0.01).mean()
print(f"    Baris dengan pergerakan > 1%: {above_1pct:.1%}")
print(f"    → Kalau < 40%, berarti pasar terlalu sideways untuk threshold 1%")

# ── 4. Cek korelasi fitur dengan label ──
print(f"\n[4] Korelasi fitur dengan label (top 10):")
numeric_cols = FEATURE_COLS
corr = df[numeric_cols + [LABEL_COL]].corr()[LABEL_COL].drop(LABEL_COL)
corr_abs = corr.abs().sort_values(ascending=False)
for feat, val in corr_abs.head(10).items():
    direction = "+" if corr[feat] > 0 else "-"
    print(f"    {feat:<20} {direction}{abs(val):.4f}")

# ── 5. Simulasi: coba berbagai threshold & lookahead ──
print(f"\n[5] Simulasi distribusi label dengan berbagai parameter:")
print(f"    {'Lookahead':<12} {'Threshold':<12} {'SELL%':<8} {'HOLD%':<8} {'BUY%':<8} {'Non-HOLD%'}")
print(f"    " + "-" * 60)

from app.services.features import compute_label

for lookahead in [1, 3, 5]:
    for threshold in [0.005, 0.01, 0.015, 0.02]:
        lbl = compute_label(close, lookahead=lookahead, threshold=threshold)
        counts = lbl.value_counts(normalize=True).sort_index()
        sell_pct = counts.get(0, 0) * 100
        hold_pct = counts.get(1, 0) * 100
        buy_pct  = counts.get(2, 0) * 100
        non_hold = 100 - hold_pct
        print(f"    {lookahead:<12} {threshold:<12.3f} {sell_pct:<8.1f} {hold_pct:<8.1f} {buy_pct:<8.1f} {non_hold:.1f}%")

# ── 6. Cek data leakage potential ──
print(f"\n[6] Cek potensi masalah lain:")
nan_pct = df[FEATURE_COLS].isna().mean()
high_nan = nan_pct[nan_pct > 0.05]
if high_nan.empty:
    print(f"    ✅ Tidak ada kolom dengan NaN > 5%")
else:
    print(f"    ⚠️  Kolom dengan banyak NaN: {high_nan.to_dict()}")

# Cek apakah close_price ada di fitur (leakage!)
if "close_price" in FEATURE_COLS:
    print(f"    ⚠️  PERINGATAN: close_price ada di fitur — ini bisa menyebabkan leakage!")
    print(f"       Model belajar dari harga hari ini untuk prediksi 3 hari ke depan = valid.")
    print(f"       Tapi kalau akurasi TRAIN jauh lebih tinggi dari TEST = overfitting.")

# ── 7. Cek train vs test split ──
print(f"\n[7] Cek train/test split untuk {ticker}:")
split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx]
test_df   = df.iloc[split_idx:]
print(f"    Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} ({len(train_df)} baris)")
print(f"    Test : {test_df['date'].min().date()} → {test_df['date'].max().date()} ({len(test_df)} baris)")

# ── 8. Quick sanity check: train accuracy ──
print(f"\n[8] Quick sanity check — apakah model bisa fit training data?")
try:
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    import pickle
    from pathlib import Path

    model_path = Path("mlruns/saved_models/ihsg_xgb_latest.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        model       = payload["model"]
        feat_cols   = payload["feature_cols"]

        # Ambil data semua ticker
        all_dfs = []
        for t in IHSG_TICKERS[:5]:  # sample 5 ticker
            d = build_features(t)
            if d is not None:
                d["ticker_encoded"] = hash(t) % 100
                all_dfs.append(d)

        combined = pd.concat(all_dfs).sort_values("date").reset_index(drop=True)
        split    = int(len(combined) * 0.8)
        train    = combined.iloc[:split]
        test     = combined.iloc[split:]

        X_train = train[feat_cols].astype(float)
        y_train = train[LABEL_COL].astype(int)
        X_test  = test[feat_cols].astype(float)
        y_test  = test[LABEL_COL].astype(int)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))

        print(f"    Train accuracy : {train_acc:.4f}")
        print(f"    Test accuracy  : {test_acc:.4f}")
        gap = train_acc - test_acc
        if gap > 0.15:
            print(f"    ⚠️  Gap {gap:.4f} — model OVERFITTING")
        elif test_acc < 0.36:
            print(f"    ⚠️  Test accuracy di bawah random chance (0.333)")
            print(f"       → Model tidak belajar pola apapun dari data ini")
        else:
            print(f"    ✅ Gap {gap:.4f} — relatif normal")
    else:
        print(f"    ⚠️  File model tidak ditemukan, jalankan training dulu.")
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 60)
print("REKOMENDASI:")
print("=" * 60)
print("""
Lihat angka di [5] — cari kombinasi lookahead & threshold yang:
  • Non-HOLD% di kisaran 50-70% (tidak terlalu banyak HOLD)
  • SELL% dan BUY% seimbang (tidak jauh berbeda)

Rekomendasi umum untuk saham IDX yang volatilitasnya rendah:
  • lookahead = 5 hari (lebih mudah diprediksi dari 3 hari)
  • threshold  = 0.015 (1.5%) atau 0.02 (2%)
""")