"""
train.py — Latih model dengan 1X2 multiclass + O/U terpisah + class weight + ensemble bobot + versioning.
Versi 7:
- 1X2 dipecah jadi 1 model MULTICLASS (Home/Draw/Away)
- O/U 1.5, 2.5, 3.5 dilatih TERPISAH sebagai 3 model
- Class weight untuk handle imbalance
- Ensemble bobot (XGB + LGBM)
- Cross-validation untuk deteksi overfitting
- Model metadata logging dengan version tag
- ✅ FIX #1.2: Normalisasi dari TRAINING SET SAJA (tidak ada leakage)
- ✅ FIX #3.3: Enforce daily limit pada save_tracking
- ✅ Simpan scaler bersama model untuk digunakan predict.py
"""
import sys
import shutil
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path, MODEL_VERSION, N_LEAGUES, TRAINING_DATE, CV_OVERFITTING_THRESHOLD


FEATURES = [
    "home_form_5", "away_form_5",
    "home_form_10", "away_form_10",
    "home_goals_for_avg_5", "away_goals_for_avg_5",
    "home_goals_against_avg_5", "away_goals_against_avg_5",
    "home_win_rate", "away_win_rate",
    "h2h_avg_goals", "h2h_home_wins",
    "is_home",
    "home_form_std_5", "away_form_std_5",
    "home_clean_sheet_rate_5", "away_clean_sheet_rate_5",
    "home_failed_score_rate_5", "away_failed_score_rate_5",
    "home_rest_days", "away_rest_days",
    "home_congestion_7d", "away_congestion_7d",
]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

ENSEMBLE_WEIGHTS = {
    "1x2_multiclass": {"xgb": 0.6, "lgbm": 0.4},
    "ou_1_5": {"xgb": 0.5, "lgbm": 0.5},
    "ou_2_5": {"xgb": 0.5, "lgbm": 0.5},
    "ou_3_5": {"xgb": 0.5, "lgbm": 0.5},
    "btts": {"xgb": 0.7, "lgbm": 0.3},
}


def load_data():
    p = path("data/processed/matches_features.csv")
    if not p.exists():
        raise FileNotFoundError(f"Input tidak ada: {p}")
    df = pd.read_csv(p, low_memory=False)
    print(f"  Loaded: {len(df)} matches")
    return df


def split_data_timeseries(df, test_size=0.2):
    """Split chronologically menggunakan TimeSeriesSplit concept."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def normalize_features(X_train, X_test, feature_cols):
    """FIX #1.2: Normalisasi HANYA dari TRAINING SET.
    
    Workflow yang benar:
    1. Fit scaler pada X_train
    2. Transform X_train dengan scaler yang sama
    3. Transform X_test (TANPA re-fit) dengan scaler yang sudah fitted
    
    Ini mencegah test set leakage ke scaler computation.
    """
    scaler = StandardScaler()
    # HANYA fit pada training set
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test tanpa re-fit
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler


def prepare_xy(df, label_col):
    cols = [c for c in FEATURES if c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features: {missing}")
    if not cols:
        raise ValueError(f"Tidak ada features yang valid untuk {label_col}")
    sub = df[cols + [label_col]].dropna()
    if len(sub) == 0:
        raise ValueError(f"Semua baris dropped untuk {label_col}")
    X = sub[cols].astype(float)
    y = sub[label_col].astype(int)
    print(f"    Class distribution: {dict(y.value_counts())}")
    return X, y, cols


def cross_validate_model(model_class, X, y, cv_folds=5):
    try:
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = cross_val_score(model_class, X, y, cv=tscv, scoring="accuracy")
        return scores.mean(), scores.std()
    except Exception as e:
        print(f"    CV error: {e}")
        return float("nan"), float("nan")


def train_ensemble(X_train, y_train, X_test, y_test, name, task="binary"):
    print(f"\n  ── {name} ──")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    weights = ENSEMBLE_WEIGHTS.get(name, {"xgb": 0.5, "lgbm": 0.5})

    xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=1)
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)

    lgbm = LGBMClassifier(**LGBM_PARAMS, class_weight="balanced")
    lgbm.fit(X_train, y_train)
    lgbm_prob = lgbm.predict_proba(X_test)

    ens_prob = weights["xgb"] * xgb_prob + weights["lgbm"] * lgbm_prob
    ens_pred = (ens_prob[:, 1] > 0.5).astype(int) if task == "binary" else np.argmax(ens_prob, axis=1)

    try:
        acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
        acc_lgbm = accuracy_score(y_test, lgbm.predict(X_test))
        acc_ens = accuracy_score(y_test, ens_pred)
        ll_ens = log_loss(y_test, ens_prob)
    except Exception as e:
        print(f"    Eval error: {e}")
        acc_xgb = acc_lgbm = acc_ens = ll_ens = float("nan")

    print(f"  XGBoost : acc={acc_xgb:.4f}")
    print(f"  LightGBM: acc={acc_lgbm:.4f}")
    print(f"  Ensemble: acc={acc_ens:.4f}  logloss={ll_ens:.4f}")

    cv_mean, cv_std = cross_validate_model(xgb, X_train, y_train, cv_folds=5)
    print(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")

    overfitting_warning = False
    if not np.isnan(cv_mean):
        gap = acc_ens - cv_mean
        if gap > CV_OVERFITTING_THRESHOLD:
            print(f"  ⚠️  POTENTIAL OVERFITTING: gap={gap:.4f}")
            overfitting_warning = True

    return {"xgb": xgb, "lgbm": lgbm, "weights": weights, "features": list(X_train.columns)}, acc_ens, ll_ens, cv_mean, overfitting_warning


def backup_old_model(name):
    old = path(f"models/{name}.pkl")
    if old.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = path("models/backup")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{name}_{ts}.pkl"
        shutil.copy2(old, backup_path)
        print(f"  Backed up: {backup_path}")


def save_model(model, name, metadata=None):
    backup_old_model(name)
    out = path(f"models/{name}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"  ✓ Model saved: {out}")
    if metadata:
        meta_out = path(f"models/{name}_meta.json")
        with open(meta_out, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  ✓ Metadata: {meta_out}")


def save_scaler(scaler, name):
    """FIX #1.2: Simpan scaler untuk digunakan di predict.py."""
    out = path(f"models/{name}_scaler.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out)
    print(f"  ✓ Scaler saved: {out}")


def main():
    print("=" * 70)
    print("TRAINING VERSI 7 (Draw Model + Separate O/U + FIX Normalization)")
    print("=" * 70)
    print(f"Model Version: {MODEL_VERSION}")
    print(f"N Leagues: {N_LEAGUES}")
    print()

    df = load_data()
    train_df, test_df = split_data_timeseries(df, test_size=0.2)
    print(f"Split: train={len(train_df)}, test={len(test_df)}\n")
    test_df.to_csv(path("data/processed/matches_test.csv"), index=False)

    results = {}
    all_metadata = {
        "model_version": MODEL_VERSION,
        "training_date": TRAINING_DATE,
        "n_leagues": N_LEAGUES,
        "trained_at": datetime.now().isoformat(),
        "normalization": "StandardScaler fitted on training set ONLY (NO LEAKAGE)",
        "models": {}
    }

    # 1X2 multiclass
    print("=" * 70)
    print("Training 1X2 MULTICLASS (Home/Draw/Away)")
    print("=" * 70)
    for d in [train_df, test_df]:
        d["result_enc"] = d["result"].map({"H": 0, "D": 1, "A": 2})
    X_train, y_train, cols = prepare_xy(train_df, "result_enc")
    X_test, y_test, _ = prepare_xy(test_df, "result_enc")
    X_train, X_test, scaler_1x2 = normalize_features(X_train, X_test, cols)
    model_1x2, acc_1x2, ll_1x2, cv_1x2, overfit_1x2 = train_ensemble(X_train, y_train, X_test, y_test, "1x2_multiclass", task="multiclass")
    meta_1x2 = {
        "name": "model_1x2_multiclass",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "task": "multiclass",
        "classes": ["Home", "Draw", "Away"],
        "test_accuracy": float(acc_1x2),
        "test_logloss": float(ll_1x2),
        "cv_mean_accuracy": float(cv_1x2),
        "overfitting_warning": bool(overfit_1x2),
    }
    save_model(model_1x2, "model_1x2_multiclass", meta_1x2)
    save_scaler(scaler_1x2, "model_1x2_multiclass")
    all_metadata["models"]["1x2_multiclass"] = meta_1x2
    results["1x2_multiclass"] = {"acc": acc_1x2, "logloss": ll_1x2}

    # O/U separate models
    for line, label_col, model_name in [(1.5, "over_1_5", "ou_1_5"), (2.5, "over_2_5", "ou_2_5"), (3.5, "over_3_5", "ou_3_5")]:
        print("\n" + "=" * 70)
        print(f"Training O/U {line}")
        print("=" * 70)
        if label_col not in train_df.columns:
            print(f"  ⚠️  Label {label_col} tidak ada di data, skip")
            continue
        X_train, y_train, cols = prepare_xy(train_df, label_col)
        X_test, y_test, _ = prepare_xy(test_df, label_col)
        X_train, X_test, scaler_ou = normalize_features(X_train, X_test, cols)
        model_ou, acc_ou, ll_ou, cv_ou, overfit_ou = train_ensemble(X_train, y_train, X_test, y_test, model_name, task="binary")
        meta_ou = {
            "name": f"model_ou_{int(line*10)}",
            "model_version": MODEL_VERSION,
            "trained_at": datetime.now().isoformat(),
            "ou_line": float(line),
            "task": "binary",
            "test_accuracy": float(acc_ou),
            "test_logloss": float(ll_ou),
            "cv_mean_accuracy": float(cv_ou),
            "overfitting_warning": bool(overfit_ou),
        }
        save_model(model_ou, f"model_ou_{int(line*10)}", meta_ou)
        save_scaler(scaler_ou, f"model_ou_{int(line*10)}")
        all_metadata["models"][f"ou_{int(line*10)}"] = meta_ou
        results[f"ou_{line}"] = {"acc": acc_ou, "logloss": ll_ou}

    # BTTS
    print("\n" + "=" * 70)
    print("Training BTTS (Both Teams To Score)")
    print("=" * 70)
    X_train, y_train, cols = prepare_xy(train_df, "btts")
    X_test, y_test, _ = prepare_xy(test_df, "btts")
    X_train, X_test, scaler_btts = normalize_features(X_train, X_test, cols)
    model_btts, acc_btts, ll_btts, cv_btts, overfit_btts = train_ensemble(X_train, y_train, X_test, y_test, "btts", task="binary")
    meta_btts = {
        "name": "model_btts",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": float(acc_btts),
        "test_logloss": float(ll_btts),
        "cv_mean_accuracy": float(cv_btts),
        "overfitting_warning": bool(overfit_btts),
    }
    save_model(model_btts, "model_btts", meta_btts)
    save_scaler(scaler_btts, "model_btts")
    all_metadata["models"]["btts"] = meta_btts
    results["btts"] = {"acc": acc_btts, "logloss": ll_btts}

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:20s}: acc={v['acc']:.4f}  logloss={v['logloss']:.4f}")

    log_p = path("logs/training_log.json")
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with open(log_p, "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)
    print(f"\n✓ Master training log: {log_p}")
    print(f"✓ Model version: {MODEL_VERSION}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
