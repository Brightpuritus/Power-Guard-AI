"""
theft_core.py - logic behind the web app (no Streamlit code in here, so it can be tested alone).

Needs train_compare.py in the same folder (re-uses its cleaning + feature code, so
uploaded data is processed EXACTLY like the training data).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_compare as tc

FEATURE_LABELS = {
    "mean_90": "การใช้ไฟเฉลี่ย 90 วันล่าสุด",
    "std_90": "ความผันผวนของการใช้ไฟ 90 วันล่าสุด",
    "cv_90": "ความผันผวนสัมพัทธ์ (90 วัน)",
    "cv_365": "ความผันผวนสัมพัทธ์ (1 ปี)",
    "skew_365": "ความเบ้ของการใช้ไฟ (1 ปี)",
    "kurt_365": "ความโด่งของการใช้ไฟ (1 ปี)",
    "iqr_over_median_365": "ช่วงกว้างของการใช้ไฟเทียบค่ากลาง (1 ปี)",
    "drop_ratio_30_90": "การใช้ไฟ 30 วันล่าสุด ÷ 90 วันก่อนหน้า",
    "drop_ratio_90_365": "การใช้ไฟ 90 วันล่าสุด ÷ ช่วงก่อนหน้า (ถึง 1 ปี)",
    "last_vs_first_year": "การใช้ไฟปีล่าสุด ÷ ปีแรก",
    "max_mom_drop": "การลดลงรายเดือนที่มากที่สุด",
    "trend_slope_180": "แนวโน้มการใช้ไฟ 180 วัน (ติดลบ = ลดลง)",
    "zero_day_ratio": "สัดส่วนวันที่ใช้ไฟเป็นศูนย์",
    "longest_zero_streak": "จำนวนวันศูนย์ติดต่อกันนานที่สุด",
    "n_zero_streaks": "จำนวนช่วงที่ใช้ไฟเป็นศูนย์",
    "missing_ratio": "สัดส่วนข้อมูลที่ขาดหาย",
    "flat_line_ratio": "สัดส่วนวันที่ค่าซ้ำกับวันก่อนหน้า",
    "autocorr_lag1": "ความสม่ำเสมอรายวัน (autocorr lag 1)",
    "autocorr_lag7": "ความสม่ำเสมอรายสัปดาห์ (autocorr lag 7)",
    "diff_std_norm": "ความแกว่งของการเปลี่ยนแปลงรายวัน",
    "weekend_weekday_ratio": "การใช้ไฟวันหยุด ÷ วันธรรมดา",
}


def label(name):
    return FEATURE_LABELS.get(name, name)


# ----------------------------------------------------------------------------
# artifacts produced by train_compare.py / data_cleaning.py
# ----------------------------------------------------------------------------
def load_artifacts(results_dir):
    d = Path(results_dir)
    need = [d / "models" / "xgboost.joblib", d / "run_summary.json", d / "test_predictions.csv", d / "features.csv"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files (run train_compare.py first): " + ", ".join(missing))
    summary = json.loads((d / "run_summary.json").read_text())
    preds = pd.read_csv(d / "test_predictions.csv", dtype={"CONS_NO": str})
    feats = pd.read_csv(d / "features.csv", dtype={"CONS_NO": str}).set_index("CONS_NO")
    metrics = pd.read_csv(d / "metrics_table.csv") if (d / "metrics_table.csv").exists() else None
    model = joblib.load(d / "models" / "xgboost.joblib")
    ref = np.sort(preds["score_XGBoost"].to_numpy())  # reference distribution for risk scores
    return {"model": model, "features": summary["features"], "preds": preds, "feats": feats.drop(columns=["FLAG"], errors="ignore"),
            "labels": feats["FLAG"], "metrics": metrics, "ref_scores": ref, "summary": summary}


def load_clean_series(cleaned_dir):
    """Cleaned customers x days table from data_cleaning.py -> (ids, values, dates) or None."""
    d = Path(cleaned_dir)
    if (d / "cleaned_data.parquet").exists():
        df = pd.read_parquet(d / "cleaned_data.parquet")
    elif (d / "cleaned_data.csv").exists():
        df = pd.read_csv(d / "cleaned_data.csv", dtype={"CONS_NO": str})
    else:
        return None
    cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    parsed = pd.to_datetime(pd.Series(cols), errors="coerce")
    dates = pd.DatetimeIndex(parsed.values) if parsed.notna().all() else None
    ids = df["CONS_NO"].astype(str).to_numpy()
    return {"index": {c: i for i, c in enumerate(ids)}, "values": df[cols].to_numpy(dtype=np.float32), "dates": dates}


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def risk_percentile(proba, ref_sorted):
    """0-100: share of TEST customers whose model score is lower (same scale for any upload size)."""
    lo = np.searchsorted(ref_sorted, proba, side="left")
    hi = np.searchsorted(ref_sorted, proba, side="right")
    return (lo + hi) / 2 / len(ref_sorted) * 100  # mid-rank, so tied scores do not all jump to a high percentile


def risk_level(score, high=95, medium=80):
    return np.select([score >= high, score >= medium], ["สูง", "ปานกลาง"], default="ต่ำ")


def score_features(model, feats, ref_sorted, high=95, medium=80):
    names = getattr(model, "feature_names_in_", None)
    X = feats[list(names)] if names is not None else feats
    proba = model.predict_proba(X)[:, 1]
    risk = risk_percentile(proba, ref_sorted)
    return pd.DataFrame({"CONS_NO": feats.index.astype(str), "probability": proba, "risk_score": risk,
                         "risk_level": risk_level(risk, high, medium)}, index=feats.index)


def level_table(risk, y, high=95, medium=80):
    """What happened in the TEST set for each risk level (this is the value proposition of the system)."""
    lv = risk_level(risk, high, medium)
    rows = []
    for name in ("สูง", "ปานกลาง", "ต่ำ"):
        m = lv == name
        rows.append({"ระดับความเสี่ยง": name, "จำนวนมิเตอร์": int(m.sum()), "ผู้ลักลอบใช้ไฟจริง": int(y[m].sum()),
                     "อัตราที่เจอจริง (%)": round(float(y[m].mean() * 100), 2) if m.any() else 0.0,
                     "สัดส่วนผู้โกงที่จับได้ (%)": round(float(y[m].sum() / max(y.sum(), 1) * 100), 1)})
    return pd.DataFrame(rows)


def explain(model, feats):
    """Per-customer feature contributions in log-odds (positive = pushes risk UP). Uses XGBoost's built-in SHAP values."""
    if not hasattr(model, "get_booster"):
        return None
    import xgboost as xgb
    booster = model.get_booster()
    X = feats[booster.feature_names] if booster.feature_names else feats
    best = getattr(model, "best_iteration", None)
    rounds = (0, best + 1) if best is not None else (0, 0)
    contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=rounds)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=feats.index)


def top_reasons(contrib_row, value_row, population_median, k=5):
    c = contrib_row.sort_values(key=lambda s: s.abs(), ascending=False).head(k)
    return pd.DataFrame({
        "ปัจจัย": [label(n) for n in c.index],
        "ค่าของมิเตอร์นี้": [value_row[n] for n in c.index],
        "ค่ากลางของลูกค้าทั้งหมด": [population_median[n] for n in c.index],
        "ผลต่อความเสี่ยง": ["เพิ่มขึ้น" if v > 0 else "ลดลง" for v in c.values],
        "contribution": c.values,
    })


def series_frame(values, dates=None, window=30):
    idx = dates if dates is not None else pd.RangeIndex(len(values))
    s = pd.Series(values, index=idx)
    return pd.DataFrame({"รายวัน (kWh)": s, f"เฉลี่ย {window} วัน": s.rolling(window, min_periods=10).mean()})


# ----------------------------------------------------------------------------
# new data (upload)
# ----------------------------------------------------------------------------
def prepare_upload(df, feature_cols, min_days=365):
    """SGCC-format table (CONS_NO, [FLAG], one column per day) -> features, cleaned series, dates."""
    if "CONS_NO" not in df.columns:
        raise ValueError("ไฟล์ต้องมีคอลัมน์ CONS_NO")
    if df.empty:
        raise ValueError("ไฟล์ต้องมีข้อมูลผู้ใช้ไฟอย่างน้อย 1 ราย")
    df = df.copy()
    if df["CONS_NO"].isna().any() or df["CONS_NO"].astype(str).str.strip().eq("").any():
        raise ValueError("รหัส CONS_NO ต้องไม่ว่าง")
    df["CONS_NO"] = df["CONS_NO"].astype(str).str.strip()
    if df["CONS_NO"].duplicated().any():
        raise ValueError("พบรหัส CONS_NO ซ้ำ กรุณาตรวจสอบข้อมูลก่อนอัปโหลด")
    if "FLAG" in df.columns:
        flags = pd.to_numeric(df["FLAG"], errors="coerce")
        if not flags.isin([0, 1]).all():
            raise ValueError("FLAG ต้องเป็น 0 หรือ 1 เท่านั้น")
        df["FLAG"] = flags
    df = df.drop_duplicates(subset="CONS_NO").reset_index(drop=True)
    date_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    if len(date_cols) < min_days:
        raise ValueError(f"ต้องมีข้อมูลรายวันอย่างน้อย {min_days} วัน (พบ {len(date_cols)} คอลัมน์)")
    parsed = pd.to_datetime(pd.Series(date_cols), errors="coerce")
    if parsed.isna().any() or parsed.duplicated().any():
        raise ValueError("คอลัมน์การใช้ไฟต้องเป็นวันที่ที่ถูกต้องและไม่ซ้ำกัน")
    dates, dow = None, None
    if parsed.notna().all():
        order = np.argsort(parsed.values)
        date_cols = [date_cols[i] for i in order]
        dates = pd.DatetimeIndex(parsed.iloc[order].values)
        dow = dates.dayofweek.to_numpy()
    raw = df[date_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    raw[raw < 0] = np.nan
    clean = tc.clean_series(raw)
    feats = tc.build_features(clean, raw, dow)
    feats.index = df["CONS_NO"].astype(str).to_numpy()
    absent = [c for c in feature_cols if c not in feats.columns]
    feats = feats.reindex(columns=feature_cols).fillna(0)
    y = df["FLAG"].astype(int).to_numpy() if "FLAG" in df.columns else None
    return {"feats": feats, "clean": clean, "dates": dates, "absent": absent, "y": y}
