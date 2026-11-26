"""Inference features copied from the selected training pipeline; no training dependencies."""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def _runs(mask):
    """longest run of True and number of runs, per row (vectorised over rows)."""
    n, T = mask.shape
    cur = np.zeros(n, dtype=np.int32)
    best = np.zeros(n, dtype=np.int32)
    nruns = np.zeros(n, dtype=np.int32)
    prev = np.zeros(n, dtype=bool)
    for t in range(T):
        m = mask[:, t]
        nruns += (m & ~prev)
        cur = np.where(m, cur + 1, 0)
        best = np.maximum(best, cur)
        prev = m
    return best, nruns

def ratio(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.abs(b) > 1e-6)

def pair_corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(valid, a, np.nan), np.where(valid, b, np.nan)
    da, db = a - np.nanmean(a, axis=1)[:, None], b - np.nanmean(b, axis=1)[:, None]
    result = ratio(np.nansum(da * db, axis=1), np.sqrt(np.nansum(da**2, axis=1) * np.nansum(db**2, axis=1)))
    result[valid.sum(1) < 3] = np.nan
    return result

def observed_features(raw, dow):
    """Same 21 concepts, computed on observed readings without zero imputation.
    Clip upper outliers using observed per-customer mean + 3 std.
    Undefined statistics stay NaN for native XGBoost missing-value handling.
    """
    x = np.ascontiguousarray(raw, dtype=np.float64)
    cap = np.nanmean(x, axis=1) + 3 * np.nanstd(x, axis=1)
    x = np.minimum(x, cap[:, None])
    f = {}
    a, b = x[:, -90:], x[:, -365:]
    mean = lambda z: np.nanmean(z, axis=1)
    f['mean_90'], f['std_90'] = mean(a), np.nanstd(a, axis=1)
    f['cv_90'] = ratio(f['std_90'], f['mean_90'])
    f['cv_365'] = ratio(np.nanstd(b, axis=1), mean(b))
    f['skew_365'], f['kurt_365'] = skew(b, axis=1, nan_policy='omit'), kurtosis(b, axis=1, nan_policy='omit')
    f['iqr_over_median_365'] = ratio(np.nanpercentile(b, 75, axis=1)-np.nanpercentile(b, 25, axis=1), np.nanmedian(b, axis=1))
    f['drop_ratio_30_90'] = np.clip(ratio(mean(x[:, -30:]), mean(x[:, -120:-30])), 0, 10)
    f['drop_ratio_90_365'] = np.clip(ratio(mean(a), mean(x[:, -365:-90])), 0, 10)
    f['last_vs_first_year'] = np.clip(ratio(mean(b), mean(x[:, :365])), 0, 10)
    n, days = x.shape
    blocks = np.nanmean(x[:, -(days//30)*30:].reshape(n, days//30, 30), axis=2)
    f['max_mom_drop'] = np.clip(-np.nanmin(ratio(blocks[:, 1:]-blocks[:, :-1], blocks[:, :-1]), axis=1), 0, 1)
    z = x[:, -180:]
    t = np.where(np.isfinite(z), np.arange(180)[None, :], np.nan)
    dt = t - mean(t)[:, None]
    slope = ratio(np.nansum(dt*(z-mean(z)[:, None]), axis=1), np.nansum(dt**2, axis=1))
    f['trend_slope_180'] = np.clip(ratio(slope*180, mean(z)), -5, 5)
    # Keep raw zero/missing definitions identical to baseline for this ablation.
    f['zero_day_ratio'] = (raw == 0).mean(1)
    f['longest_zero_streak'], f['n_zero_streaks'] = _runs(raw == 0)
    f['missing_ratio'] = np.isnan(raw).mean(1)
    pairs = np.isfinite(x[:, 1:]) & np.isfinite(x[:, :-1])
    f['flat_line_ratio'] = ratio(((np.diff(x, axis=1) == 0) & (x[:, 1:] > 0)).sum(1).astype(float), pairs.sum(1))
    f['autocorr_lag1'], f['autocorr_lag7'] = pair_corr(b[:, :-1], b[:, 1:]), pair_corr(b[:, :-7], b[:, 7:])
    f['diff_std_norm'] = ratio(np.nanstd(np.diff(x, axis=1), axis=1), mean(x))
    wk = dow[-365:] >= 5
    f['weekend_weekday_ratio'] = np.clip(ratio(mean(b[:, wk]), mean(b[:, ~wk])), 0, 10)
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)

def extra_features(raw):
    """Observed-only multi-window and quality features; no fitted population stats."""
    raw = np.ascontiguousarray(raw, dtype=np.float64)
    f = {}
    for window in (30, 90, 180, 365):
        x = raw[:, -window:].astype(float)
        count = np.isfinite(x).sum(1)
        f[f'observed_fraction_{window}'] = count / x.shape[1]
        f[f'observed_mean_{window}'] = np.nanmean(x, axis=1)
        f[f'observed_median_{window}'] = np.nanmedian(x, axis=1)
        f[f'observed_std_{window}'] = np.nanstd(x, axis=1)
        f[f'observed_zero_fraction_{window}'] = ratio((x == 0).sum(1).astype(float), count)
        f[f'missing_run_{window}'], f[f'missing_runs_{window}'] = _runs(np.isnan(x))
    for month in range(12):
        end = raw.shape[1]-month*30
        block = raw[:, end-30:end]
        f[f'month_mean_lag{month}'] = np.nanmean(block, axis=1)
        f[f'month_observed_lag{month}'] = np.isfinite(block).mean(1)
    f['missing_change_30_90'] = np.isnan(raw[:, -30:]).mean(1)-np.isnan(raw[:, -120:-30]).mean(1)
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)

def make_features(raw, dow, variant):
    if variant != "combined":
        raise ValueError("This runtime supports the selected combined model only")
    return pd.concat([observed_features(raw, dow), extra_features(raw)], axis=1)


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
