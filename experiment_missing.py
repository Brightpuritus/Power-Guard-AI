"""Controlled missing-data/feature ablation. Test is evaluated after selection.
Run: python experiment_missing.py --out results_missing
"""
import argparse
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import train_compare as tc
from improve_recall import recall_threshold, metrics


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
    f['longest_zero_streak'], f['n_zero_streaks'] = tc._runs(raw == 0)
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
        f[f'missing_run_{window}'], f[f'missing_runs_{window}'] = tc._runs(np.isnan(x))
    for month in range(12):
        end = raw.shape[1]-month*30
        block = raw[:, end-30:end]
        f[f'month_mean_lag{month}'] = np.nanmean(block, axis=1)
        f[f'month_observed_lag{month}'] = np.isfinite(block).mean(1)
    f['missing_change_30_90'] = np.isnan(raw[:, -30:]).mean(1)-np.isnan(raw[:, -120:-30]).mean(1)
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)


def make_features(raw, dow, variant):
    base = observed_features(raw, dow) if variant in ('missing_only', 'combined') else tc.build_features(tc.clean_series(raw), raw, dow)
    if variant in ('features_only', 'combined'):
        base = pd.concat([base, extra_features(raw)], axis=1)
    return base


def score_raw(bundle, raw, dow):
    """Inference entry point for this experiment (raw customers x days, sorted dates)."""
    raw = np.where(np.isfinite(raw) & (raw >= 0), raw, np.nan)
    X = make_features(raw, dow, bundle['variant'])[bundle['features']]
    scores = bundle['model'].predict_proba(X)[:, 1]
    return scores, scores >= bundle['threshold']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_missing')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        ap.error('Use a new directory to preserve experiments')
    out.mkdir(parents=True)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    ids, y, raw, dow = tc.load_sgcc('data.csv')
    src = Path('results_recall')
    summary = json.loads((src/'run_summary.json').read_text())
    latest = joblib.load(src/'models/xgboost.joblib')
    base = pd.read_csv(src/'features.csv', dtype={'CONS_NO':str})
    assert np.array_equal(ids, base.CONS_NO) and np.array_equal(y, base.FLAG)
    tv, te = train_test_split(np.arange(len(y)), test_size=.15, stratify=y, random_state=42)
    tr, va = train_test_split(tv, test_size=.15/.85, stratify=y[tv], random_state=42)
    saved = pd.read_csv(src/'test_predictions.csv', dtype={'CONS_NO':str}).set_index('CONS_NO')
    assert set(ids[te]) == set(saved.index)
    floor = summary['recall_experiment']['validation_precision_floor']
    objects = {'latest': (latest, base[summary['features']], summary['thresholds']['XGBoost'])}
    val = [metrics('latest', y[va], latest.predict_proba(objects['latest'][1].iloc[va])[:,1], objects['latest'][2])]
    params = latest.get_params()
    for variant in ('missing_only', 'features_only', 'combined'):
        print('Building '+variant, flush=True)
        X = make_features(raw, dow, variant)
        assert not np.isinf(X.to_numpy()).any()
        model = XGBClassifier(**params)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        scores = model.predict_proba(X.iloc[va])[:,1]
        threshold = recall_threshold(y[va], scores, floor)
        if threshold is None:
            raise RuntimeError('No feasible validation threshold: '+variant)
        row = metrics(variant, y[va], scores, threshold)
        val.append(row)
        objects[variant] = model, X, threshold
        print(f"{variant}: {X.shape[1]} features, val P={row['Precision']:.4f} R={row['Recall']:.4f} AP={row['PR-AUC']:.4f}", flush=True)
    winner = max(val, key=lambda r:(r['Recall'],r['Precision'],r['PR-AUC']))['Model']
    selection = {'winner':winner, 'precision_floor':floor, 'rule':'Validation recall then precision then AP, including latest baseline',
                 'same_hyperparameters':True, 'validation':val}
    (out/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    pd.DataFrame(val).to_csv(out/'validation_comparison.csv', index=False)
    # All test variants reported as a predeclared ablation, not for choosing a winner.
    rows=[]
    for variant, (model,X,threshold) in objects.items():
        scores=model.predict_proba(X.iloc[te])[:,1]
        if variant == 'latest':
            np.testing.assert_allclose(scores,saved.loc[ids[te],'score_XGBoost'],atol=1e-7)
        rows.append(metrics(variant,y[te],scores,threshold))
        bundle={'model':model,'variant':variant,'threshold':threshold,'features':list(X.columns)}
        joblib.dump(bundle,out/(variant+'.joblib'))
        pd.DataFrame({'CONS_NO':ids[te],'y_true':y[te],'score':scores}).to_csv(out/(variant+'_test_predictions.csv'),index=False)
        # Inference parity on held-out raw rows, independent of cached feature table.
        actual,_=score_raw(bundle,raw[te[:25]],dow)
        np.testing.assert_allclose(actual,scores[:25],atol=1e-7)
    table=pd.DataFrame(rows)
    table.to_csv(out/'test_comparison.csv',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    xx=np.arange(len(table))
    for i,col in enumerate(['Precision','Recall','F1','PR-AUC']):
        ax.bar(xx+i*.19,table[col],width=.19,label=col)
    ax.set(xticks=xx+.285,xticklabels=table.Model,ylim=(0,.65),title='Missing data and feature ablation (test)')
    ax.legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    print('Selected on validation: '+winner,flush=True)
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
