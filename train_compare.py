#!/usr/bin/env python3
"""
Electricity Theft Detection - train & compare 3 models on the SGCC dataset
===========================================================================

Models
  1. Logistic Regression   (baseline, supervised, tabular features)
  2. XGBoost               (main model, supervised, tabular features)
  3. Autoencoder           (unsupervised, raw daily series; trained on NORMAL
                            customers only, anomaly score = reconstruction error)
     -> falls back to Isolation Forest if PyTorch is missing or --third-model isoforest

Install
  pip install numpy pandas scikit-learn matplotlib joblib xgboost torch shap

Run
  python train_compare.py --data data.csv --out results
  python train_compare.py --data data.csv --out results --cv      # + 5-fold CV report
  python train_compare.py --data data.csv --out results --quick   # smoke test (small + fast)

Input
  SGCC "data.csv" (Kaggle: Electricity Theft Detection Dataset):
  columns = CONS_NO, FLAG, <one column per day>

Split
  70 / 15 / 15 train / validation / test, stratified by customer.
  - validation: early stopping + choosing the decision threshold (best F1)
  - test: touched once, for the final report only

Outputs (in --out)
  metrics_table.csv / .md / .png      comparison table (test set)
  pr_curves.png, roc_curves.png, confusion_matrices.png, metrics_bar.png,
  precision_at_k.png, feature_importance_xgb.png, shap_summary.png,
  examples_best_model.png, autoencoder_loss.png
  test_predictions.csv                per-customer scores (use it in the dashboard)
  features.csv, models/               saved features and trained models
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
EPS = 1e-6
COLORS = {"Logistic Regression": "#1f77b4", "XGBoost": "#d62728", "Autoencoder": "#2ca02c",
          "Isolation Forest": "#2ca02c"}


# ----------------------------------------------------------------------------
# 1. Load + clean
# ----------------------------------------------------------------------------
def load_sgcc(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="CONS_NO").reset_index(drop=True)
    date_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    parsed = pd.to_datetime(pd.Series(date_cols), errors="coerce")
    dow = None
    if parsed.notna().all():  # sort day columns chronologically (the raw file is not always sorted)
        order = np.argsort(parsed.values)
        date_cols = [date_cols[i] for i in order]
        dow = parsed.iloc[order].dt.dayofweek.to_numpy()
    ids = df["CONS_NO"].astype(str).to_numpy()
    y = df["FLAG"].astype(int).to_numpy()
    raw = df[date_cols].to_numpy(dtype=np.float32)
    raw[raw < 0] = np.nan
    return ids, y, raw, dow


def clean_series(raw, max_gap=7):
    """negatives -> NaN, short gaps interpolated, rest -> 0, clip at mean + 3*std per customer."""
    arr = np.where(raw >= 0, raw, np.nan)
    df = pd.DataFrame(arr).T.interpolate(limit=max_gap, limit_direction="both").T
    arr = df.fillna(0).to_numpy(dtype=np.float32)
    upper = arr.mean(1, keepdims=True) + 3 * arr.std(1, keepdims=True)
    return np.minimum(arr, upper).astype(np.float32)


# ----------------------------------------------------------------------------
# 2. Features (MVP set that works with SGCC's daily kWh only)
# ----------------------------------------------------------------------------
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


def _autocorr(x, lag):
    a, b = x[:, :-lag], x[:, lag:]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + EPS)


def build_features(clean, raw, dow=None):
    n, T = clean.shape
    x = clean.astype(np.float64)
    f = {}

    # A. basic stats
    last90, last365 = x[:, -90:], x[:, -365:]
    f["mean_90"] = last90.mean(1)
    f["std_90"] = last90.std(1)
    f["cv_90"] = f["std_90"] / (f["mean_90"] + EPS)
    f["cv_365"] = last365.std(1) / (last365.mean(1) + EPS)
    f["skew_365"] = skew(last365, axis=1)
    f["kurt_365"] = kurtosis(last365, axis=1)
    f["iqr_over_median_365"] = (np.percentile(last365, 75, axis=1) - np.percentile(last365, 25, axis=1)) / (
        np.median(last365, axis=1) + EPS)

    # B. change / trend
    m30 = x[:, -30:].mean(1)
    prev90 = x[:, -120:-30].mean(1)
    old = x[:, -365:-90].mean(1)
    f["drop_ratio_30_90"] = np.clip(m30 / (prev90 + EPS), 0, 10)
    f["drop_ratio_90_365"] = np.clip(f["mean_90"] / (old + EPS), 0, 10)
    f["last_vs_first_year"] = np.clip(last365.mean(1) / (x[:, :365].mean(1) + EPS), 0, 10)
    nm = T // 30
    blocks = x[:, T - nm * 30:].reshape(n, nm, 30).mean(2)
    pct = (blocks[:, 1:] - blocks[:, :-1]) / np.maximum(blocks[:, :-1], EPS)
    f["max_mom_drop"] = np.clip(-pct.min(1), 0, 1)
    y180 = x[:, -180:]
    xs = np.arange(180) - 89.5
    f["trend_slope_180"] = np.clip((y180 * xs).sum(1) / (xs ** 2).sum() * 180 / (y180.mean(1) + EPS), -5, 5)

    # C. zeros / missing / flat line (computed from RAW so gaps are not confused with zeros)
    zero_mask = raw == 0
    f["zero_day_ratio"] = zero_mask.mean(1)
    f["longest_zero_streak"], f["n_zero_streaks"] = _runs(zero_mask)
    f["missing_ratio"] = np.isnan(raw).mean(1)
    f["flat_line_ratio"] = ((np.diff(x, axis=1) == 0) & (x[:, 1:] > 0)).mean(1)

    # D. time-series shape
    f["autocorr_lag1"] = _autocorr(last365, 1)
    f["autocorr_lag7"] = _autocorr(last365, 7)
    f["diff_std_norm"] = np.diff(x, axis=1).std(1) / (x.mean(1) + EPS)

    # E-lite. weekday / weekend (only if dates could be parsed)
    if dow is not None:
        wk = (dow[-365:] >= 5)
        f["weekend_weekday_ratio"] = np.clip(last365[:, wk].mean(1) / (last365[:, ~wk].mean(1) + EPS), 0, 10)

    feats = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan).fillna(0)
    return feats


def minmax_rows(clean):
    mn, mx = clean.min(1, keepdims=True), clean.max(1, keepdims=True)
    return np.ascontiguousarray((clean - mn) / (mx - mn + EPS), dtype=np.float32)


# ----------------------------------------------------------------------------
# 3. Models
# ----------------------------------------------------------------------------
def make_logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0, random_state=SEED))


def make_gbm(pos_weight, early_stopping=True):
    try:
        from xgboost import XGBClassifier
        kw = dict(n_estimators=1000 if early_stopping else 300, learning_rate=0.05, max_depth=5, subsample=0.8,
                  colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr", n_jobs=-1,
                  random_state=SEED)
        if early_stopping:
            kw["early_stopping_rounds"] = 50
        return XGBClassifier(**kw), "XGBoost"
    except ImportError:
        print("[warn] xgboost not installed -> using sklearn HistGradientBoosting as the 'XGBoost' slot")
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED,
                                              early_stopping=early_stopping), "XGBoost"


def train_autoencoder(Xtr, Xva, epochs=40, batch=256, lr=1e-3, patience=6):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Xtr.shape[1]
    net = nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
        nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, d), nn.Sigmoid()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr)), batch_size=batch, shuffle=True)
    Xv = torch.from_numpy(Xva).to(dev)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xv), Xv).item()
        hist.append((tot / len(Xtr), vl))
        print(f"  AE epoch {ep + 1:02d}  train={hist[-1][0]:.5f}  val(normal)={vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def score(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.from_numpy(X[i:i + 2048]).to(dev)
                out.append(((net(xb) - xb) ** 2).mean(1).cpu().numpy())
        return np.concatenate(out)

    return net, score, hist


# ----------------------------------------------------------------------------
# 4. Evaluation helpers
# ----------------------------------------------------------------------------
def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


def prec_rec_at_k(y, s, k):
    k = min(k, len(y))
    top = np.argsort(-s)[:k]
    tp = y[top].sum()
    return tp / k, tp / max(y.sum(), 1)


def evaluate(name, y, s, thr, ks):
    pred = (s >= thr).astype(int)
    row = {"Model": name,
           "Precision": precision_score(y, pred, zero_division=0),
           "Recall": recall_score(y, pred, zero_division=0),
           "F1": f1_score(y, pred, zero_division=0),
           "PR-AUC": average_precision_score(y, s),
           "ROC-AUC": roc_auc_score(y, s)}
    for k in ks:
        p, r = prec_rec_at_k(y, s, k)
        row[f"Precision@{k}"] = p
        row[f"Recall@{k}"] = r
    return row


# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
def plot_curves(y, scores, out):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=COLORS.get(n), lw=2, label=f"{n} (AP={average_precision_score(y, s):.3f})")
    ax.axhline(y.mean(), ls="--", c="gray", label=f"Random (prevalence={y.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve (test set)", ylim=(0, 1.02))
    ax.legend(loc="upper right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=COLORS.get(n), lw=2, label=f"{n} (AUC={roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="gray", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=200)
    plt.close(fig)


def plot_confusions(y, scores, thrs, out):
    fig, axes = plt.subplots(1, len(scores), figsize=(4.6 * len(scores), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (n, s) in zip(axes, scores.items()):
        cm = confusion_matrix(y, (s >= thrs[n]).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Theft"], yticklabels=["Normal", "Theft"],
               xlabel="Predicted", ylabel="Actual", title=n)
    fig.suptitle("Confusion Matrix (test set, threshold = best F1 on validation)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def plot_metric_bars(table, out):
    cols = ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
    x = np.arange(len(cols))
    w = 0.8 / len(table)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (_, r) in enumerate(table.iterrows()):
        bars = ax.bar(x + i * w - 0.4 + w / 2, [r[c] for c in cols], w, label=r["Model"],
                      color=COLORS.get(r["Model"]))
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .01, f"{b.get_height():.2f}",
                    ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=cols, ylim=(0, 1.1), title="Model comparison (test set)")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "metrics_bar.png", dpi=200)
    plt.close(fig)


def plot_precision_at_k(y, scores, out):
    kmax = min(1000, len(y))
    ks = np.unique(np.linspace(20, kmax, 60).astype(int))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, s in scores.items():
        order = np.argsort(-s)
        cum = np.cumsum(y[order])
        ax.plot(ks, cum[ks - 1] / ks, color=COLORS.get(n), lw=2, label=n)
    ax.axhline(y.mean(), ls="--", c="gray", label="Random inspection")
    ax.set(xlabel="K = number of customers inspected (top-K by risk)", ylabel="Precision@K",
           title="Precision@K: how many inspections find a thief", ylim=(0, 1.02))
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "precision_at_k.png", dpi=200)
    plt.close(fig)


def save_table_png(table, out):
    show = table[["Model", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"] +
                 [c for c in table.columns if c.startswith("Precision@")]].copy()
    cells = [[r["Model"]] + [f"{r[c]:.3f}" for c in show.columns[1:]] for _, r in show.iterrows()]
    fig, ax = plt.subplots(figsize=(1.45 * len(show.columns) + 1, 0.7 * len(show) + 1.3))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=show.columns, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 1.8)
    for j in range(len(show.columns)):
        tb[(0, j)].set_facecolor("#37474f")
        tb[(0, j)].set_text_props(color="white", weight="bold")
        if j > 0:
            i_best = int(show.iloc[:, j].astype(float).values.argmax())
            tb[(i_best + 1, j)].set_facecolor("#c8e6c9")
            tb[(i_best + 1, j)].set_text_props(weight="bold")
    ax.set_title("Test-set results (green = best per metric)", pad=12)
    fig.tight_layout()
    fig.savefig(out / "metrics_table.png", dpi=200)
    plt.close(fig)


def table_to_markdown(table):
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join([r["Model"]] + [f"{r[c]:.3f}" for c in cols[1:]]) + " |")
    return "\n".join(lines)


def plot_examples(clean_te, ids_te, y_te, s, thr, name, out):
    order = np.argsort(-s)
    tp = [i for i in order if y_te[i] == 1 and s[i] >= thr][:2]
    fp = [i for i in order if y_te[i] == 0 and s[i] >= thr][:2]
    fn = [i for i in order[::-1] if y_te[i] == 1 and s[i] < thr][:2]
    groups = [("True Positive (caught)", tp), ("False Positive (false alarm)", fp), ("False Negative (missed)", fn)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for r, (title, idxs) in enumerate(groups):
        for c in range(2):
            ax = axes[r, c]
            if c >= len(idxs):
                ax.axis("off")
                continue
            i = idxs[c]
            ts = clean_te[i]
            ax.plot(ts, lw=.6, alpha=.5, color="gray")
            ax.plot(np.arange(29, len(ts)), np.convolve(ts, np.ones(30) / 30, mode="valid"), lw=1.8,
                    color=COLORS.get(name, "C0"))
            ax.set_title(f"{title} | id={ids_te[i][:10]} | score={s[i]:.3g}", fontsize=9)
            ax.set_ylabel("kWh/day")
    for ax in axes[-1]:
        ax.set_xlabel("day index")
    fig.suptitle(f"Example customers - {name} (gray = daily, color = 30-day mean)")
    fig.tight_layout()
    fig.savefig(out / "examples_best_model.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to SGCC data.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--third-model", choices=["autoencoder", "isoforest"], default="autoencoder")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cv", action="store_true", help="run 5-fold CV on train+val for the supervised models")
    ap.add_argument("--quick", action="store_true", help="subsample + few epochs for a smoke test")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("== Loading data ==")
    ids, y, raw, dow = load_sgcc(args.data)
    if args.quick:
        keep, _ = train_test_split(np.arange(len(y)), train_size=min(6000, len(y)), stratify=y, random_state=SEED)
        ids, y, raw = ids[keep], y[keep], raw[keep]
        args.epochs = min(args.epochs, 8)
    print(f"customers={len(y):,}  days={raw.shape[1]}  theft={y.sum():,} ({y.mean():.2%})")

    print("== Cleaning + feature engineering ==")
    clean = clean_series(raw)
    feats = build_features(clean, raw, dow)
    feats.insert(0, "CONS_NO", ids)
    feats["FLAG"] = y
    feats.to_csv(out / "features.csv", index=False)
    X = feats.drop(columns=["CONS_NO", "FLAG"])
    X_ae = minmax_rows(clean)
    print(f"features ({X.shape[1]}): {list(X.columns)}")

    print("== Split 70/15/15 (stratified) ==")
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
    tr, va = train_test_split(tv, test_size=0.15 / 0.85, stratify=y[tv], random_state=SEED)
    print(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}  (theft rate test={y[te].mean():.2%})")

    pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    val_scores, test_scores = {}, {}

    # --- Model 1: Logistic Regression
    print("== Model 1: Logistic Regression ==")
    lr = make_logreg().fit(X.iloc[tr], y[tr])
    val_scores["Logistic Regression"] = lr.predict_proba(X.iloc[va])[:, 1]
    test_scores["Logistic Regression"] = lr.predict_proba(X.iloc[te])[:, 1]
    joblib.dump(lr, out / "models" / "logreg.joblib")

    # --- Model 2: XGBoost
    print("== Model 2: XGBoost ==")
    gbm, gbm_name = make_gbm(pos_w, early_stopping=True)
    if hasattr(gbm, "get_booster"):  # real XGBoost (supports eval_set / early stopping)
        gbm.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        print(f"  best iteration: {getattr(gbm, 'best_iteration', 'n/a')}")
    else:
        gbm.fit(X.iloc[tr], y[tr])
    val_scores["XGBoost"] = gbm.predict_proba(X.iloc[va])[:, 1]
    test_scores["XGBoost"] = gbm.predict_proba(X.iloc[te])[:, 1]
    joblib.dump(gbm, out / "models" / "xgboost.joblib")

    # --- Model 3: Autoencoder (or Isolation Forest)
    third = args.third_model
    if third == "autoencoder":
        try:
            import torch  # noqa: F401
        except ImportError:
            print("[warn] PyTorch not installed -> using Isolation Forest as model 3")
            third = "isoforest"
    if third == "autoencoder":
        print("== Model 3: Autoencoder (trained on normal customers only) ==")
        tr_n = tr[y[tr] == 0]
        va_n = va[y[va] == 0]
        net, ae_score, hist = train_autoencoder(X_ae[tr_n], X_ae[va_n], epochs=args.epochs)
        name3 = "Autoencoder"
        val_scores[name3] = ae_score(X_ae[va])
        test_scores[name3] = ae_score(X_ae[te])
        import torch
        torch.save(net.state_dict(), out / "models" / "autoencoder.pt")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([h[0] for h in hist], label="train (normal)")
        ax.plot([h[1] for h in hist], label="validation (normal)")
        ax.set(xlabel="epoch", ylabel="MSE", title="Autoencoder training loss")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "autoencoder_loss.png", dpi=200)
        plt.close(fig)
    else:
        print("== Model 3: Isolation Forest (fit on normal customers' features) ==")
        name3 = "Isolation Forest"
        iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1)
        iso.fit(StandardScaler().fit(X.iloc[tr[y[tr] == 0]]).transform(X.iloc[tr[y[tr] == 0]]))
        sc = StandardScaler().fit(X.iloc[tr[y[tr] == 0]])
        val_scores[name3] = -iso.score_samples(sc.transform(X.iloc[va]))
        test_scores[name3] = -iso.score_samples(sc.transform(X.iloc[te]))
        joblib.dump((sc, iso), out / "models" / "isoforest.joblib")

    # --- Thresholds from VALIDATION, final metrics on TEST
    print("== Evaluating ==")
    yv, yt = y[va], y[te]
    thrs = {n: best_f1_threshold(yv, s) for n, s in val_scores.items()}
    ks = [k for k in (100, 250, 500) if k <= len(yt)]
    table = pd.DataFrame([evaluate(n, yt, s, thrs[n], ks) for n, s in test_scores.items()])
    val_ap = {n: average_precision_score(yv, s) for n, s in val_scores.items()}
    best_name = max(val_ap, key=val_ap.get)  # choose "best" on validation, not on test

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(f"\nBest model by validation PR-AUC: {best_name}  {val_ap}")
    table.to_csv(out / "metrics_table.csv", index=False)
    (out / "metrics_table.md").write_text(table_to_markdown(table), encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps(
        {"n_customers": int(len(y)), "theft_rate": float(y.mean()), "n_train": int(len(tr)), "n_val": int(len(va)),
         "n_test": int(len(te)), "thresholds": thrs, "val_pr_auc": val_ap, "best_model_by_val": best_name,
         "features": list(X.columns)}, indent=2))

    # --- Optional 5-fold CV (supervised models), on train+val only
    if args.cv:
        print("== 5-fold stratified CV on train+val (PR-AUC) ==")
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        rows = []
        for n, mk in (("Logistic Regression", make_logreg), ("XGBoost", lambda: make_gbm(pos_w, False)[0])):
            sc = cross_val_score(mk(), X.iloc[tv], y[tv], cv=skf, scoring="average_precision")
            rows.append({"Model": n, "CV PR-AUC mean": sc.mean(), "CV PR-AUC std": sc.std()})
            print(f"  {n}: {sc.mean():.3f} +/- {sc.std():.3f}")
        pd.DataFrame(rows).to_csv(out / "cv_results.csv", index=False)

    # --- Plots
    print("== Plotting ==")
    plot_curves(yt, test_scores, out)
    plot_confusions(yt, test_scores, thrs, out)
    plot_metric_bars(table, out)
    plot_precision_at_k(yt, test_scores, out)
    save_table_png(table, out)
    plot_examples(clean[te], ids[te], yt, test_scores[best_name], thrs[best_name], best_name, out)

    if hasattr(gbm, "feature_importances_"):
        imp = pd.Series(gbm.feature_importances_, index=X.columns).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh(imp.index, imp.values, color=COLORS["XGBoost"])
        ax.set(title="XGBoost feature importance (top 15)", xlabel="importance")
        fig.tight_layout()
        fig.savefig(out / "feature_importance_xgb.png", dpi=200)
        plt.close(fig)
    if hasattr(gbm, "get_booster"):
        try:
            import shap
            sample = X.iloc[te].sample(min(2000, len(te)), random_state=SEED)
            sv = shap.TreeExplainer(gbm).shap_values(sample)
            plt.figure()
            shap.summary_plot(sv, sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(out / "shap_summary.png", dpi=200, bbox_inches="tight")
            plt.close("all")
        except Exception as e:  # shap is optional
            print(f"[info] SHAP skipped: {e}")

    # --- Predictions for the dashboard (risk score 0-100 = percentile rank of best model's score)
    pred = pd.DataFrame({"CONS_NO": ids[te], "y_true": yt})
    for n, s in test_scores.items():
        pred[f"score_{n}"] = s
    pred["risk_score_best"] = pd.Series(test_scores[best_name]).rank(pct=True).to_numpy() * 100
    pred.sort_values("risk_score_best", ascending=False).to_csv(out / "test_predictions.csv", index=False)

    print(f"\nDone. Everything is in: {out.resolve()}")


if __name__ == "__main__":
    main()
