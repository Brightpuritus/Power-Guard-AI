#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) for the SGCC Electricity Theft dataset
======================================================================
Works on the RAW data.csv (before cleaning), so it shows the real state of the data.
Everything compares NORMAL (FLAG=0) vs THEFT (FLAG=1) customers.

Run
  python eda.py --data data.csv --out eda_results

Needs: numpy pandas matplotlib scipy   (already installed if train_compare.py runs)

Charts (in --out)
  01_class_distribution.png       how many normal vs theft customers (bar + counts)
  02_missing_values.png           missing ratio per customer + missing by day, by class
  03_zero_days.png                share of days with 0 kWh, by class
  04_consumption_level.png        average kWh per day per customer, by class
  05_monthly_trend.png            monthly median kWh (absolute and relative to own average)
  06_theft_rate_by_quality.png    theft rate inside each missing-ratio / zero-day bucket
  07_sample_customers.png         random example series (normal vs theft)
Tables
  eda_summary.csv                 normal vs theft statistics + Mann-Whitney test
  theft_rate_by_bucket.csv        data behind chart 06
  eda_overview.json               dataset-level numbers
  customer_stats.csv              per-customer stats
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", category=RuntimeWarning)
COL = {0: "#1f77b4", 1: "#d62728"}
NAME = {0: "Normal", 1: "Theft"}


# ----------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path).drop_duplicates(subset="CONS_NO").reset_index(drop=True)
    date_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    parsed = pd.to_datetime(pd.Series(date_cols), errors="coerce")
    dates = None
    if parsed.notna().all():  # sort day columns chronologically
        order = np.argsort(parsed.values)
        date_cols = [date_cols[i] for i in order]
        dates = pd.DatetimeIndex(parsed.iloc[order].values)
    raw = df[date_cols].to_numpy(dtype=np.float32, copy=True)
    raw[raw < 0] = np.nan                       # negative kWh is impossible -> missing
    return df["CONS_NO"].astype(str).to_numpy(), df["FLAG"].astype(int).to_numpy(), raw, dates


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print("saved", name)


def rate_by_bucket(ax, values, y, bins, labels, title, xlabel):
    cat = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    g = pd.DataFrame({"bucket": cat, "y": y}).groupby("bucket", observed=False)["y"].agg(["mean", "count"])
    bars = ax.bar(labels, g["mean"] * 100, color="#4c78a8", edgecolor="black", alpha=.85)
    top = max((g["mean"] * 100).max(), y.mean() * 100)
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * .02, f"{r['mean'] * 100:.1f}%\n(n={int(r['count']):,})",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(y.mean() * 100, ls="--", c="gray", label=f"overall {y.mean() * 100:.2f}%")
    ax.set(title=title, xlabel=xlabel, ylabel="theft rate (%)", ylim=(0, top * 1.3))
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    return g.reset_index().assign(chart=title)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eda_results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("== Loading ==")
    ids, y, raw, dates = load(args.data)
    n, T = raw.shape
    na = np.isnan(raw)
    print(f"customers={n:,}  days={T:,}  theft={y.sum():,} ({y.mean():.2%})")

    # ---- per-customer statistics (computed only on days that have data)
    q = pd.DataFrame({
        "CONS_NO": ids, "FLAG": y,
        "missing_ratio": na.mean(1),
        "zero_day_ratio": (raw == 0).mean(1),
        "mean_kwh": np.nanmean(raw, 1),
        "std_kwh": np.nanstd(raw, 1),
    })
    q["cv"] = np.where(q["mean_kwh"] > 0, q["std_kwh"] / q["mean_kwh"], np.nan)
    q.to_csv(out / "customer_stats.csv", index=False)

    # ---- overview numbers
    overview = {
        "customers": int(n), "days": int(T),
        "date_range": [str(dates.min().date()), str(dates.max().date())] if dates is not None else None,
        "normal": int((y == 0).sum()), "theft": int((y == 1).sum()), "theft_rate": float(y.mean()),
        "total_cells": int(n * T), "missing_pct": float(na.mean() * 100),
        "zero_cells_pct": float((raw == 0).mean() * 100),
        "customers_all_missing": int(na.all(1).sum()),
        "customers_all_zero": int((((raw == 0) | na).all(1) & ~na.all(1)).sum()),
        "customers_with_any_missing": int(na.any(1).sum()),
    }
    (out / "eda_overview.json").write_text(json.dumps(overview, indent=2))

    # ---- summary table + Mann-Whitney test
    rows = []
    for m, label in [("missing_ratio", "missing ratio"), ("zero_day_ratio", "zero-day ratio"),
                     ("mean_kwh", "mean kWh/day"), ("std_kwh", "std kWh/day"), ("cv", "coefficient of variation")]:
        a, b = q.loc[q.FLAG == 0, m].dropna(), q.loc[q.FLAG == 1, m].dropna()
        U, p = mannwhitneyu(b, a)
        rows.append({"metric": label, "normal_mean": a.mean(), "normal_median": a.median(),
                     "theft_mean": b.mean(), "theft_median": b.median(),
                     "P(theft > normal)": U / (len(a) * len(b)), "p_value": p})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "eda_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nNormal vs theft (P(theft > normal): 0.5 = no difference):")
    print(summary.round(4).to_string(index=False))

    # ---------------------------------------------------------------- 01 class distribution
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cnt = np.bincount(y, minlength=2)
    bars = ax.bar([NAME[0], NAME[1]], cnt, color=[COL[0], COL[1]], edgecolor="black", alpha=.85)
    for b, v in zip(bars, cnt):
        ax.text(b.get_x() + b.get_width() / 2, v + cnt.max() * .01, f"{v:,}\n({v / cnt.sum():.2%})",
                ha="center", va="bottom", fontweight="bold")
    ax.set(title="Distribution of Customers by Class (SGCC)", ylabel="Number of customers", ylim=(0, cnt.max() * 1.15))
    ax.grid(axis="y", alpha=.3)
    savefig(fig, out, "01_class_distribution.png")

    # ---------------------------------------------------------------- 02 missing values
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in (0, 1):
        axes[0].hist(q.loc[q.FLAG == c, "missing_ratio"] * 100, bins=40, density=True, histtype="step", lw=2,
                     color=COL[c], label=f"{NAME[c]} (mean {q.loc[q.FLAG == c, 'missing_ratio'].mean() * 100:.1f}%)")
        x = dates if dates is not None else np.arange(T)
        axes[1].plot(x, na[y == c].mean(0) * 100, lw=1, color=COL[c], label=NAME[c])
    axes[0].set(title="Missing days per customer", xlabel="missing days (%)", ylabel="density")
    axes[1].set(title="Missing values by day", ylabel="% of customers with no reading")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    savefig(fig, out, "02_missing_values.png")

    # ---------------------------------------------------------------- 03 zero days
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for c in (0, 1):
        v = q.loc[q.FLAG == c, "zero_day_ratio"] * 100
        ax.hist(v, bins=40, density=True, histtype="step", lw=2, color=COL[c],
                label=f"{NAME[c]} (mean {v.mean():.1f}%, median {v.median():.1f}%)")
    ax.set(title="Share of days with 0 kWh", xlabel="zero-kWh days (%)", ylabel="density", yscale="log")
    ax.legend()
    ax.grid(alpha=.3)
    savefig(fig, out, "03_zero_days.png")

    # ---------------------------------------------------------------- 04 consumption level
    pos = q[q.mean_kwh > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in (0, 1):
        v = np.log10(pos.loc[pos.FLAG == c, "mean_kwh"])
        axes[0].hist(v, bins=60, density=True, histtype="step", lw=2, color=COL[c],
                     label=f"{NAME[c]} (median {10 ** v.median():.2f} kWh/day)")
    axes[0].set(title="Average consumption per customer", xlabel="log10(mean kWh per day)", ylabel="density")
    axes[0].legend()
    axes[0].grid(alpha=.3)
    bp = axes[1].boxplot([pos.loc[pos.FLAG == c, "mean_kwh"] for c in (0, 1)], showfliers=False, patch_artist=True)
    axes[1].set_xticklabels([NAME[0], NAME[1]])
    for patch, c in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(COL[c])
        patch.set_alpha(.6)
    axes[1].set(title="Mean kWh per day (outliers hidden)", ylabel="kWh/day")
    axes[1].grid(axis="y", alpha=.3)
    savefig(fig, out, "04_consumption_level.png")

    # ---------------------------------------------------------------- 05 monthly trend
    if dates is not None:
        mkeys = dates.strftime("%Y-%m").to_numpy()
        uniq = pd.unique(mkeys)
        xm = pd.to_datetime([u + "-01" for u in uniq])
    else:
        mkeys = np.repeat(np.arange(T // 30 + 1), 30)[:T]
        uniq = pd.unique(mkeys)
        xm = uniq
    monthly = np.column_stack([np.nanmean(raw[:, mkeys == u], axis=1) for u in uniq])
    rel = np.where((q.mean_kwh.to_numpy() > 0)[:, None], monthly / q.mean_kwh.to_numpy()[:, None], np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in (0, 1):
        axes[0].plot(xm, np.nanmedian(monthly[y == c], axis=0), lw=2, color=COL[c], label=NAME[c])
        axes[1].plot(xm, np.nanmedian(rel[y == c], axis=0), lw=2, color=COL[c], label=NAME[c])
    axes[0].set(title="Median consumption per month", ylabel="kWh/day")
    axes[1].set(title="Relative to each customer's own average (1.0 = usual level)", ylabel="relative consumption")
    axes[1].axhline(1, c="gray", ls="--")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    savefig(fig, out, "05_monthly_trend.png")

    # ---------------------------------------------------------------- 06 theft rate by data quality
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    t1 = rate_by_bucket(axes[0], q.missing_ratio.to_numpy(), y, [0, .1, .25, .5, .75, 1.0],
                        ["0-10%", "10-25%", "25-50%", "50-75%", ">75%"], "Theft rate by missing ratio",
                        "missing days per customer")
    t2 = rate_by_bucket(axes[1], q.zero_day_ratio.to_numpy(), y, [0, .01, .1, .25, .5, 1.0],
                        ["0-1%", "1-10%", "10-25%", "25-50%", ">50%"], "Theft rate by zero-day ratio",
                        "days with 0 kWh per customer")
    pd.concat([t1, t2]).to_csv(out / "theft_rate_by_bucket.csv", index=False)
    savefig(fig, out, "06_theft_rate_by_quality.png")

    # ---------------------------------------------------------------- 07 sample customers
    ok = (q.missing_ratio < 0.3) & (q.mean_kwh > 0)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    x = dates if dates is not None else np.arange(T)
    for j, c in enumerate((0, 1)):
        pool = np.where(ok & (q.FLAG == c))[0]
        pick = rng.choice(pool, size=min(3, len(pool)), replace=False)
        for i in range(3):
            ax = axes[i, j]
            if i >= len(pick):
                ax.axis("off")
                continue
            k = pick[i]
            ax.plot(x, raw[k], lw=.5, color="gray", alpha=.6)
            ax.plot(x, pd.Series(raw[k]).rolling(30, min_periods=10).mean(), lw=1.8, color=COL[c])
            ax.set_title(f"{NAME[c]} | id={ids[k][:10]} | missing {q.missing_ratio[k]:.0%}", fontsize=9)
            ax.set_ylabel("kWh/day")
    fig.suptitle("Random example customers (gray = daily, color = 30-day mean)")
    savefig(fig, out, "07_sample_customers.png")

    print(f"\nDone. Everything is in: {out.resolve()}")


if __name__ == "__main__":
    main()
