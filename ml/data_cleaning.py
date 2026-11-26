#!/usr/bin/env python3
"""
Data Cleaning for the SGCC Electricity Theft dataset
====================================================
Audits the raw file, cleans it step by step, and records how many values each
step touched, so the numbers can go straight into the slides / Chapter 3-4.

Cleaning rules (same rules as train_compare.py, so results stay consistent)
  1. Validate + de-duplicate customers (CONS_NO), check FLAG
  2. Parse + sort day columns chronologically, check for calendar gaps
  3. Negative kWh            -> treated as missing
  4. Missing values          -> linear interpolation of gaps (max --max-gap days)
  5. Still missing           -> 0
  6. Outliers                -> clipped at mean + 3*std of each customer
  Zeros that were really in the raw data are KEPT (they can be a theft signal).

Run
  python data_cleaning.py --data data.csv --out cleaned
  python data_cleaning.py --data data.csv --out cleaned --strict-gaps
  python data_cleaning.py --data data.csv --out cleaned --drop-bad --max-missing 0.5

Install: pip install numpy pandas matplotlib   (pyarrow optional, for parquet output)

Outputs (in --out)
  cleaning_steps.csv / cleaning_report.json / cleaning_report.md   what each step changed
  customer_quality.csv       per-customer audit (missing, zeros, negatives, clipped...)
  eda_by_class.csv           normal vs theft comparison (fills the EDA slide)
  class_balance.png, missing_distribution.png, missing_by_day.png,
  cleaning_example.png, before_after_mean.png
  cleaned_data.parquet (or .csv)   cleaned customers x days table
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

warnings.filterwarnings("ignore", category=RuntimeWarning)  # e.g. nanmean of an all-missing customer


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def gap_lengths(na):
    """For each missing cell: length of the run of consecutive missing days it belongs to (0 if not missing)."""
    n, T = na.shape
    fwd = np.zeros((n, T), dtype=np.int16)
    bwd = np.zeros((n, T), dtype=np.int16)
    for t in range(T):
        prev = fwd[:, t - 1] if t else 0
        fwd[:, t] = np.where(na[:, t], prev + 1, 0)
    for t in range(T - 1, -1, -1):
        nxt = bwd[:, t + 1] if t < T - 1 else 0
        bwd[:, t] = np.where(na[:, t], nxt + 1, 0)
    return np.where(na, fwd.astype(np.int32) + bwd - 1, 0)


def pct(a, b):
    return float(a) / float(b) * 100 if b else 0.0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to SGCC data.csv")
    ap.add_argument("--out", default="data/processed/cleaned")
    ap.add_argument("--max-gap", type=int, default=7, help="longest gap (days) to interpolate")
    ap.add_argument("--strict-gaps", action="store_true",
                    help="gaps longer than --max-gap are NOT interpolated at all (default, like train_compare.py: "
                         "pandas fills the first --max-gap days of a longer gap)")
    ap.add_argument("--sigma", type=float, default=3.0, help="outlier cap = mean + sigma*std per customer")
    ap.add_argument("--drop-bad", action="store_true", help="drop customers that are all-zero or too incomplete")
    ap.add_argument("--max-missing", type=float, default=0.5, help="with --drop-bad: max allowed missing ratio")
    ap.add_argument("--skip-save", action="store_true", help="do not write the cleaned table")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = []

    def log(step, rule, cells=None, customers=None, total_cells=None):
        steps.append({"Step": step, "Rule": rule,
                      "Cells affected": cells,
                      "% of all cells": round(pct(cells, total_cells), 3) if cells is not None and total_cells else None,
                      "Customers affected": customers})

    # ------------------------------------------------------------------ 0. load
    print("== 0. Load ==")
    df = pd.read_csv(args.data)
    n_rows_raw = len(df)
    date_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    print(f"raw file: {n_rows_raw:,} rows x {len(date_cols):,} day columns")

    block = df[date_cols]
    n_nonnumeric = 0
    if not all(pd.api.types.is_numeric_dtype(t) for t in block.dtypes):
        coerced = block.apply(pd.to_numeric, errors="coerce")
        n_nonnumeric = int((coerced.isna() & block.notna()).to_numpy().sum())
        df[date_cols] = coerced
    log("0 Load", f"{n_rows_raw:,} rows, {len(date_cols):,} day columns, {n_nonnumeric:,} non-numeric cells -> missing")

    # ------------------------------------------------------------------ 1. IDs + labels
    print("== 1. Customer IDs and labels ==")
    dup_mask = df["CONS_NO"].duplicated(keep=False)
    n_dup_rows = int(df["CONS_NO"].duplicated(keep="first").sum())
    conflict = int((df[dup_mask].groupby("CONS_NO")["FLAG"].nunique() > 1).sum()) if dup_mask.any() else 0
    df = df.drop_duplicates(subset="CONS_NO", keep="first")
    bad_flag = ~df["FLAG"].isin([0, 1])
    n_bad_flag = int(bad_flag.sum())
    df = df[~bad_flag].reset_index(drop=True)
    log("1 IDs + labels",
        f"drop duplicate CONS_NO ({n_dup_rows:,} rows, {conflict:,} ids with conflicting FLAG); "
        f"drop invalid FLAG ({n_bad_flag:,} rows)", customers=n_dup_rows + n_bad_flag)
    print(f"duplicate rows removed: {n_dup_rows:,} (conflicting flags: {conflict}); invalid FLAG rows: {n_bad_flag}")

    ids = df["CONS_NO"].astype(str).to_numpy()
    y = df["FLAG"].astype(int).to_numpy()
    n, T = len(df), len(date_cols)
    total_cells = n * T

    # ------------------------------------------------------------------ 2. dates
    print("== 2. Day columns ==")
    parsed = pd.to_datetime(pd.Series(date_cols), errors="coerce")
    n_unparsed = int(parsed.isna().sum())
    was_sorted, calendar_gaps = None, None
    if n_unparsed == 0:
        order = np.argsort(parsed.values)
        was_sorted = bool(parsed.is_monotonic_increasing)
        date_cols = [date_cols[i] for i in order]
        dates = pd.DatetimeIndex(parsed.iloc[order].values)
        calendar_gaps = int((dates.max() - dates.min()).days + 1 - len(dates))
        rule = (f"parse dates, sort chronologically (was sorted: {was_sorted}); range "
                f"{dates.min().date()} to {dates.max().date()}; missing calendar days: {calendar_gaps}")
    else:
        dates = pd.RangeIndex(T)
        rule = f"{n_unparsed} column names are not dates -> original order kept"
    log("2 Day columns", rule)
    print(rule)

    raw = df[date_cols].to_numpy(dtype=np.float32)
    del df

    # ------------------------------------------------------------------ 3. negatives
    print("== 3. Negative values ==")
    neg = raw < 0
    na0 = np.isnan(raw)                      # missing in the file itself
    arr = np.where(neg, np.nan, raw)
    na1 = np.isnan(arr)                      # missing + negatives
    log("3 Negative kWh", "negative values are impossible -> set to missing",
        cells=int(neg.sum()), customers=int(neg.any(1).sum()), total_cells=total_cells)
    print(f"negative cells: {int(neg.sum()):,} in {int(neg.any(1).sum()):,} customers")

    # ------------------------------------------------------------------ 4. missing audit
    print("== 4. Missing values ==")
    gl = gap_lengths(na1)
    n_missing = int(na1.sum())
    in_short = int(((gl > 0) & (gl <= args.max_gap)).sum())
    miss_ratio = na1.mean(1)
    log("4a Missing (audit)", f"{pct(n_missing, total_cells):.2f}% of all cells; "
        f"{pct(in_short, n_missing):.1f}% of them are in gaps <= {args.max_gap} days",
        cells=n_missing, customers=int(na1.any(1).sum()), total_cells=total_cells)
    print(f"missing cells: {n_missing:,} ({pct(n_missing, total_cells):.2f}%); in gaps <= {args.max_gap} days: "
          f"{pct(in_short, n_missing):.1f}%; customers with any missing: {int(na1.any(1).sum()):,}")

    # ------------------------------------------------------------------ 5. interpolate / fill 0
    print("== 5. Fill missing ==")
    filled = pd.DataFrame(arr).T.interpolate(limit=args.max_gap, limit_direction="both").T.to_numpy(dtype=np.float32, copy=True)
    if args.strict_gaps:
        filled[gl > args.max_gap] = np.nan
    interp_mask = na1 & ~np.isnan(filled)
    zerofill_mask = np.isnan(filled)
    filled = np.nan_to_num(filled, nan=0.0)
    mode = "strict (long gaps untouched)" if args.strict_gaps else "pandas limit (first days of a long gap are filled)"
    log("5a Interpolate", f"linear interpolation, max {args.max_gap} consecutive days per gap, mode: {mode}",
        cells=int(interp_mask.sum()), customers=int(interp_mask.any(1).sum()), total_cells=total_cells)
    log("5b Fill with 0", "cells that are still missing are set to 0",
        cells=int(zerofill_mask.sum()), customers=int(zerofill_mask.any(1).sum()), total_cells=total_cells)
    print(f"interpolated cells: {int(interp_mask.sum()):,}; filled with 0: {int(zerofill_mask.sum()):,}")

    # ------------------------------------------------------------------ 6. outliers
    print("== 6. Outliers ==")
    mu = filled.mean(1, keepdims=True)
    sd = filled.std(1, keepdims=True)
    cap = mu + args.sigma * sd
    clip_mask = filled > cap
    cleaned = np.minimum(filled, cap).astype(np.float32)
    energy_removed = pct(float((filled - cleaned).sum()), float(filled.sum()))
    log("6 Outliers", f"clip at mean + {args.sigma:g}*std per customer; removes {energy_removed:.2f}% of total kWh",
        cells=int(clip_mask.sum()), customers=int(clip_mask.any(1).sum()), total_cells=total_cells)
    print(f"clipped cells: {int(clip_mask.sum()):,} in {int(clip_mask.any(1).sum()):,} customers "
          f"({energy_removed:.2f}% of total kWh)")

    # ------------------------------------------------------------------ 7. zeros / bad customers
    print("== 7. Zeros and unusable customers ==")
    zero_raw = raw == 0                                   # real zeros in the file (NaN == 0 is False)
    zero_ratio = zero_raw.mean(1)
    all_missing = na1.all(1)
    all_zero = (cleaned == 0).all(1)
    constant = cleaned.std(1) == 0
    too_incomplete = miss_ratio > args.max_missing
    bad = all_missing | all_zero | too_incomplete
    log("7a Real zeros (kept)", "zero kWh days in the raw data are kept: they can indicate meter tampering",
        cells=int(zero_raw.sum()), customers=int(zero_raw.any(1).sum()), total_cells=total_cells)
    log("7b Unusable customers (audit)",
        f"all missing: {int(all_missing.sum())}, all zero: {int(all_zero.sum())}, constant: {int(constant.sum())}, "
        f"missing > {args.max_missing:.0%}: {int(too_incomplete.sum())}  (theft rate among these: "
        f"{pct(y[bad].sum(), bad.sum()):.1f}%)", customers=int(bad.sum()))
    print(f"all-zero: {int(all_zero.sum())}, all-missing: {int(all_missing.sum())}, constant: {int(constant.sum())}, "
          f"missing > {args.max_missing:.0%}: {int(too_incomplete.sum())}; "
          f"theft rate among 'bad': {pct(y[bad].sum(), bad.sum()):.1f}% (overall {y.mean() * 100:.2f}%)")

    # ------------------------------------------------------------------ per-customer table
    quality = pd.DataFrame({
        "CONS_NO": ids, "FLAG": y,
        "missing_ratio": miss_ratio, "n_negative": neg.sum(1), "zero_day_ratio": zero_ratio,
        "n_interpolated": interp_mask.sum(1), "n_zero_filled": zerofill_mask.sum(1), "n_clipped": clip_mask.sum(1),
        "mean_kwh_raw": np.nanmean(np.where(na1, np.nan, raw), axis=1),
        "mean_kwh_clean": cleaned.mean(1), "std_kwh_clean": cleaned.std(1),
        "all_zero": all_zero, "all_missing": all_missing, "flag_unusable": bad,
    })
    quality.to_csv(out / "customer_quality.csv", index=False)

    by_class = quality.groupby("FLAG").agg(
        customers=("CONS_NO", "count"), missing_ratio=("missing_ratio", "mean"),
        zero_day_ratio=("zero_day_ratio", "mean"), mean_kwh_per_day=("mean_kwh_clean", "mean"),
        median_kwh_per_day=("mean_kwh_clean", "median"), clipped_cells=("n_clipped", "mean"),
        unusable=("flag_unusable", "sum")).rename(index={0: "normal (0)", 1: "theft (1)"})
    by_class.to_csv(out / "eda_by_class.csv")
    print("\nNormal vs theft (use these numbers on the EDA slide):")
    print(by_class.round(4).to_string())

    # ------------------------------------------------------------------ optional drop
    keep = np.ones(n, dtype=bool)
    if args.drop_bad:
        keep = ~bad
        log("8 Drop customers", "--drop-bad: removed unusable customers", customers=int(bad.sum()))
        print(f"\n--drop-bad: removing {int(bad.sum())} customers")
    n_final = int(keep.sum())

    # ------------------------------------------------------------------ report files
    steps_df = pd.DataFrame(steps)
    steps_df.to_csv(out / "cleaning_steps.csv", index=False)
    summary = {
        "customers_raw_rows": n_rows_raw, "customers_final": n_final, "days": T, "total_cells": total_cells,
        "theft_customers": int(y[keep].sum()), "theft_rate": float(y[keep].mean()),
        "missing_cells": n_missing, "missing_pct": pct(n_missing, total_cells),
        "missing_in_short_gaps_pct": pct(in_short, n_missing), "negative_cells": int(neg.sum()),
        "interpolated_cells": int(interp_mask.sum()), "zero_filled_cells": int(zerofill_mask.sum()),
        "clipped_cells": int(clip_mask.sum()), "clipped_energy_pct": energy_removed,
        "real_zero_cells": int(zero_raw.sum()), "mean_missing_ratio": float(miss_ratio.mean()),
        "mean_zero_day_ratio": float(zero_ratio.mean()), "max_gap": args.max_gap, "strict_gaps": args.strict_gaps,
        "sigma": args.sigma, "dropped_customers": int(n - n_final)}
    (out / "cleaning_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# Data cleaning report", "", "| Step | Rule | Cells | % of cells | Customers |", "|---|---|---|---|---|"]
    for r in steps:
        c = "" if r["Cells affected"] is None else f"{r['Cells affected']:,}"
        p = "" if r["% of all cells"] is None else f"{r['% of all cells']}"
        k = "" if r["Customers affected"] is None else f"{r['Customers affected']:,}"
        md.append(f"| {r['Step']} | {r['Rule']} | {c} | {p} | {k} |")
    md += ["", f"Final: {n_final:,} customers x {T:,} days, theft rate {y[keep].mean():.2%}"]
    (out / "cleaning_report.md").write_text("\n".join(md), encoding="utf-8")

    # ------------------------------------------------------------------ plots
    print("\n== Plots ==")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cnt = np.bincount(y, minlength=2)
    ax.bar(["Normal", "Theft"], cnt, color=["#1f77b4", "#d62728"])
    for i, v in enumerate(cnt):
        ax.text(i, v, f"{v:,}\n({v / cnt.sum():.2%})", ha="center", va="bottom")
    ax.set(title="Class balance", ylabel="customers", ylim=(0, cnt.max() * 1.18))
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(miss_ratio * 100, bins=50, color="#1f77b4")
    ax.axvline(miss_ratio.mean() * 100, color="k", ls="--", label=f"mean = {miss_ratio.mean() * 100:.1f}%")
    ax.set(xlabel="missing days per customer (%)", ylabel="customers (log)", yscale="log",
           title="Missing values per customer (raw data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "missing_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, na1.mean(0) * 100, lw=1)
    ax.set(ylabel="% of customers missing", title="Missing values by day (raw data)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "missing_by_day.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(dates, np.nanmean(np.where(na1, np.nan, raw), axis=0), lw=1, label="raw (missing ignored)", alpha=.8)
    ax.plot(dates, cleaned.mean(0), lw=1, label="cleaned")
    ax.set(ylabel="mean kWh per customer per day", title="Average consumption: before vs after cleaning")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "before_after_mean.png", dpi=200)
    plt.close(fig)

    picks = [("most interpolated", int(np.argmax(interp_mask.sum(1)))),
             ("most clipped outliers", int(np.argmax(clip_mask.sum(1))))]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    x = np.arange(T) if isinstance(dates, pd.RangeIndex) else dates
    for ax, (label, i) in zip(axes, picks):
        ax.plot(x, np.where(na1[i], np.nan, raw[i]), lw=.8, color="gray", label="raw")
        ax.plot(x, cleaned[i], lw=.8, color="tab:blue", alpha=.8, label="cleaned")
        ii = np.where(interp_mask[i])[0]
        ax.scatter(np.asarray(x)[ii], filled[i][ii], s=10, color="orange", label=f"interpolated ({len(ii)})", zorder=3)
        cc = np.where(clip_mask[i])[0]
        ax.scatter(np.asarray(x)[cc], filled[i][cc], s=25, color="red", marker="x", label=f"clipped ({len(cc)})", zorder=3)
        zz = np.where(zerofill_mask[i])[0]
        if len(zz):
            ax.scatter(np.asarray(x)[zz], np.zeros(len(zz)), s=6, color="black", label=f"filled 0 ({len(zz)})")
        ax.set(title=f"Customer with {label}  (id={ids[i][:10]}, FLAG={y[i]})", ylabel="kWh/day")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cleaning_example.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------ save cleaned table
    if not args.skip_save:
        print("== Saving cleaned table ==")
        tbl = pd.DataFrame(cleaned[keep], columns=[str(c) for c in date_cols])
        tbl.insert(0, "FLAG", y[keep])
        tbl.insert(0, "CONS_NO", ids[keep])
        try:
            tbl.to_parquet(out / "cleaned_data.parquet", index=False)
            print(f"saved {out / 'cleaned_data.parquet'}")
        except Exception:
            tbl.to_csv(out / "cleaned_data.csv", index=False, float_format="%.4f")
            print(f"saved {out / 'cleaned_data.csv'} (install pyarrow for a smaller, faster parquet file)")

    print("\n" + "=" * 60)
    print(steps_df[["Step", "Cells affected", "% of all cells", "Customers affected"]].to_string(index=False))
    print(f"\nFinal: {n_final:,} customers x {T:,} days | theft rate {y[keep].mean():.2%}")
    print(f"Everything is in: {out.resolve()}")


if __name__ == "__main__":
    main()
