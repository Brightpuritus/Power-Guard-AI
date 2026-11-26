"""Select recall-oriented XGBoost and threshold on validation, then compare on test.

Run: python improve_recall.py --out results_recall
The original customer split and features are retained. Test is never used to select.
"""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from train_compare import evaluate, SEED


def recall_threshold(y, scores, precision_floor):
    """Maximize recall subject to validation precision; tie-break by precision."""
    p, r, t = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(p[:-1] >= precision_floor)
    if not len(eligible):
        return None
    best = max(eligible, key=lambda i: (r[i], p[i], t[i]))
    return float(t[best])


def metrics(name, y, scores, threshold):
    row = evaluate(name, y, scores, threshold, [100, 250, 500])
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    row.update(Threshold=threshold, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
               Inspections=int(tp + fp))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="models/experiments/results")
    ap.add_argument("--out", default="models/experiments/results_recall")
    ap.add_argument("--max-precision-drop", type=float, default=0.05)
    args = ap.parse_args()
    if not 0 <= args.max_precision_drop < 1:
        ap.error("--max-precision-drop must be in [0, 1)")
    src, out = Path(args.baseline), Path(args.out)
    if out.exists():
        ap.error("Use a new output directory to preserve previous experiments")
    out.mkdir(parents=True)
    (out / "models").mkdir()
    summary = json.loads((src / "run_summary.json").read_text())
    frame = pd.read_csv(src / "features.csv", dtype={"CONS_NO": str})
    X, y = frame[summary["features"]], frame.FLAG.to_numpy()
    idx = np.arange(len(frame))
    tv, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
    tr, va = train_test_split(tv, test_size=0.15 / 0.85, stratify=y[tv], random_state=SEED)
    saved = pd.read_csv(src / "test_predictions.csv", dtype={"CONS_NO": str}).set_index("CONS_NO")
    assert set(frame.CONS_NO.iloc[te]) == set(saved.index), "Test split mismatch"
    baseline = joblib.load(src / "models" / "xgboost.joblib")
    original_threshold = summary["thresholds"]["XGBoost"]
    base_val_scores = baseline.predict_proba(X.iloc[va])[:, 1]
    base_val = metrics("Original", y[va], base_val_scores, original_threshold)
    # A precision loss of five percentage points is operationalized on VALIDATION.
    floor = max(0.35, base_val["Precision"] - args.max_precision_drop)
    print(f"Validation baseline P={base_val['Precision']:.4f}, R={base_val['Recall']:.4f}; precision floor={floor:.4f}", flush=True)
    candidates, objects = [], {}

    def register(name, model, scores):
        threshold = recall_threshold(y[va], scores, floor)
        if threshold is None:
            print(f"{name}: no feasible threshold", flush=True)
            return
        row = metrics(name, y[va], scores, threshold)
        candidates.append(row)
        objects[name] = (model, threshold)
        print(f"{name}: P={row['Precision']:.4f} R={row['Recall']:.4f} AP={row['PR-AUC']:.4f} threshold={threshold:.5f}", flush=True)

    register("Original + tuned threshold", baseline, base_val_scores)
    weight = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    configs = [
        ("Depth3 balanced", dict(max_depth=3, min_child_weight=5, scale_pos_weight=weight)),
        ("Depth4 balanced", dict(max_depth=4, min_child_weight=5, scale_pos_weight=weight)),
        ("Depth6 regularized", dict(max_depth=6, min_child_weight=10, reg_lambda=10, scale_pos_weight=weight)),
        ("Depth4 sqrt weight", dict(max_depth=4, min_child_weight=5, scale_pos_weight=np.sqrt(weight))),
        ("Depth5 sqrt weight", dict(max_depth=5, min_child_weight=5, scale_pos_weight=np.sqrt(weight))),
        ("Depth4 unweighted", dict(max_depth=4, min_child_weight=5, scale_pos_weight=1)),
    ]
    for name, config in configs:
        params = dict(n_estimators=1400, learning_rate=0.03, subsample=0.85,
                      colsample_bytree=0.9, eval_metric="aucpr", early_stopping_rounds=80,
                      n_jobs=4, random_state=SEED, tree_method="hist", reg_lambda=3)
        params.update(config)
        model = XGBClassifier(**params)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        register(name, model, model.predict_proba(X.iloc[va])[:, 1])
    # Freeze the winner before computing any candidate test result.
    winner = max(candidates, key=lambda r: (r["Recall"], r["Precision"], r["PR-AUC"]))
    selected, threshold = objects[winner["Model"]]
    pd.DataFrame(candidates).to_csv(out / "validation_candidates.csv", index=False)
    selection = {"selected_candidate": winner["Model"], "threshold": threshold,
                 "validation_precision_floor": floor, "max_validation_precision_drop": args.max_precision_drop,
                 "baseline_validation": base_val, "selected_validation": winner,
                 "selection_rule": "Maximum validation recall subject to precision floor; ties: precision, PR-AUC",
                 "seed": SEED, "note": "Existing test set has been reported before; fresh external validation remains necessary."}
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    joblib.dump(selected, out / "models" / "xgboost.joblib")
    original_test = baseline.predict_proba(X.iloc[te])[:, 1]
    np.testing.assert_allclose(original_test, saved.loc[frame.CONS_NO.iloc[te], "score_XGBoost"], atol=1e-7)
    tuned_threshold = objects["Original + tuned threshold"][1]
    selected_test = selected.predict_proba(X.iloc[te])[:, 1]
    rows = [metrics("Original", y[te], original_test, original_threshold),
            metrics("Original + tuned threshold", y[te], original_test, tuned_threshold),
            metrics("Selected by validation", y[te], selected_test, threshold)]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison_test.csv", index=False)
    pd.DataFrame([evaluate("XGBoost", y[te], selected_test, threshold, [100, 250, 500])]).to_csv(out / "metrics_table.csv", index=False)
    shutil.copyfile(src / "features.csv", out / "features.csv")
    pd.DataFrame({"CONS_NO": frame.CONS_NO.iloc[te].to_numpy(), "y_true": y[te],
                  "score_XGBoost": selected_test}).to_csv(out / "test_predictions.csv", index=False)
    summary.update(thresholds={"XGBoost": threshold}, best_model_by_val="XGBoost",
                   val_pr_auc={"XGBoost": winner["PR-AUC"]}, recall_experiment=selection)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"CONS_NO": frame.CONS_NO.iloc[va].to_numpy(), "y_true": y[va],
                  "score_original": base_val_scores,
                  "score_selected": selected.predict_proba(X.iloc[va])[:, 1]}).to_csv(out / "validation_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, scores, cut in [("Original", original_test, original_threshold),
                               ("Selected", selected_test, threshold)]:
        p, r, _ = precision_recall_curve(y[te], scores)
        axes[0].plot(r, p, label=name)
        row = metrics(name, y[te], scores, cut)
        axes[0].scatter(row["Recall"], row["Precision"], s=55)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve", ylim=(0, 1))
    axes[0].legend()
    x = np.arange(3)
    for i, col in enumerate(["Precision", "Recall", "F1"]):
        axes[1].bar(x + i * .24, comparison[col], width=.24, label=col)
    axes[1].set(xticks=x+.24, xticklabels=["Original", "Threshold only", "Selected"], ylim=(0, 1), title="Frozen operating points on test")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    report = ["# Recall improvement experiment", "", f"Selected candidate: {winner['Model']}",
              f"Validation precision floor: {floor:.4f}. All selection used validation only.", "",
              "| Variant | Precision | Recall | F1 | PR-AUC | TP | FP | FN | Inspections |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['Model']} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['PR-AUC']:.4f} | {row['TP']} | {row['FP']} | {row['FN']} | {row['Inspections']} |")
    report += ["", "Precision constraints on validation do not guarantee the same precision on test or new data.",
               "The original split and 21 features were reused. No candidate was selected using test metrics.",
               "Because this test set was previously inspected, these are comparative development results, not fresh external confirmation.",
               "Dashboard percentile risk levels are separate from the probability decision threshold in this report.",
               "To inspect the selected artifact in the dashboard, set the results folder to this output directory."]
    (out / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + comparison.to_string(index=False), flush=True)
    print(f"Selected: {winner['Model']}; saved in {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
