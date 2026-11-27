"""หน้าข้อมูลโมเดล: ป้ายชื่อ Features และค่าประเมินเมื่อชุดข้อมูลมี FLAG"""

import numpy as np
from ml.runtime_features import FEATURE_LABELS


# แปลงชื่อ Feature เป็นคำอธิบายภาษาไทย รวมชื่อที่มีช่วงวันหรือเดือนต่อท้าย
def feature_label(name):
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    labels = {
        "observed_fraction": "สัดส่วนวันที่มีข้อมูล",
        "observed_mean": "การใช้ไฟเฉลี่ย",
        "observed_median": "ค่ามัธยฐานการใช้ไฟ",
        "observed_std": "ความผันผวนการใช้ไฟ",
        "observed_zero_fraction": "สัดส่วนวันศูนย์ในวันที่มีข้อมูล",
        "missing_run": "ข้อมูลขาดต่อเนื่องนานที่สุด",
        "missing_runs": "จำนวนช่วงข้อมูลขาด",
    }
    for prefix, label in labels.items():
        if name.startswith(prefix + "_") and name[len(prefix) + 1 :].isdigit():
            return f"{label} ({name[len(prefix)+1:]} วัน)"
    if name.startswith("month_mean_lag"):
        return f'การใช้ไฟเฉลี่ย ช่วงย้อนหลังที่ {name.split("lag")[1]} (30 วัน)'
    if name.startswith("month_observed_lag"):
        return f'ความครบถ้วนข้อมูล ช่วงย้อนหลังที่ {name.split("lag")[1]} (30 วัน)'
    if name == "missing_change_30_90":
        return "การเปลี่ยนแปลงข้อมูลขาด 30 วันล่าสุด"
    return name


class ModelPage:
    # คำนวณค่าประเมินและ Precision@K จากป้ายกำกับ โดยไม่เปลี่ยนโมเดลหรือ Threshold
    def model_metrics(self, d):
        result = {}
        scores = d["scores"]
        alert = scores >= self.threshold
        if d["truth"] is not None:
            y = d["truth"]
            tp = int(((y == 1) & alert).sum())
            fp = int(((y == 0) & alert).sum())
            fn = int(((y == 1) & ~alert).sum())
            tn = int(((y == 0) & ~alert).sum())
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            result.update(
                precision=p,
                recall=r,
                f1=2 * p * r / max(p + r, 1e-12),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
            )
            result["topk"] = [
                {"k": k, "found": int(y[np.argsort(-scores, kind="stable")[:k]].sum())}
                for k in [100, 250, 500]
                if k <= len(scores)
            ]
        return result
