"""หน้ารายการตรวจสอบ: จัดอันดับ รายละเอียด SHAP และการส่งผลตรวจไปฐานข้อมูล"""

import numpy as np
import pandas as pd
import xgboost as xgb
from .model import feature_label


class InspectionPage:
    # เรียงมิเตอร์ตามคะแนน ผูกสถานะงาน แล้วกรองด้วยรหัส ผลคัดกรอง และสถานะงาน
    def ranked(self, token="", search="", status="all", work="all"):
        d = self.dataset(token)
        frame = pd.DataFrame(
            {
                "id": d["feats"].index.astype(str),
                "score": d["scores"],
                "missing": d["feats"].missing_ratio.to_numpy(),
            }
        )
        frame["alert"] = frame.score >= self.threshold
        frame["truth"] = d["truth"] if d["truth"] is not None else None
        frame = frame.sort_values(["score", "id"], ascending=[False, True]).reset_index(
            drop=True
        )
        frame["rank"] = np.arange(1, len(frame) + 1)
        records = self.store.records(token)
        for field in [
            "work_status",
            "outcome",
            "officer",
            "inspected_on",
            "notes",
            "updated_at",
        ]:
            frame[field] = [records.get(m, self.store.empty())[field] for m in frame.id]
        if work == "open":
            frame = frame[frame.work_status != "completed"]
        elif work in ["pending", "in_progress", "completed"]:
            frame = frame[frame.work_status == work]
        if search:
            frame = frame[frame.id.str.contains(search, case=False, regex=False)]
        if status == "alert":
            frame = frame[frame.alert]
        elif status == "below":
            frame = frame[~frame.alert]
        elif status == "queue":
            frame = frame[frame.alert | (frame.work_status != "pending")]
        return frame

    # คืนประวัติการใช้ไฟ ค่าเฉลี่ย 30 วัน SHAP และผลตรวจของมิเตอร์ที่เลือก
    def detail(self, meter_id, token=""):
        d = self.dataset(token)
        if meter_id not in d["feats"].index:
            raise ValueError("ไม่พบรหัสมิเตอร์นี้")
        pos = d["feats"].index.get_loc(meter_id)
        feats = d["feats"].loc[[meter_id]]
        with self.lock:
            contrib = self.model.get_booster().predict(
                xgb.DMatrix(feats),
                pred_contribs=True,
                iteration_range=(0, self.model.best_iteration + 1),
            )[0]
        values, dates = d["raw"][pos].astype(float), d["dates"]
        values = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
        reasons = []
        for i in np.argsort(-np.abs(contrib[:-1]))[:8]:
            value = float(feats.iloc[0, i])
            reasons.append(
                {
                    "feature": feats.columns[i],
                    "label": feature_label(feats.columns[i]),
                    "value": value if np.isfinite(value) else None,
                    "impact": float(contrib[i]),
                }
            )
        margin = float(contrib.sum())
        reconstructed = float(1 / (1 + np.exp(-margin)))
        if not np.isclose(reconstructed, d["scores"][pos], atol=1e-5):
            raise RuntimeError("SHAP score mismatch")
        return {
            "id": meter_id,
            "score": float(d["scores"][pos]),
            "alert": bool(d["scores"][pos] >= self.threshold),
            "missing": float(feats.missing_ratio.iloc[0]),
            "truth": int(d["truth"][pos]) if d["truth"] is not None else None,
            "dates": dates,
            "usage": [float(v) if np.isfinite(v) else None for v in values],
            "average": pd.Series(values)
            .rolling(30, min_periods=10)
            .mean()
            .replace({np.nan: None})
            .tolist(),
            "reasons": reasons,
            "bias": float(contrib[-1]),
            "score_from_shap": reconstructed,
            "inspection": self.store.get(token, meter_id),
        }


# ตรวจว่ามิเตอร์อยู่ในชุดข้อมูลก่อนส่งผลตรวจให้ฐานข้อมูลบันทึก
def save_inspection(service, data):
    if not isinstance(data, dict):
        raise ValueError("ข้อมูลไม่ถูกต้อง")
    token = data.get("dataset", "")
    meter = data.get("id", "")
    if not isinstance(token, str) or not isinstance(meter, str):
        raise ValueError("รหัสไม่ถูกต้อง")
    if meter not in service.dataset(token)["feats"].index:
        raise ValueError("ไม่พบมิเตอร์ในชุดข้อมูลนี้")
    return service.store.save(token, meter, data)
