"""หน้าศูนย์ปฏิบัติงาน: สรุปความเสี่ยงและข้อมูลประกอบภาพรวม"""


class OverviewPage:
    # สรุปจำนวนรายที่วิเคราะห์ รายที่ข้าม ผลคัดกรอง และคิวงาน; ประเมินกับ FLAG เฉพาะเมื่อมีป้ายกำกับ
    def summary(self, token=""):
        d = self.dataset(token)
        scores = d["scores"]
        alert = scores >= self.threshold
        result = {
            "name": d["name"],
            "count": len(scores),
            "alerts": int(alert.sum()),
            "source_count": d.get("source_count", len(scores)),
            "skipped_count": len(d.get("skipped_ids", [])),
            "threshold": self.threshold,
            "features": len(self.bundle["features"]),
            "mean_missing": float(d["feats"].missing_ratio.mean()),
            "labelled": d["truth"] is not None,
        }
        result.update(self.model_metrics(d))
        queue = self.ranked(token)
        queue = queue[queue.alert | (queue.work_status != "pending")]
        result["work"] = {
            k: int((queue.work_status == k).sum())
            for k in ["pending", "in_progress", "completed"]
        }
        result["work"]["total"] = len(queue)
        return result
