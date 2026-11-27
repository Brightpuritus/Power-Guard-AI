"""หน้านำเข้าข้อมูล: ตรวจ CSV ตรวจไฟล์ซ้ำ ข้ามรายที่ใช้ไม่ได้ และทำนาย"""

import csv
import io
import uuid
import warnings
import numpy as np
import pandas as pd
from ml.runtime_features import make_features
from database.inspection_store import ConflictError


class UploadPage:
    # อ่าน CSV ตรวจขนาด รหัส วันที่ และ FLAG; เรียงวันและแทนค่าที่ใช้ไม่ได้ด้วย NaN โดยยังไม่ตัดรายออก
    def parse_upload(self, body):
        if not body or len(body) > 200 * 1024 * 1024:
            raise ValueError("ไฟล์ต้องมีข้อมูลและมีขนาดไม่เกิน 200 MB")
        try:
            text = body.decode("utf-8-sig")
            header = next(csv.reader(io.StringIO(text)))
            if len(header) != len(set(header)):
                raise ValueError("ชื่อคอลัมน์ซ้ำกัน")
            df = pd.read_csv(io.StringIO(text), dtype={"CONS_NO": str}, nrows=50001)
        except (
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            StopIteration,
        ) as e:
            raise ValueError("อ่าน CSV ไม่ได้ กรุณาใช้ไฟล์ CSV แบบ UTF-8") from e
        if "CONS_NO" not in df:
            raise ValueError("ไฟล์ต้องมีคอลัมน์ CONS_NO")
        if not 1 <= len(df) <= 50000:
            raise ValueError("รองรับผู้ใช้ไฟ 1–50,000 รายต่อไฟล์")
        if df.CONS_NO.isna().any() or df.CONS_NO.str.strip().eq("").any():
            raise ValueError("รหัส CONS_NO ต้องไม่ว่าง")
        df.CONS_NO = df.CONS_NO.str.strip()
        if df.CONS_NO.duplicated().any():
            raise ValueError("พบรหัส CONS_NO ซ้ำ")
        dates_cols = [c for c in df if c not in ["CONS_NO", "FLAG"]]
        if not 365 <= len(dates_cols) <= 2000:
            raise ValueError("ต้องมีคอลัมน์รายวัน 365–2,000 วัน")
        dates = pd.to_datetime(pd.Series(dates_cols), errors="coerce")
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("ชื่อคอลัมน์ต้องเป็นวันที่ที่ถูกต้องและไม่ซ้ำ")
        order = np.argsort(dates.to_numpy())
        dates = dates.iloc[order]
        values = (
            df[[dates_cols[i] for i in order]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        values = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
        y = None
        if "FLAG" in df:
            flags = pd.to_numeric(df.FLAG, errors="coerce")
            if not flags.isin([0, 1]).all():
                raise ValueError("FLAG ต้องเป็น 0 หรือ 1 เท่านั้น")
            y = flags.to_numpy(dtype=int)
        return df, dates, values, y

    # สรุปคุณภาพไฟล์ รหัสที่ข้อมูลไม่เพียงพอ และไฟล์ซ้ำ โดยยังไม่ทำนายหรือบันทึกชุดใหม่
    def preview_upload(self, body):
        df, dates, values, y = self.parse_upload(body)
        missing = int(np.isnan(values).sum())
        insufficient = (~np.isfinite(values)).all(axis=1)
        return {
            "count": len(df),
            "eligible_count": int((~insufficient).sum()),
            "insufficient_ids": df.loc[insufficient, "CONS_NO"].tolist(),
            "insufficient_count": int(insufficient.sum()),
            "days": len(dates),
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d"),
            "missing": missing,
            "missing_percent": round(missing / values.size * 100, 2),
            "has_labels": y is not None,
            "duplicates": self.store.duplicates(body),
        }

    # ตรวจไฟล์และการยืนยันข้าม สร้าง Features ทำนาย บันทึกไฟล์ต้นฉบับ แล้วคืนรหัสชุดข้อมูล
    def upload(
        self,
        body,
        name="ข้อมูลอัปโหลด",
        existing_token=None,
        allow_duplicate=True,
        skip_insufficient=False,
    ):
        name = name.strip()
        if not name or len(name) > 120:
            raise ValueError("กรุณาตั้งชื่อชุดข้อมูล 1–120 ตัวอักษร")
        df, dates, values, y = self.parse_upload(body)
        insufficient = (~np.isfinite(values)).all(axis=1)
        skipped = df.loc[insufficient, "CONS_NO"].tolist()
        source_count = len(df)
        if insufficient.all():
            raise ValueError(
                "ไม่มีรายที่มีข้อมูลเพียงพอสำหรับวิเคราะห์ กรุณาใช้ไฟล์อื่น"
            )
        if skipped and not (skip_insufficient or existing_token):
            raise ValueError(
                f"พบข้อมูลไม่เพียงพอ {len(skipped)} ราย กรุณายืนยันข้ามรายเหล่านี้ก่อนนำเข้า"
            )
        if skipped:
            df = df.loc[~insufficient].copy()
            values = values[~insufficient]
            if y is not None:
                y = y[~insufficient]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats = pd.concat(
                [
                    make_features(
                        values[i : i + 1000],
                        dates.dt.dayofweek.to_numpy(),
                        self.bundle["variant"],
                    )[self.bundle["features"]]
                    for i in range(0, len(values), 1000)
                ],
                ignore_index=True,
            )
        feats.index = df.CONS_NO.to_numpy()
        with self.lock:
            if (
                not existing_token
                and not allow_duplicate
                and self.store.duplicates(body)
            ):
                raise ConflictError("พบไฟล์ซ้ำ กรุณาตรวจสอบไฟล์และยืนยันการนำเข้าซ้ำ")
            scores = self.model.predict_proba(feats)[:, 1]
            token = existing_token or uuid.uuid4().hex
            if not existing_token:
                self.store.put_upload(token, name[:120], body)
            self.uploads[token] = {
                "name": name[:120],
                "feats": feats,
                "scores": scores,
                "truth": y,
                "source_count": source_count,
                "skipped_ids": skipped,
                "raw": values,
                "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
            }
            while len(self.uploads) > 3:
                self.uploads.popitem(last=False)
        return {
            "dataset": token,
            "count": len(df),
            "source_count": source_count,
            "skipped_count": len(skipped),
            "message": "วิเคราะห์ข้อมูลเรียบร้อย",
        }
