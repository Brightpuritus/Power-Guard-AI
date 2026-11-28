"""Durable, dataset-scoped inspection records and append-only revision history."""

import json
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path


class ConflictError(ValueError):
    pass


class InspectionStore:
    # สร้างโฟลเดอร์และตาราง SQLite; อัปเกรดฐานข้อมูลเดิมให้รองรับแฮชตรวจไฟล์ซ้ำ
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS uploads (
                    token TEXT PRIMARY KEY, name TEXT NOT NULL, body BLOB NOT NULL,
                    created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inspections (
                    dataset TEXT NOT NULL, meter TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(dataset,meter));
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, meter TEXT NOT NULL,
                    payload TEXT NOT NULL);
            """)

            columns = {row[1] for row in db.execute("PRAGMA table_info(uploads)")}
            if "digest" not in columns:
                db.execute("ALTER TABLE uploads ADD COLUMN digest TEXT")
            for token, body in db.execute(
                "SELECT token,body FROM uploads WHERE digest IS NULL"
            ):
                db.execute(
                    "UPDATE uploads SET digest=? WHERE token=?",
                    (hashlib.sha256(body).hexdigest(), token),
                )
            db.execute("CREATE INDEX IF NOT EXISTS uploads_digest ON uploads(digest)")

    # คำนวณ SHA-256 ของไฟล์แล้วค้นชุดข้อมูลที่มีเนื้อหาเหมือนกันทุกไบต์
    def duplicates(self, body):
        with self.connect() as db:
            return [
                dict(token=t, name=n, created=c)
                for t, n, c in db.execute(
                    "SELECT token,name,created FROM uploads WHERE digest=? ORDER BY created DESC",
                    (hashlib.sha256(body).hexdigest(),),
                )
            ]

    # เปิดการเชื่อมต่อ SQLite จัดการ commit/rollback และปิดการเชื่อมต่อเสมอ
    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    # คืนสถานะเริ่มต้นสำหรับมิเตอร์ที่ยังไม่มีการบันทึกผลตรวจ
    @staticmethod
    def empty():
        return dict(
            work_status="pending",
            outcome="",
            officer="",
            inspected_on="",
            notes="",
            updated_at="",
            version=0,
        )

    # อ่านผลตรวจล่าสุดทั้งหมดของชุดข้อมูล โดยจับคู่ด้วยรหัสมิเตอร์
    def records(self, dataset):
        with self.connect() as db:
            return {
                m: json.loads(p)
                for m, p in db.execute(
                    "SELECT meter,payload FROM inspections WHERE dataset=?", (dataset,)
                )
            }

    # อ่านผลตรวจล่าสุดและประวัติการแก้ไขของมิเตอร์ในชุดข้อมูลที่ระบุ
    def get(self, dataset, meter):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM inspections WHERE dataset=? AND meter=?",
                (dataset, meter),
            ).fetchone()
            history = [
                json.loads(p)
                for (p,) in db.execute(
                    "SELECT payload FROM history WHERE dataset=? AND meter=? ORDER BY id DESC",
                    (dataset, meter),
                )
            ]
        return {
            "record": json.loads(row[0]) if row else self.empty(),
            "history": history,
        }

    # ตรวจช่องกรอกและ version ก่อนบันทึกผลล่าสุดกับประวัติในธุรกรรมเดียว ป้องกันการเขียนทับงานผู้อื่น
    def save(self, dataset, meter, data):
        if not isinstance(data, dict):
            raise ValueError("ข้อมูลบันทึกไม่ถูกต้อง")
        record = {
            k: data.get(k, "")
            for k in ["work_status", "outcome", "officer", "inspected_on", "notes"]
        }
        if not all(isinstance(v, str) for v in record.values()):
            raise ValueError("รูปแบบข้อมูลไม่ถูกต้อง")
        record = {k: v.strip() for k, v in record.items()}
        if record["work_status"] not in ["pending", "in_progress", "completed"]:
            raise ValueError("สถานะงานไม่ถูกต้อง")
        if record["outcome"] not in ["", "found", "not_found", "inconclusive"]:
            raise ValueError("ผลตรวจไม่ถูกต้อง")
        if not record["officer"] or len(record["officer"]) > 100:
            raise ValueError("ระบุชื่อผู้บันทึกไม่เกิน 100 ตัวอักษร")
        if len(record["notes"]) > 5000:
            raise ValueError("หมายเหตุต้องไม่เกิน 5,000 ตัวอักษร")
        if record["inspected_on"]:
            try:
                date.fromisoformat(record["inspected_on"])
            except ValueError:
                raise ValueError("วันที่ตรวจไม่ถูกต้อง")
            if date.fromisoformat(record["inspected_on"]) > date.today():
                raise ValueError("วันที่ตรวจต้องไม่อยู่ในอนาคต")
        if record["work_status"] == "completed":
            if not record["outcome"] or not record["inspected_on"]:
                raise ValueError("งานตรวจแล้วต้องระบุผลตรวจและวันที่ตรวจ")
        elif record["outcome"] or record["inspected_on"]:
            raise ValueError("ระบุผลตรวจและวันที่ตรวจได้เมื่อสถานะเป็นตรวจแล้วเท่านั้น")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM inspections WHERE dataset=? AND meter=?",
                (dataset, meter),
            ).fetchone()
            if not db.execute("SELECT 1 FROM uploads WHERE token=?", (dataset,)).fetchone():
                raise ValueError("ไม่พบชุดข้อมูล อาจถูกลบแล้ว")
            version = json.loads(row[0])["version"] if row else 0
            if type(data.get("version")) is not int or data["version"] != version:
                raise ConflictError(
                    "รายการถูกแก้ไขแล้ว กรุณาปิดและเปิดรายละเอียดใหม่ก่อนบันทึก"
                )
            record.update(
                version=version + 1, updated_at=datetime.now(timezone.utc).isoformat()
            )
            payload = json.dumps(record, ensure_ascii=False)
            db.execute(
                "INSERT OR REPLACE INTO inspections VALUES (?,?,?)",
                (dataset, meter, payload),
            )
            db.execute(
                "INSERT INTO history(dataset,meter,payload) VALUES (?,?,?)",
                (dataset, meter, payload),
            )
        return self.get(dataset, meter)

    # บันทึกรหัส ชื่อ ไฟล์ต้นฉบับ เวลา และ SHA-256 ของชุดข้อมูลใหม่
    def put_upload(self, token, name, body):
        with self.connect() as db:
            db.execute(
                "INSERT INTO uploads(token,name,body,created,digest) VALUES (?,?,?,?,?)",
                (
                    token,
                    name,
                    body,
                    datetime.now(timezone.utc).isoformat(),
                    hashlib.sha256(body).hexdigest(),
                ),
            )

    # อ่านชื่อและเนื้อหาไฟล์ต้นฉบับจากรหัสชุดข้อมูลเพื่อใช้คำนวณใหม่
    def upload(self, token):
        with self.connect() as db:
            return db.execute(
                "SELECT name,body FROM uploads WHERE token=?", (token,)
            ).fetchone()

    # คืนรายการชุดข้อมูล เรียงจากนำเข้าล่าสุดไปเก่าสุด
    def datasets(self):
        with self.connect() as db:
            return [
                dict(token=t, name=n, created=c)
                for t, n, c in db.execute(
                    "SELECT token,name,created FROM uploads ORDER BY created DESC"
                )
            ]

    # ลบชุดข้อมูลพร้อมผลตรวจและประวัติในธุรกรรมเดียว โดยไม่กระทบชุดอื่น
    def delete_upload(self, token):
        if not isinstance(token, str) or not token:
            raise ValueError("กรุณาเลือกชุดข้อมูลที่ต้องการลบ")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM uploads WHERE token=?", (token,)).fetchone():
                raise ValueError("ไม่พบชุดข้อมูล อาจถูกลบแล้ว")
            db.execute("DELETE FROM history WHERE dataset=?", (token,))
            db.execute("DELETE FROM inspections WHERE dataset=?", (token,))
            db.execute("DELETE FROM uploads WHERE token=?", (token,))
