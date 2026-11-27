r"""Local web application for the selected 74-feature XGBoost model.
Run from project root: .\venv\Scripts\python.exe backend/server.py
"""

import json
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))
import joblib
from backend.pages.overview import OverviewPage
from backend.pages.inspection import InspectionPage, save_inspection
from backend.pages.upload import UploadPage
from backend.pages.model import ModelPage
from database.inspection_store import InspectionStore, ConflictError


# รวมบริการของทั้งสี่หน้า โดยใช้โมเดล ฐานข้อมูล และแคชร่วมกัน
class ModelService(OverviewPage, InspectionPage, UploadPage, ModelPage):
    # เปิดฐานข้อมูล โหลดโมเดลและ Threshold แล้วเตรียมล็อกและแคชสำหรับคำขอพร้อมกัน
    def __init__(self, db_path=None):
        self.store = InspectionStore(
            db_path or ROOT / "database/runtime/inspections.sqlite3"
        )
        self.bundle = joblib.load(ROOT / "models/selected_p55.joblib")
        self.model = self.bundle["model"]
        self.model.set_params(n_jobs=4)
        self.threshold = float(self.bundle["threshold"])
        self.lock = threading.RLock()
        self.uploads = OrderedDict()

    # ค้นหาชุดข้อมูลจากแคช หากไม่มีให้อ่านไฟล์ที่บันทึกแล้วคำนวณใหม่โดยใช้รหัสเดิม
    def dataset(self, token=""):
        if not token or token == "test":
            raise ValueError("กรุณานำเข้าหรือเลือกชุดข้อมูลก่อน")
        with self.lock:
            if token not in self.uploads:
                saved = self.store.upload(token)
                if not saved:
                    raise ValueError("ไม่พบชุดข้อมูล กรุณาเลือกชุดข้อมูลอีกครั้ง")
                self.upload(saved[1], saved[0], existing_token=token)
            return self.uploads[token]


SERVICE = None
STATIC = ROOT / "frontend"
PAGE_NAMES = ("overview", "inspection", "upload", "model")
STATIC_ROUTES = {"/": "index.html", "/index.html": "index.html"}
STATIC_ROUTES.update(
    {"/" + name: name for name in ("style.css", "officer.css", "app.js")}
)
STATIC_ROUTES.update({f"/pages/{name}.js": f"pages/{name}.js" for name in PAGE_NAMES})


# ประกอบโครงหลักกับ HTML ทั้งสี่หน้าก่อนส่ง; โหลดสคริปต์หลังมีองค์ประกอบครบแล้ว
def render_index():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in PAGE_NAMES:
        html = html.replace(
            "<!-- page:" + name + " -->",
            (STATIC / "pages" / (name + ".html")).read_text(encoding="utf-8"),
        )
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    # ส่งเนื้อหา HTTP พร้อมชนิดไฟล์ ขนาด และชื่อไฟล์ดาวน์โหลดเมื่อระบุ
    def send_bytes(
        self, body, kind="application/json; charset=utf-8", status=200, download=None
    ):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{download}"'
            )
        self.end_headers()
        self.wfile.write(body)

    # แปลงผลเป็น JSON ภาษาไทย ไม่อนุญาต NaN แล้วส่งผ่าน send_bytes
    def json(self, value, status=200):
        self.send_bytes(
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            status=status,
        )

    # เลือกเส้นทางอ่านข้อมูล API ดาวน์โหลด CSV หรือส่งไฟล์หน้าเว็บ พร้อมจัดการข้อผิดพลาด
    def do_GET(self):
        path = urlparse(self.path)
        q = parse_qs(path.query)
        get = lambda k, default="": q.get(k, [default])[0]
        try:
            dataset = get("dataset", "")
            if path.path == "/api/datasets":
                self.json(SERVICE.store.datasets())
            elif path.path == "/api/summary":
                self.json(SERVICE.summary(dataset))
            elif path.path in ["/api/meters", "/api/export"]:
                rows = SERVICE.ranked(
                    dataset, get("search"), get("status", "all"), get("work", "all")
                )
                if path.path == "/api/export":
                    safe = rows.copy()
                    for col in safe.select_dtypes(include=["object", "str"]).columns:
                        safe[col] = safe[col].map(
                            lambda v: (
                                "'" + v
                                if isinstance(v, str)
                                and v.startswith(("=", "+", "-", "@", "\t", "\r"))
                                else v
                            )
                        )
                    self.send_bytes(
                        safe.to_csv(index=False).encode("utf-8-sig"),
                        "text/csv; charset=utf-8",
                        download="meter_ranking.csv",
                    )
                else:
                    page = max(1, int(get("page", "1")))
                    size = 20
                    self.json(
                        {
                            "count": len(rows),
                            "page": page,
                            "pages": max(1, (len(rows) + size - 1) // size),
                            "rows": rows.iloc[(page - 1) * size : page * size].to_dict(
                                "records"
                            ),
                        }
                    )
            elif path.path == "/api/meter":
                self.json(SERVICE.detail(get("id"), dataset))
            elif path.path in STATIC_ROUTES:
                name = STATIC_ROUTES[path.path]
                body = (
                    render_index()
                    if name == "index.html"
                    else (STATIC / name).read_bytes()
                )
                self.send_bytes(
                    body,
                    {
                        "html": "text/html; charset=utf-8",
                        "css": "text/css; charset=utf-8",
                        "js": "text/javascript; charset=utf-8",
                    }[name.rsplit(".", 1)[-1]],
                )
            else:
                self.json({"error": "ไม่พบหน้าเว็บ"}, 404)
        except (ValueError, KeyError) as e:
            self.json({"error": str(e)}, 400)
        except Exception:
            import traceback

            traceback.print_exc()
            self.json({"error": "ระบบประมวลผลไม่สำเร็จ กรุณาลองใหม่"}, 500)

    # ตรวจต้นทางและขนาดคำขอ แล้วส่งต่อไปตรวจไฟล์ นำเข้า หรือบันทึกผลตรวจ
    def do_POST(self):
        # This local service accepts same-origin browser requests only.
        origin = self.headers.get("Origin")
        if origin and origin not in [f'http://{self.headers.get("Host")}']:
            self.json({"error": "ไม่อนุญาตคำขอจากเว็บไซต์อื่น"}, 403)
            return
        path = urlparse(self.path)
        q = parse_qs(path.query)
        if path.path not in ["/api/upload", "/api/upload/preview", "/api/inspection"]:
            self.json({"error": "ไม่พบปลายทาง"}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 200 * 1024 * 1024:
                raise ValueError("ไฟล์ต้องมีขนาดไม่เกิน 200 MB")
            body = self.rfile.read(size)
            if path.path == "/api/inspection":
                self.json(save_inspection(SERVICE, json.loads(body)))
            elif path.path == "/api/upload/preview":
                self.json(SERVICE.preview_upload(body))
            else:
                self.json(
                    SERVICE.upload(
                        body,
                        name=q.get("name", [""])[0],
                        allow_duplicate=q.get("allow_duplicate", ["false"])[0]
                        == "true",
                        skip_insufficient=q.get("skip_insufficient", ["false"])[0]
                        == "true",
                    )
                )
        except ConflictError as e:
            self.json({"error": str(e)}, 409)
        except ValueError as e:
            self.json({"error": str(e)}, 400)
        except Exception:
            import traceback

            traceback.print_exc()
            self.json({"error": "วิเคราะห์ไฟล์ไม่สำเร็จ กรุณาตรวจรูปแบบ CSV"}, 500)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("Loading selected XGBoost model...", flush=True)
    SERVICE = ModelService()
    print(f"Power Guard AI ready at http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
