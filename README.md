# Power Guard AI

เว็บคัดกรองความเสี่ยงการลักลอบใช้ไฟสำหรับเจ้าหน้าที่ ใช้ XGBoost 74 Features และ Threshold 0.2990

## โครงสร้างโปรเจค

```text
frontend/              HTML, CSS, JavaScript ของเว็บปัจจุบัน
backend/               API, การนำเข้า และการเรียกใช้โมเดล
  server.py            จุดเริ่มเซิร์ฟเวอร์
ml/                    โค้ดสร้าง Features เทรน ประเมิน และทำรายงาน
models/experiments/    โมเดลที่เทรนแล้วและผลทดลอง (ไม่ขึ้น Git)
data/raw/              ข้อมูลต้นฉบับ เช่น data.csv (ไม่ขึ้น Git)
data/processed/        ข้อมูลที่ล้างแล้วและผล EDA (ไม่ขึ้น Git)
database/              โค้ดจัดเก็บผลตรวจและประวัติ
  inspection_store.py  จัดการ SQLite
  runtime/             ฐานข้อมูลจริงและไฟล์นำเข้า (ไม่ขึ้น Git)
tests/                 ชุดทดสอบเว็บและโมเดล
docs/                  คู่มือเว็บและบันทึกการตรวจสอบ
scripts/               เครื่องมือย้ายข้อมูลจากโครงสร้างเก่า
legacy/                เว็บ Streamlit รุ่นเดิม
bundles/               ZIP สำหรับส่งข้อมูลแยก (ไม่ขึ้น Git)
requirements.txt       ไลบรารี Python
```

แก้ UI ที่ `frontend/` แก้ API ที่ `backend/` แก้ขั้นตอนโมเดลที่ `ml/` และแก้การบันทึกผลที่ `database/` โดย `models/` เก็บไฟล์ผลลัพธ์ ไม่ใช่โค้ดฝึก

## ติดตั้งและรัน

รันคำสั่งจากโฟลเดอร์โปรเจค:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m backend.server --port 8766
```

เปิด http://127.0.0.1:8766 เริ่มที่หน้านำเข้า โดยไม่โหลดชุดสาธิตอัตโนมัติ

ก่อนเปิดเว็บต้องมี `models/experiments/results_precision50/selected_p55.joblib` จากเจ้าของโปรเจคที่เชื่อถือได้ ส่วนการใช้ชุดสาธิต/ทดสอบต้องมีไฟล์ต่อไปนี้ด้วย:

```text
data/raw/data.csv
models/experiments/results_precision50/test_predictions.csv
models/experiments/results_precision50/test_comparison.csv
models/experiments/results_all_updated/features.csv
models/experiments/results_all_updated/metrics_table.csv
```

## เพื่อนที่มี ZIP หรือโครงสร้างเดิม

หยุดเว็บก่อนย้ายไฟล์ ดึงโค้ดล่าสุด หากเพิ่งได้รับ ZIP เดิม ให้แตกที่โฟลเดอร์โปรเจค แล้วรัน:

```powershell
.\venv\Scripts\python.exe scripts/migrate_bundle.py
```

คำสั่งนี้ย้าย data.csv, results*, cleaned, eda_results และ local_data ไปที่ใหม่ โดยไม่เขียนทับปลายทางที่มีอยู่แล้ว และไม่แก้เนื้อหาข้อมูล ZIP เดิมยังใช้งานได้ ฐานข้อมูลต้องย้ายจาก local_data ไป database/runtime เพื่อให้เห็นงานที่บันทึกไว้เดิม

## ตรวจสอบ

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

ชุดทดสอบทั้งหมดต้องมีผลทดลองเดิมครบ ถ้าตรวจเฉพาะเว็บ ใช้ `-m unittest tests.test_server -v`

## ฝึกโมเดลและเว็บรุ่นเดิม

รันจากโฟลเดอร์โปรเจค พาธในอาร์กิวเมนต์อ้างอิงโฟลเดอร์ปัจจุบัน:

```powershell
.\venv\Scripts\python.exe ml/train_compare.py --data data/raw/data.csv --out models/experiments/results_new
.\venv\Scripts\python.exe ml/train_precision50.py --out models/experiments/results_precision50_new
.\venv\Scripts\python.exe -m streamlit run legacy/app.py
```

การเทรนรุ่นต่อยอดต้องมีผลทดลองรอบก่อนตามที่สคริปต์ระบุ หลีกเลี่ยงเขียนทับรุ่นที่เว็บใช้อยู่

คู่มือเว็บ: [docs/WEB.md](docs/WEB.md)

ข้อมูลดิบ โมเดล ZIP และฐานข้อมูลจริงไม่ถูกส่งขึ้น GitHub ห้ามใช้ git add -f กับข้อมูลเหล่านี้ เว็บรันเฉพาะเครื่อง ยังไม่มีระบบบัญชีผู้ใช้ คะแนนเป็นสัญญาณคัดกรอง ไม่ใช่ข้อยืนยันการลักลอบ
