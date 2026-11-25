# Power Guard AI

เว็บช่วยเจ้าหน้าที่คัดกรองและจัดลำดับการตรวจสอบการใช้ไฟฟ้า ใช้ XGBoost รุ่นล่าสุด 74 Features และ Threshold 0.29896414279937744 (แสดงเป็น 0.2990)

## เริ่มทำงานร่วมกัน

1. Clone repository แล้วเปิดโฟลเดอร์โปรเจค
2. สร้าง Python virtual environment และติดตั้ง dependencies:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. ขอไฟล์ข้อมูลและโมเดลจากเจ้าของโปรเจค แล้ววางตามโครงสร้างนี้ (ไฟล์เหล่านี้ไม่ได้อยู่ใน Git):

```text
data.csv
results_precision50/
  selected_p55.joblib
  test_predictions.csv
  test_comparison.csv
results_all_updated/
  features.csv
  metrics_table.csv
```

4. เปิดเว็บ:

```powershell
.\venv\Scripts\python.exe web_v2/server.py
```

เข้า http://127.0.0.1:8765 — ต้องมีไฟล์ข้างต้นครบจึงจะเปิดเว็บได้ ใช้โมเดล joblib จากเจ้าของโปรเจคที่เชื่อถือได้เท่านั้น

## โครงสร้างสำคัญ

- `web_v2/static/`: หน้าเว็บ HTML, CSS, JavaScript; `officer.css` คือดีไซน์สำหรับเจ้าหน้าที่
- `web_v2/server.py`: API, โหลดโมเดล, อัปโหลด CSV และ SHAP
- `experiment_missing.py`: ขั้นตอนสร้าง Features รุ่นล่าสุด
- `train_precision50.py`: ทดลองโมเดลภายใต้เงื่อนไข Precision
- `train_compare.py`, `theft_core.py`: โค้ดที่ใช้ร่วมกัน
- `app.py`: เว็บ Streamlit รุ่นเดิม

คู่มือเพิ่มเติม: [web_v2/README.md](web_v2/README.md)

## ตรวจสอบก่อนส่งงาน

```powershell
.\venv\Scripts\python.exe -m unittest web_v2.test_server -v
```

ต้องมีข้อมูลและโมเดลข้างต้นก่อนรันทดสอบ โดยทดสอบคะแนน, Features, SHAP, การกรอง และการตรวจรูปแบบไฟล์

## แนวทางทำงานร่วมกัน

สร้าง branch แยกสำหรับแต่ละงาน เช่น `feature/officer-dashboard` แล้วเปิด Pull Request ให้เพื่อนตรวจทานก่อนรวมเข้า `main` ห้ามเพิ่ม `venv`, ข้อมูลผู้ใช้ไฟ, secrets หรือไฟล์ผลทดลองด้วย `git add -f` ไฟล์ข้อมูลและโมเดลให้แชร์แยกผ่านช่องทางที่ตกลงกัน

เว็บนี้รันเฉพาะเครื่อง มีระบบบันทึกผลตรวจและประวัติใน SQLite แต่ยังไม่มีระบบบัญชีผู้ใช้ คะแนนเป็นสัญญาณคัดกรองและไม่ใช่ข้อยืนยันว่าลักลอบใช้ไฟ
