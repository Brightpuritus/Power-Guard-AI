# Power Guard AI

เว็บภาษาไทยสำหรับ XGBoost รุ่นล่าสุด รันภายในเครื่องด้วย Python backend และ HTML/CSS/JavaScript

## เปิดใช้งาน

จากโฟลเดอร์โปรเจค:

```powershell
.\venv\Scripts\python.exe web_v2/server.py
```

เปิด http://127.0.0.1:8765 หากพอร์ตถูกใช้ ให้เพิ่ม `--port 8766`

## ความสามารถ

- ภาพรวมชุดทดสอบ 6,356 ราย พร้อม Precision, Recall และ Precision@K
- ค้นหา กรอง จัดอันดับ และส่งออก CSV
- กราฟรายวัน พร้อมค่าเฉลี่ย 30 วัน และ SHAP รายมิเตอร์
- อัปโหลด CSV แบบ UTF-8 มี CONS_NO และคอลัมน์วันที่ 365–2,000 วัน สูงสุด 2,000 ราย/20 MB; FLAG เป็นตัวเลือก
- เปรียบเทียบผล XGBoost ล่าสุดกับ Logistic Regression และ Autoencoder

ใช้ `results_precision50/selected_p55.joblib` และขั้นตอน 74 Features ใน `experiment_missing.py` ค่า threshold จริงคือ 0.29896414279937744 แสดงผลเป็น 0.2990; SHAP ใช้ต้นไม้ถึง best_iteration ให้ตรงกับการทำนาย

ต้องเก็บไฟล์โมเดล, results_all_updated/features.csv, metrics_table.csv, results_precision50/test_predictions.csv, test_comparison.csv และ data.csv ไว้ตามตำแหน่งเดิม การเปิดรายละเอียดครั้งแรกอาจช้าขณะอ่านประวัติรายวัน

ไฟล์อัปโหลดอยู่ในหน่วยความจำล่าสุดไม่เกิน 3 ชุด และหายเมื่อปิดโปรแกรม ค่าติดลบ/ไม่ใช่ตัวเลขถือเป็นข้อมูลขาด โดยไม่แทนด้วยศูนย์ คะแนนไม่ได้ผ่าน probability calibration และต้องยืนยันผลด้วยการตรวจสอบจริง ผลทดสอบเดิมถูกใช้ระหว่างพัฒนาแล้ว จึงควรทดสอบกับข้อมูลใหม่ก่อนใช้งานภาคสนาม

เซิร์ฟเวอร์เปิดเฉพาะ localhost ยังไม่ได้ติดตั้งบนอินเทอร์เน็ตหรือเพิ่มระบบผู้ใช้ เว็บเดิม app.py ยังคงอยู่

## ทดสอบ

```powershell
.\venv\Scripts\python.exe -m unittest web_v2.test_server -v
```
