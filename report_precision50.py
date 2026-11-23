"""Summarize a completed precision-constrained XGBoost experiment."""
import json
from pathlib import Path
import pandas as pd


def main():
    out=Path('results_precision50')
    table=pd.read_csv(out/'test_comparison.csv')
    winners=json.loads((out/'selection.json').read_text())
    labels={'Current':'โมเดลปัจจุบัน เกณฑ์เดิม','Current threshold-only P50':'โมเดลปัจจุบัน ปรับเกณฑ์ P50',
            'Retrained P50':'ตัวเลือกจาก Validation P50','Retrained P55 buffer':'ตัวเลือกจาก Validation P55'}
    text=['# เพิ่ม Recall โดยรักษา Precision อย่างน้อย 50%',
          '', 'ใช้ Features 74 ตัวและชุดแบ่งเดิม: Train 29,660 / Validation 6,356 / Test 6,356 ราย (seed 42)',
          'ฝึก XGBoost ใหม่ 16 configurations และรวมโมเดลปัจจุบันเป็นตัวเลือก เลือกโมเดล/threshold จาก Validation โดยจัดลำดับ Recall, Precision, PR-AUC ตามลำดับ',
          'กำหนดสองนโยบายก่อนดู Test: Precision บน Validation อย่างน้อย 50% และทางเลือกเผื่อความคลาดเคลื่อนอย่างน้อย 55% ทั้งสองนโยบายเป็นข้อจำกัดเชิงประจักษ์ ไม่รับประกันผลบน Test หรือข้อมูลใหม่',
          '', '| วิธี | Threshold | Precision | Recall | F1 | PR-AUC | TP | FP | FN | จำนวนตรวจ |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in table.iterrows():
        text.append(f"| {labels[r.Model]} | {r.Threshold:.6f} | {r.Precision:.2%} | {r.Recall:.2%} | {r.F1:.3f} | {r['PR-AUC']:.3f} | {r.TP} | {r.FP} | {r.FN} | {r.Inspections} |")
    base=table.iloc[0]
    text += ['', '## ผลของทางเลือกที่ตรึงไว้จาก Validation']
    for key,modelname in [('0.5','Retrained P50'),('0.55','Retrained P55 buffer')]:
        r=table[table.Model==modelname].iloc[0];v=winners[key]
        text += [f"- {labels[modelname]}: candidate {r.Candidate}, Validation Precision {v['Precision']:.2%}, Recall {v['Recall']:.2%}",
                 f"  Test {'ผ่าน' if r.Precision>=.5 else 'ไม่ผ่าน'}เป้าหมาย Precision 50%; Recall เปลี่ยน {100*(r.Recall-base.Recall):+.2f} จุดเปอร์เซ็นต์ พบจริงเปลี่ยน {int(r.TP-base.TP):+d} ราย จำนวนตรวจเปลี่ยน {int(r.Inspections-base.Inspections):+d} ราย"]
    text += ['', '## การใช้และข้อจำกัด',
             '- selected_p50.joblib และ selected_p55.joblib เป็น bundle ของโมเดล, Features และ threshold เรียกผ่าน experiment_missing.score_raw ได้',
             '- ถ้า candidate ที่ชนะเป็น current หมายถึงการฝึกใหม่ยังไม่ชนะโมเดลปัจจุบันตามเงื่อนไขนั้น ชื่อ P50/P55 เป็นนโยบาย threshold ไม่ใช่การรับประกัน Precision ของข้อมูลใหม่',
             '- กราฟ comparison.png และ comparison.svg แสดงผลที่ threshold แต่ละนโยบายและ PR curves',
             '- validation_candidates.csv บันทึกผลทุก candidate บน Validation; selection.json ตรึงตัวเลือกก่อนทำนาย Test; plan.json บันทึกแผนการทดลอง',
             '- ตรวจคะแนน baseline ตรงกับผลเดิม ตรวจ inference จากข้อมูลดิบ 40 ราย และ SHAP additivity 30 รายสำหรับทุกนโยบาย',
             '- Test ชุดนี้เคยถูกตรวจดูระหว่างการพัฒนาแล้ว ผลเป็นหลักฐานเปรียบเทียบระหว่างพัฒนา ควรยืนยันด้วยข้อมูลใหม่ และยังไม่มี confidence interval',
             '- Dashboard เดิมยังไม่ได้เชื่อม pipeline นี้',
             '', 'รันซ้ำจากโฟลเดอร์โปรเจค: `.\\venv\\Scripts\\python.exe train_precision50.py --out results_precision50_repeat`']
    (out/'README_TH.md').write_text('\n'.join(text),encoding='utf-8')
    print('\n'.join(text))


if __name__=='__main__':main()
