"""Write Thai comparison notes and a chart archive from a completed run."""
import json
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def markdown_table(frame, columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        vals=[]
        for c in columns:
            value=row[c]
            if pd.isna(value):value='ไม่ผ่านเกณฑ์'
            elif c in ['Precision','Recall','Precision@100','Precision@250','Precision@500']:value=f'{float(value)*100:.2f}%'
            elif c in ['F1','PR-AUC','ROC-AUC']:value=f'{float(value):.3f}'
            elif c in ['TP','FP','FN','TN','Inspections']:value=f'{int(value):,}'
            vals.append(str(value))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def main():
    out=Path('models/experiments/results_all_updated')
    table=pd.read_csv(out/'metrics_table.csv')
    original=pd.read_csv('models/experiments/results/metrics_table.csv')
    constrained=pd.read_csv(out/'constrained_metrics.csv')
    summary=json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(11,5.5))
    xx=np.arange(3)
    for i,(col,color) in enumerate([('Precision','#3476B8'),('Recall','#128048'),('F1','#D5812B')]):
        bars=ax.bar(xx+(i-1)*.24,constrained[col],.24,label=col,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=10)
    ax.set(xticks=xx,xticklabels=constrained.Model,ylim=(0,.65),ylabel='Test score',
           title='Recall-oriented thresholds: validation precision floor 37.46%\nThe precision floor is not guaranteed on test')
    ax.legend();fig.tight_layout()
    for extension in ['png','svg']:
        fig.savefig(out/'charts'/('07_recall_operating_points.'+extension),dpi=200,bbox_inches='tight')
    plt.close(fig)
    text=['# เปรียบเทียบการฝึกใหม่ทั้ง 3 โมเดล',
          '', 'ใช้ชุดข้อมูล SGCC เดิม 42,372 ราย แบ่ง Train 29,660 / Validation 6,356 / Test 6,356 รายด้วย seed 42 โดยรักษาสัดส่วนป้ายกำกับ',
          'ใช้ Features ใหม่ 74 ตัวจาก observed-only statistics และ Features หลายช่วงเวลา/คุณภาพข้อมูล เลือกโมเดลและ threshold จาก Validation ก่อนประเมิน Test',
          '', '## ผลรอบใหม่: เกณฑ์ best F1 บน Validation',
          markdown_table(table,['Model','Precision','Recall','F1','PR-AUC','ROC-AUC','Precision@100','Precision@500']),
          '', '## ผลเดิม: เกณฑ์ best F1 บน Validation',
          markdown_table(original,['Model','Precision','Recall','F1','PR-AUC','ROC-AUC','Precision@100','Precision@500']),
          '', '## จำนวนที่ตรวจพบและภาระการตรวจในรอบใหม่',
          markdown_table(table,['Model','TP','FP','FN','TN','Inspections']),
          '', '## เป้าหมายเพิ่ม Recall ภายใต้ Precision ขั้นต่ำ',
          f"กำหนด Precision บน Validation อย่างน้อย {summary['precision_floor']*100:.2f}% และเลือก threshold ที่ให้ Recall สูงสุด ข้อจำกัดนี้ไม่รับประกัน Precision บน Test",
          markdown_table(constrained,['Model','Precision','Recall','F1','TP','FP','FN','Inspections']),
          '', f"โมเดลที่เลือกด้วย Validation AP: {summary['best_model_by_validation_AP']}",
          '', '## วิธีฝึก',
          f"- Logistic Regression: median imputation และ StandardScaler fit บน Train เท่านั้น เลือก C จาก 0.1, 1, 10 ด้วย Validation AP ได้ C={summary['logreg_selected_C']} ใช้ class_weight=balanced",
          '- XGBoost: ใช้ NaN โดยตรง ฝึกใหม่ด้วยพารามิเตอร์ที่เลือกจากรอบ results_recall และ early stopping บน Validation',
          f"- Autoencoder: เปลี่ยนจากข้อมูลรายวันเดิมเป็น Features 74 ตัว เติม median และแปลง quantile เป็นช่วง 0–1 โดย fit preprocessing จากผู้ใช้ไฟปกติใน Train เท่านั้น ฝึก network จากกลุ่มปกติและ early stop ด้วย validation reconstruction loss ของกลุ่มปกติ จำนวน epoch ที่รัน {summary['autoencoder_epochs']}",
          '- Autoencoder ใช้คะแนน reconstruction error ค่าสูงแสดงความผิดปกติ การเลือก threshold ใช้ป้ายกำกับ Validation แต่ไม่ได้ใช้ป้ายกำกับ Test ฝึก',
          '', '## ข้อควรตีความ',
          '- กราฟก่อนและหลังเป็นการเปลี่ยนทั้ง preprocessing และ Features โดย Autoencoder เปลี่ยน representation ด้วย จึงไม่ใช่ผลของ missing-value handling เพียงอย่างเดียว',
          '- ผล best-F1 กับผล precision-constrained ใช้ threshold คนละชุด ห้ามนำ Precision ของชุดหนึ่งไปจับคู่กับ Recall ของอีกชุด',
          '- Autoencoder ที่เกณฑ์ precision-constrained แจ้งเตือนเพียง 10 รายบน Test และพบจริง 1 ราย จึงไม่เหมาะกับเป้าหมายเพิ่ม Recall แม้ threshold นี้ผ่านเกณฑ์บน Validation',
          '- ใช้ Test เดิมที่เคยรายงานแล้ว ผลนี้เป็นการเปรียบเทียบระหว่างพัฒนา ควรยืนยันด้วยข้อมูลใหม่ก่อนใช้งานจริง',
          '- เป็นการฝึกและประเมินรอบเดียว ยังไม่มี confidence interval หรือการทำซ้ำหลาย seed',
          '- โมเดลใช้ pipeline ใหม่ 74 ตัว ยังไม่ได้เชื่อมเข้า Dashboard เดิม ฟังก์ชัน train_all_updated.score_saved รองรับ inference ของทั้งสามโมเดล',
          '', '## ไฟล์กราฟสำหรับ Canva',
          '- charts/01_model_comparison.png: Precision, Recall, F1, PR-AUC และ ROC-AUC ของรอบใหม่',
          '- charts/02_pr_roc_curves.png: PR และ ROC curves',
          '- charts/03_confusion_matrices.png: Confusion matrix ของแต่ละโมเดล',
          '- charts/04_precision_recall_at_k.png: Precision@K และ Recall@K ภายใต้งบตรวจเท่ากัน',
          '- charts/05_original_vs_updated.png: ผลเดิมเทียบรอบใหม่แยกตามโมเดล',
          '- charts/06_autoencoder_loss.png: Training/Validation loss ของ Autoencoder',
          '- charts/07_recall_operating_points.png: ผล Test ที่ threshold เน้น Recall ภายใต้ Precision ขั้นต่ำบน Validation',
          'ทุกกราฟมี SVG สำหรับขยายภาพ และ PNG ความละเอียดสูง',
          '', '## การตรวจสอบ',
          'ยืนยัน test IDs และ labels ตรงกับผลเดิม และโหลดโมเดลจากไฟล์มาคำนวณจากข้อมูลดิบ 30 ราย ให้คะแนนตรงกับผลการประเมินทั้ง 3 โมเดล',
          '', 'รันซ้ำ: `.\\venv\\Scripts\\python.exe train_all_updated.py --out results_all_updated_repeat`']
    (out/'README_TH.md').write_text('\n\n'.join(text),encoding='utf-8')
    with zipfile.ZipFile(out/'charts_for_canva.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((out/'charts').glob('*')):archive.write(p,'charts/'+p.name)
        for name in ['README_TH.md','metrics_table.csv','original_vs_updated.csv','constrained_metrics.csv']:
            archive.write(out/name,name)
    print(table.to_string(index=False))


if __name__=='__main__':main()
