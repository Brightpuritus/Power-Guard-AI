r"""Local web application for the selected 74-feature XGBoost model.
Run from project root: .\venv\Scripts\python.exe backend/server.py
"""
import csv
import io
import json
import mimetypes
import sys
import threading
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from runtime_features import make_features, FEATURE_LABELS
from database.inspection_store import InspectionStore, ConflictError


def feature_label(name):
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    labels = {'observed_fraction': 'สัดส่วนวันที่มีข้อมูล', 'observed_mean': 'การใช้ไฟเฉลี่ย',
              'observed_median': 'ค่ามัธยฐานการใช้ไฟ', 'observed_std': 'ความผันผวนการใช้ไฟ',
              'observed_zero_fraction': 'สัดส่วนวันศูนย์ในวันที่มีข้อมูล',
              'missing_run': 'ข้อมูลขาดต่อเนื่องนานที่สุด', 'missing_runs': 'จำนวนช่วงข้อมูลขาด'}
    for prefix, label in labels.items():
        if name.startswith(prefix + '_') and name[len(prefix)+1:].isdigit():
            return f'{label} ({name[len(prefix)+1:]} วัน)'
    if name.startswith('month_mean_lag'):
        return f'การใช้ไฟเฉลี่ย ช่วงย้อนหลังที่ {name.split("lag")[1]} (30 วัน)'
    if name.startswith('month_observed_lag'):
        return f'ความครบถ้วนข้อมูล ช่วงย้อนหลังที่ {name.split("lag")[1]} (30 วัน)'
    if name == 'missing_change_30_90': return 'การเปลี่ยนแปลงข้อมูลขาด 30 วันล่าสุด'
    return name


class ModelService:
    def __init__(self, db_path=None):
        self.store = InspectionStore(db_path or ROOT/"database/runtime/inspections.sqlite3")
        self.bundle = joblib.load(ROOT/'models/selected_p55.joblib')
        self.model = self.bundle['model']
        self.model.set_params(n_jobs=4)
        self.threshold = float(self.bundle['threshold'])
        self.lock = threading.RLock()
        self.uploads = OrderedDict()

    def dataset(self, token=''):
        if not token or token == 'test': raise ValueError('กรุณานำเข้าหรือเลือกชุดข้อมูลก่อน')
        with self.lock:
            if token not in self.uploads:
                saved = self.store.upload(token)
                if not saved: raise ValueError('ไม่พบชุดข้อมูล กรุณาเลือกชุดข้อมูลอีกครั้ง')
                self.upload(saved[1], saved[0], existing_token=token)
            return self.uploads[token]

    def summary(self, token=''):
        d = self.dataset(token); scores = d['scores']; alert = scores >= self.threshold
        result = {'name': d['name'], 'count': len(scores), 'alerts': int(alert.sum()),
                  'source_count':d.get('source_count',len(scores)), 'skipped_count':len(d.get('skipped_ids',[])),
                  'threshold': self.threshold, 'features': len(self.bundle['features']),
                  'mean_missing': float(d['feats'].missing_ratio.mean()), 'labelled': d['truth'] is not None}
        if d['truth'] is not None:
            y = d['truth']; tp = int(((y == 1) & alert).sum()); fp = int(((y == 0) & alert).sum())
            fn = int(((y == 1) & ~alert).sum()); tn = int(((y == 0) & ~alert).sum())
            p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
            result.update(precision=p, recall=r, f1=2*p*r/max(p+r,1e-12), tp=tp, fp=fp, fn=fn, tn=tn)
            result['topk'] = [{'k': k, 'found': int(y[np.argsort(-scores, kind='stable')[:k]].sum())}
                              for k in [100,250,500] if k <= len(scores)]
        queue=self.ranked(token)
        queue=queue[queue.alert | (queue.work_status != 'pending')]
        result['work']={k:int((queue.work_status==k).sum()) for k in ['pending','in_progress','completed']}
        result['work']['total']=len(queue)
        return result

    def ranked(self, token='', search='', status='all', work='all'):
        d = self.dataset(token)
        frame = pd.DataFrame({'id': d['feats'].index.astype(str), 'score': d['scores'],
                              'missing': d['feats'].missing_ratio.to_numpy()})
        frame['alert'] = frame.score >= self.threshold
        frame['truth'] = d['truth'] if d['truth'] is not None else None
        frame = frame.sort_values(['score','id'],ascending=[False,True]).reset_index(drop=True)
        frame['rank'] = np.arange(1,len(frame)+1)
        records=self.store.records(token)
        for field in ['work_status','outcome','officer','inspected_on','notes','updated_at']:
            frame[field]=[records.get(m,self.store.empty())[field] for m in frame.id]
        if work == 'open': frame=frame[frame.work_status != 'completed']
        elif work in ['pending','in_progress','completed']: frame=frame[frame.work_status == work]
        if search: frame = frame[frame.id.str.contains(search,case=False,regex=False)]
        if status == 'alert': frame = frame[frame.alert]
        elif status == 'below': frame = frame[~frame.alert]
        elif status == 'queue': frame = frame[frame.alert | (frame.work_status != 'pending')]
        return frame

    def detail(self, meter_id, token=''):
        d = self.dataset(token)
        if meter_id not in d['feats'].index: raise ValueError('ไม่พบรหัสมิเตอร์นี้')
        pos = d['feats'].index.get_loc(meter_id); feats = d['feats'].loc[[meter_id]]
        with self.lock:
            contrib = self.model.get_booster().predict(xgb.DMatrix(feats), pred_contribs=True,
                         iteration_range=(0,self.model.best_iteration+1))[0]
        values,dates = d['raw'][pos].astype(float), d['dates']
        values = np.where(np.isfinite(values)&(values>=0),values,np.nan)
        reasons=[]
        for i in np.argsort(-np.abs(contrib[:-1]))[:8]:
            value=float(feats.iloc[0,i])
            reasons.append({'feature':feats.columns[i],'label':feature_label(feats.columns[i]),
                            'value':value if np.isfinite(value) else None,'impact':float(contrib[i])})
        margin=float(contrib.sum()); reconstructed=float(1/(1+np.exp(-margin)))
        if not np.isclose(reconstructed,d['scores'][pos],atol=1e-5): raise RuntimeError('SHAP score mismatch')
        return {'id':meter_id,'score':float(d['scores'][pos]),'alert':bool(d['scores'][pos]>=self.threshold),
                'missing':float(feats.missing_ratio.iloc[0]),'truth':int(d['truth'][pos]) if d['truth'] is not None else None,
                'dates':dates,'usage':[float(v) if np.isfinite(v) else None for v in values],
                'average':pd.Series(values).rolling(30,min_periods=10).mean().replace({np.nan:None}).tolist(),
                'reasons':reasons,'bias':float(contrib[-1]),'score_from_shap':reconstructed, 'inspection':self.store.get(token,meter_id)}

    def parse_upload(self, body):
        if not body or len(body)>200*1024*1024: raise ValueError('ไฟล์ต้องมีข้อมูลและมีขนาดไม่เกิน 200 MB')
        try:
            text=body.decode('utf-8-sig'); header=next(csv.reader(io.StringIO(text)))
            if len(header)!=len(set(header)): raise ValueError('ชื่อคอลัมน์ซ้ำกัน')
            df=pd.read_csv(io.StringIO(text),dtype={'CONS_NO':str},nrows=50001)
        except (UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError,StopIteration) as e:
            raise ValueError('อ่าน CSV ไม่ได้ กรุณาใช้ไฟล์ CSV แบบ UTF-8') from e
        if 'CONS_NO' not in df: raise ValueError('ไฟล์ต้องมีคอลัมน์ CONS_NO')
        if not 1<=len(df)<=50000: raise ValueError('รองรับผู้ใช้ไฟ 1–50,000 รายต่อไฟล์')
        if df.CONS_NO.isna().any() or df.CONS_NO.str.strip().eq('').any(): raise ValueError('รหัส CONS_NO ต้องไม่ว่าง')
        df.CONS_NO=df.CONS_NO.str.strip()
        if df.CONS_NO.duplicated().any(): raise ValueError('พบรหัส CONS_NO ซ้ำ')
        dates_cols=[c for c in df if c not in ['CONS_NO','FLAG']]
        if not 365<=len(dates_cols)<=2000: raise ValueError('ต้องมีคอลัมน์รายวัน 365–2,000 วัน')
        dates=pd.to_datetime(pd.Series(dates_cols),errors='coerce')
        if dates.isna().any() or dates.duplicated().any(): raise ValueError('ชื่อคอลัมน์ต้องเป็นวันที่ที่ถูกต้องและไม่ซ้ำ')
        order=np.argsort(dates.to_numpy()); dates=dates.iloc[order]
        values=df[[dates_cols[i] for i in order]].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
        values=np.where(np.isfinite(values)&(values>=0),values,np.nan)
        y=None
        if 'FLAG' in df:
            flags=pd.to_numeric(df.FLAG,errors='coerce')
            if not flags.isin([0,1]).all(): raise ValueError('FLAG ต้องเป็น 0 หรือ 1 เท่านั้น')
            y=flags.to_numpy(dtype=int)
        return df, dates, values, y

    def preview_upload(self, body):
        df, dates, values, y = self.parse_upload(body)
        missing=int(np.isnan(values).sum())
        insufficient=(~np.isfinite(values)).all(axis=1)
        return {'count':len(df),'eligible_count':int((~insufficient).sum()),
                'insufficient_ids':df.loc[insufficient,'CONS_NO'].tolist(),'insufficient_count':int(insufficient.sum()),'days':len(dates),'start':dates.min().strftime('%Y-%m-%d'),
                'end':dates.max().strftime('%Y-%m-%d'),'missing':missing,
                'missing_percent':round(missing/values.size*100,2),'has_labels':y is not None,
                'duplicates':self.store.duplicates(body)}

    def upload(self, body, name='ข้อมูลอัปโหลด', existing_token=None, allow_duplicate=True, skip_insufficient=False):
        name=name.strip()
        if not name or len(name)>120: raise ValueError('กรุณาตั้งชื่อชุดข้อมูล 1–120 ตัวอักษร')
        df, dates, values, y = self.parse_upload(body)
        insufficient=(~np.isfinite(values)).all(axis=1)
        skipped=df.loc[insufficient,'CONS_NO'].tolist()
        source_count=len(df)
        if insufficient.all(): raise ValueError('ไม่มีรายที่มีข้อมูลเพียงพอสำหรับวิเคราะห์ กรุณาใช้ไฟล์อื่น')
        if skipped and not (skip_insufficient or existing_token):
            raise ValueError(f'พบข้อมูลไม่เพียงพอ {len(skipped)} ราย กรุณายืนยันข้ามรายเหล่านี้ก่อนนำเข้า')
        if skipped:
            df=df.loc[~insufficient].copy()
            values=values[~insufficient]
            if y is not None: y=y[~insufficient]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            feats=pd.concat([make_features(values[i:i+1000],dates.dt.dayofweek.to_numpy(),self.bundle['variant'])[self.bundle['features']] for i in range(0,len(values),1000)],ignore_index=True)
        feats.index=df.CONS_NO.to_numpy()
        with self.lock:
            if not existing_token and not allow_duplicate and self.store.duplicates(body):
                raise ConflictError('พบไฟล์ซ้ำ กรุณาตรวจสอบไฟล์และยืนยันการนำเข้าซ้ำ')
            scores=self.model.predict_proba(feats)[:,1]
            token=existing_token or uuid.uuid4().hex
            if not existing_token: self.store.put_upload(token,name[:120],body)
            self.uploads[token]={'name':name[:120],'feats':feats,'scores':scores,'truth':y,
                                 'source_count':source_count,'skipped_ids':skipped,'raw':values,'dates':dates.dt.strftime('%Y-%m-%d').tolist()}
            while len(self.uploads)>3: self.uploads.popitem(last=False)
        return {'dataset':token,'count':len(df),'source_count':source_count,'skipped_count':len(skipped),'message':'วิเคราะห์ข้อมูลเรียบร้อย'}


def save_inspection(service, data):
    if not isinstance(data,dict): raise ValueError('ข้อมูลไม่ถูกต้อง')
    token=data.get('dataset',''); meter=data.get('id','')
    if not isinstance(token,str) or not isinstance(meter,str): raise ValueError('รหัสไม่ถูกต้อง')
    if meter not in service.dataset(token)['feats'].index: raise ValueError('ไม่พบมิเตอร์ในชุดข้อมูลนี้')
    return service.store.save(token,meter,data)


SERVICE=None
STATIC=ROOT/'frontend'


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self,body,kind='application/json; charset=utf-8',status=200,download=None):
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
        self.end_headers();self.wfile.write(body)

    def json(self,value,status=200):
        self.send_bytes(json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8'),status=status)

    def do_GET(self):
        path=urlparse(self.path);q=parse_qs(path.query);get=lambda k,default='':q.get(k,[default])[0]
        try:
            dataset=get('dataset','')
            if path.path=='/api/datasets': self.json(SERVICE.store.datasets())
            elif path.path=='/api/summary': self.json(SERVICE.summary(dataset))
            elif path.path in ['/api/meters','/api/export']:
                rows=SERVICE.ranked(dataset,get('search'),get('status','all'),get('work','all'))
                if path.path=='/api/export':
                    safe=rows.copy()
                    for col in safe.select_dtypes(include=['object','str']).columns:
                        safe[col]=safe[col].map(lambda v:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v)
                    self.send_bytes(safe.to_csv(index=False).encode('utf-8-sig'),'text/csv; charset=utf-8',download='meter_ranking.csv')
                else:
                    page=max(1,int(get('page','1')));size=20
                    self.json({'count':len(rows),'page':page,'pages':max(1,(len(rows)+size-1)//size),
                               'rows':rows.iloc[(page-1)*size:page*size].to_dict('records')})
            elif path.path=='/api/meter':self.json(SERVICE.detail(get('id'),dataset))
            elif path.path in ['/','/index.html','/style.css','/officer.css','/app.js']:
                name='index.html' if path.path=='/' else path.path[1:]
                self.send_bytes((STATIC/name).read_bytes(),{'html':'text/html; charset=utf-8','css':'text/css; charset=utf-8','js':'text/javascript; charset=utf-8'}[name.split('.')[-1]])
            else:self.json({'error':'ไม่พบหน้าเว็บ'},404)
        except (ValueError,KeyError) as e:self.json({'error':str(e)},400)
        except Exception:
            import traceback;traceback.print_exc();self.json({'error':'ระบบประมวลผลไม่สำเร็จ กรุณาลองใหม่'},500)

    def do_POST(self):
        # This local service accepts same-origin browser requests only.
        origin=self.headers.get('Origin')
        if origin and origin not in [f'http://{self.headers.get("Host")}']:
            self.json({'error':'ไม่อนุญาตคำขอจากเว็บไซต์อื่น'},403);return
        path=urlparse(self.path); q=parse_qs(path.query)
        if path.path not in ['/api/upload','/api/upload/preview','/api/inspection']:self.json({'error':'ไม่พบปลายทาง'},404);return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=200*1024*1024:raise ValueError('ไฟล์ต้องมีขนาดไม่เกิน 200 MB')
            body=self.rfile.read(size)
            if path.path=='/api/inspection':
                self.json(save_inspection(SERVICE,json.loads(body)))
            elif path.path=='/api/upload/preview': self.json(SERVICE.preview_upload(body))
            else: self.json(SERVICE.upload(body,name=q.get('name',[''])[0],allow_duplicate=q.get('allow_duplicate',['false'])[0]=='true',skip_insufficient=q.get('skip_insufficient',['false'])[0]=='true'))
        except ConflictError as e:self.json({'error':str(e)},409)
        except ValueError as e:self.json({'error':str(e)},400)
        except Exception:
            import traceback;traceback.print_exc();self.json({'error':'วิเคราะห์ไฟล์ไม่สำเร็จ กรุณาตรวจรูปแบบ CSV'},500)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    print('Loading selected XGBoost model...',flush=True)
    SERVICE=ModelService()
    print(f'Power Guard AI ready at http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
