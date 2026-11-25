import unittest
import tempfile
import numpy as np
import pandas as pd
from web_v2.server import ModelService, ROOT, save_inspection
from pathlib import Path
from web_v2.inspection_store import InspectionStore, ConflictError


class ModelWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.service = ModelService(Path(cls.temp.name)/'test.sqlite3')
        cls.raw = pd.read_csv(ROOT/'data.csv', nrows=3, dtype={'CONS_NO':str})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_preview_naming_and_duplicate_guard(self):
        service=self.service
        raw=self.raw.copy()
        raw.CONS_NO=['preview-1','preview-2','preview-3']
        raw.iloc[0,2]=-10
        body=raw.to_csv(index=False).encode()
        before=len(service.store.datasets())
        preview=service.preview_upload(body)
        self.assertEqual(preview['count'],3)
        self.assertGreater(preview['missing'],0)
        self.assertEqual(len(service.store.datasets()),before)
        with self.assertRaises(ValueError):service.upload(body,name='  ')
        token=service.upload(body,name='  เขตเหนือ  ',allow_duplicate=False)['dataset']
        self.assertEqual(service.store.upload(token)[0],'เขตเหนือ')
        self.assertEqual(service.preview_upload(body)['duplicates'][0]['token'],token)
        with self.assertRaises(ConflictError):service.upload(body,name='ซ้ำ',allow_duplicate=False)
        second=service.upload(body,name='ซ้ำโดยยืนยัน',allow_duplicate=True)['dataset']
        self.assertNotEqual(token,second)
        self.assertEqual(len(InspectionStore(service.store.path).duplicates(body)),2)
        invalid=raw.copy();invalid.CONS_NO=['same']*3
        with self.assertRaises(ValueError):service.preview_upload(invalid.to_csv(index=False).encode())

    def test_inspection_persistence_isolation_and_conflict(self):
        service=self.service
        token=service.upload(self.raw.to_csv(index=False).encode())['dataset']
        meter=self.raw.CONS_NO.iloc[0]
        payload=dict(dataset=token,id=meter,work_status='in_progress',outcome='',officer='Tester',inspected_on='',notes='follow up',version=0)
        result=save_inspection(service,payload)
        self.assertEqual(result['record']['version'],1)
        store=InspectionStore(service.store.path)
        self.assertEqual(store.get(token,meter)['record']['officer'],'Tester')
        self.assertEqual(store.get('test',meter)['record']['version'],0)
        with self.assertRaises(ConflictError):save_inspection(service,payload)
        payload.update(version=1,work_status='completed')
        with self.assertRaises(ValueError):save_inspection(service,payload)
        payload.update(outcome='not_found',inspected_on='2026-01-01')
        result=save_inspection(service,payload)
        self.assertEqual(len(result['history']),2)
        service.uploads.clear()
        self.assertEqual(service.dataset(token)['feats'].index[0],meter)
        self.assertEqual(service.summary(token)['work']['completed'],1)
        self.assertEqual(len(service.ranked(token,work='completed')),1)
        with self.assertRaises(ValueError):save_inspection(service,{**payload,'id':'not-a-meter'})

    def test_startup_does_not_read_demo(self):
        from unittest.mock import patch
        with patch('web_v2.server.pd.read_csv', side_effect=AssertionError('demo read at startup')):
            service=ModelService(Path(self.temp.name)/'lazy.sqlite3')
            self.assertIsNone(service.test)
            self.assertIsNone(service.comparison)

    def test_selected_model_metrics(self):
        s = self.service.summary()
        self.assertEqual((s['count'],s['features'],s['alerts']), (6356,74,362))
        self.assertEqual((s['tp'],s['fp'],s['fn'],s['tn']), (193,169,349,5645))
        self.assertAlmostEqual(s['threshold'],0.29896414279937744)

    def test_upload_matches_training_features_and_shap(self):
        token = self.service.upload(self.raw.to_csv(index=False).encode())['dataset']
        d = self.service.dataset(token)
        saved = pd.read_csv(ROOT/'results_all_updated/features.csv',dtype={'CONS_NO':str}).set_index('CONS_NO')
        np.testing.assert_allclose(d['feats'],saved.loc[self.raw.CONS_NO,self.service.bundle['features']],rtol=1e-4,atol=1e-4,equal_nan=True)
        detail = self.service.detail(self.raw.CONS_NO.iloc[0],token)
        self.assertAlmostEqual(detail['score'],detail['score_from_shap'],places=5)
        self.assertEqual(len(detail['usage']),len(self.raw.columns)-2)

    def test_unlabelled_upload_and_validation(self):
        token = self.service.upload(self.raw.drop(columns='FLAG').to_csv(index=False).encode())['dataset']
        self.assertFalse(self.service.summary(token)['labelled'])
        self.assertNotIn('precision',self.service.summary(token))
        with self.assertRaises(ValueError):self.service.upload(b'CONS_NO,date\na,3')
        duplicate = self.raw.copy();duplicate.loc[1,'CONS_NO']=duplicate.loc[0,'CONS_NO']
        with self.assertRaises(ValueError):self.service.upload(duplicate.to_csv(index=False).encode())

    def test_filter_and_ranking(self):
        rows = self.service.ranked(status='alert')
        self.assertEqual(len(rows),362)
        self.assertTrue(rows.score.is_monotonic_decreasing)
        self.assertTrue((rows.score>=self.service.threshold).all())
        self.assertEqual(len(self.service.ranked(search='no-such-meter')),0)


if __name__=='__main__':unittest.main()
