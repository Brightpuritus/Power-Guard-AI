"""Regression checks using saved artifacts; run: python -m unittest test_project -v."""
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import theft_core as core
import train_compare as tc

BASE = Path(__file__).parent


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = core.load_artifacts(BASE / "results")
        cls.raw = pd.read_csv(BASE / "data.csv", nrows=100, dtype={"CONS_NO": str})
        cls.prep = core.prepare_upload(cls.raw, cls.art["features"])

    def test_upload_matches_saved_training_features(self):
        f = self.prep["feats"]
        np.testing.assert_allclose(f, self.art["feats"].loc[f.index, f.columns], rtol=1e-6, atol=1e-7)

    def test_saved_test_predictions_match_model(self):
        p = self.art["preds"]
        f = self.art["feats"].loc[p["CONS_NO"]]
        scored = core.score_features(self.art["model"], f, self.art["ref_scores"])
        np.testing.assert_allclose(scored["probability"], p["score_XGBoost"], atol=1e-7)

    def test_shap_reconstructs_prediction_margin(self):
        m, f = self.art["model"], self.prep["feats"]
        b = m.get_booster()
        best = getattr(m, "best_iteration", None)
        rounds = (0, best + 1) if best is not None else (0, 0)
        bias = b.predict(xgb.DMatrix(f), pred_contribs=True, iteration_range=rounds)[:, -1]
        actual = core.explain(m, f).sum(axis=1).to_numpy() + bias
        np.testing.assert_allclose(actual, m.predict(f, output_margin=True), atol=1e-5)

    def test_negative_values_match_training_path(self):
        raw = self.raw.iloc[:3].copy()
        col = next(c for c in raw.columns if c not in ("CONS_NO", "FLAG"))
        raw.loc[0, col] = -10
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            raw.to_csv(path, index=False)
            _, _, values, dow = tc.load_sgcc(path)
        trained = tc.build_features(tc.clean_series(values), values, dow)
        uploaded = core.prepare_upload(raw, self.art["features"])["feats"]
        np.testing.assert_allclose(trained, uploaded, atol=1e-7)

    def test_upload_validation(self):
        cases = [self.raw.iloc[:0], self.raw.drop(columns="CONS_NO"),
                 pd.concat([self.raw.iloc[:1]] * 2), self.raw.iloc[:, :10]]
        bad_id = self.raw.iloc[:1].copy()
        bad_id["CONS_NO"] = " "
        bad_flag = self.raw.iloc[:1].copy()
        bad_flag["FLAG"] = 0.5
        bad_date = self.raw.iloc[:1].rename(columns={self.raw.columns[-1]: "invalid"})
        for frame in cases + [bad_id, bad_flag, bad_date]:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                core.prepare_upload(frame, self.art["features"])

    def test_unlabelled_upload_and_csv_roundtrip(self):
        prep = core.prepare_upload(self.raw.drop(columns="FLAG"), self.art["features"])
        self.assertIsNone(prep["y"])
        scored = core.score_features(self.art["model"], prep["feats"], self.art["ref_scores"])
        restored = pd.read_csv(io.BytesIO(scored.to_csv(index=False).encode("utf-8-sig")))
        self.assertEqual(restored["CONS_NO"].tolist(), scored["CONS_NO"].tolist())
        np.testing.assert_allclose(restored["risk_score"], scored["risk_score"])

    def test_dashboard_and_threshold_rerun(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(BASE / "app.py"), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        self.assertEqual(len(at.tabs), 4)
        prep = self.prep
        scored = core.score_features(self.art["model"], prep["feats"], self.art["ref_scores"])
        at.file_uploader[0].upload("sample.csv", self.raw.to_csv(index=False).encode("utf-8"), "text/csv").run()
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        self.assertEqual(len(at.session_state["upload"]["scored"]), len(self.raw))
        self.assertEqual(len(at.get("download_button")), 2)
        at.get("download_button")[1].click().run()
        at.sidebar.slider[0].set_value(50).run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        expected = int((scored["risk_score"] >= 50).sum())
        metric = next(m for m in at.metric if m.label == "ความเสี่ยงสูง")
        self.assertEqual(metric.value, f"{expected:,}")
        at.selectbox[0].select(self.art["preds"]["CONS_NO"].iloc[-1]).run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        at.file_uploader[0].clear().upload("bad.csv", b"CONS_NO\nexample\n", "text/csv").run()
        self.assertNotIn("upload", at.session_state)
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        self.assertGreater(len(at.error), 0)


if __name__ == "__main__":
    unittest.main()
