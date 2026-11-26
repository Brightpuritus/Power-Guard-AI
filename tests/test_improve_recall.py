import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score

from improve_recall import recall_threshold


class ThresholdChecks(unittest.TestCase):
    def test_matches_exhaustive_search_with_tied_scores(self):
        y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        scores = np.array([.9, .8, .8, .7, .6, .6, .2, .1])
        for floor in [.3, .5, .75, 1.0]:
            actual = recall_threshold(y, scores, floor)
            feasible = [t for t in np.unique(scores)
                        if precision_score(y, scores >= t) >= floor]
            self.assertIsNotNone(actual)
            self.assertGreaterEqual(precision_score(y, scores >= actual), floor)
            expected_recall = max(recall_score(y, scores >= t) for t in feasible)
            self.assertEqual(recall_score(y, scores >= actual), expected_recall)

    def test_no_feasible_threshold(self):
        self.assertIsNone(recall_threshold(np.array([0, 1]), np.array([.9, .1]), .8))


class SelectedArtifactChecks(unittest.TestCase):
    def test_selected_model_scores_and_shap(self):
        import theft_core as core
        import xgboost as xgb
        a = core.load_artifacts(Path(__file__).resolve().parents[1] / "models/experiments/results_recall")
        p = a["preds"]
        f = a["feats"].loc[p["CONS_NO"]]
        actual = core.score_features(a["model"], f, a["ref_scores"])
        np.testing.assert_allclose(actual["probability"], p["score_XGBoost"], atol=1e-7)
        f = f.iloc[:100]
        m = a["model"]
        contributions = core.explain(m, f)
        full = m.get_booster().predict(xgb.DMatrix(f), pred_contribs=True,
                                      iteration_range=(0, m.best_iteration + 1))
        np.testing.assert_allclose(contributions.sum(axis=1) + full[:, -1],
                                   m.predict(f, output_margin=True), atol=1e-5)

    def test_dashboard_loads_selected_results(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "legacy/app.py"), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        at.sidebar.text_input[0].set_value("models/experiments/results_recall").run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        self.assertEqual(len(at.tabs), 4)


if __name__ == "__main__":
    unittest.main()
