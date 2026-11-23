import unittest
import warnings
import numpy as np
from experiment_missing import observed_features, extra_features


class MissingFeatureChecks(unittest.TestCase):
    def test_memory_layout_and_batch_invariance(self):
        rng=np.random.default_rng(42)
        raw=rng.uniform(0,20,(30,1034)).astype(np.float32)
        raw[rng.random(raw.shape)<.25]=np.nan
        raw=np.asfortranarray(raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for fn in [lambda a:observed_features(a,np.arange(1034)%7),extra_features]:
                full=fn(raw).iloc[[1,7,22]].to_numpy()
                subset=fn(raw[[1,7,22]]).to_numpy()
                np.testing.assert_allclose(full,subset,rtol=0,atol=0,equal_nan=True)

    def test_missing_does_not_lower_observed_mean(self):
        raw=np.full((2,365),10.0)
        raw[1,-30:]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f=observed_features(raw,np.arange(365)%7)
            e=extra_features(raw)
        np.testing.assert_allclose(f.mean_90,[10,10])
        self.assertEqual(e.observed_fraction_30.iloc[1],0)
        self.assertTrue(np.isnan(e.observed_mean_30.iloc[1]))
        self.assertEqual(e.missing_run_90.iloc[1],30)

    def test_real_zeros_and_all_missing_are_distinct(self):
        raw=np.array([[0.0]*365,[np.nan]*365])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f=observed_features(raw,np.arange(365)%7)
            e=extra_features(raw)
        self.assertEqual(f.mean_90.iloc[0],0)
        self.assertTrue(np.isnan(f.mean_90.iloc[1]))
        np.testing.assert_allclose(f.missing_ratio,[0,1])
        self.assertEqual(e.observed_zero_fraction_30.iloc[0],1)
        self.assertTrue(np.isnan(e.observed_zero_fraction_30.iloc[1]))
        self.assertFalse(np.isinf(f.to_numpy()).any())


if __name__=='__main__':
    unittest.main()
