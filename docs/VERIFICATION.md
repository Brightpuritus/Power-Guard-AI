# Verification — 2026-11-11

Run from the project directory:

```powershell
.\venv\Scripts\python.exe -m unittest test_project -v
.\venv\Scripts\python.exe -m streamlit run app.py
```

Result: all 7 regression tests passed using the existing saved model; no retraining.

- Upload features for the first 100 source customers match saved training features.
- Predictions for all 6,356 test customers match saved XGBoost predictions.
- SHAP contributions plus the bias reconstruct the prediction margin within 1e-5.
- Synthetic negative usage is treated as missing consistently in training and upload.
- Invalid uploads (empty data, missing/blank/duplicate IDs, too few days, invalid dates or labels) are rejected.
- Upload without FLAG and CSV serialization roundtrip pass.
- Streamlit AppTest renders all four tabs, uploads and analyzes a CSV, activates the download button, changes risk thresholds and meter selection, and handles a bad replacement upload without an unhandled exception.

Fixes: SHAP now uses the same early-stopping range as predict_proba (240 of the saved 290 boosting rounds). Before the fix, maximum SHAP margin mismatch on 100 customers was 0.432563. Upload risk levels refresh when thresholds change; changing the file or model directory clears previous upload results. Training now normalizes negative readings before calculating missing-ratio features, matching upload behavior.

Streamlit and its dependencies were installed into the existing venv because Streamlit was missing. AppTest required execution outside the Windows sandbox because local socket creation stalled inside it.

Scope: automated Streamlit interaction and CSV content checks, not a visual browser review or verification of a file saved by a browser download. Original data, trained models, and saved experiment results were not replaced.
