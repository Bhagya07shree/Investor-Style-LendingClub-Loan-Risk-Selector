# 🏦 Investor-Style LendingClub Loan Risk Selector

An **end-to-end machine learning web app** built with **Python, XGBoost, and Streamlit** to predict loan default risk and support investor-style lending decisions.
Users can enter borrower and loan details, get a calibrated default risk score, an investment recommendation, an estimated financial outcome, and a SHAP-based explanation of the prediction.

---

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)
[![Dataset](https://img.shields.io/badge/Dataset-LendingClub-blue)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
[![Test AUC](https://img.shields.io/badge/Test_AUC-0.7106-brightgreen)](reports/03_final_model_selection.md)
[![Test KS](https://img.shields.io/badge/Test_KS-0.3047-blue)](reports/03_final_model_selection.md)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)](https://matplotlib.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB0028?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)](https://shap.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 🔗 Live App
[Open Live App](https://investor-style-lendingclub-loan-risk-selector-hiiubusxvbncnk2d.streamlit.app/)

---

## ✨ Features

- Predicts a borrower's **probability of default** using a trained XGBoost model
- Classifies loans into a **risk band** (Low / Moderate / High / Very High)
- Lets the user set their own **maximum acceptable risk threshold**
- Estimates the **expected financial outcome** (interest income vs. expected credit loss)
- Explains predictions using **SHAP feature attributions**, with a rule-based fallback
- Includes **example borrower profiles** to explore the model instantly, no data entry required

---

## 📂 Project Structure

LendingClub_End_to_End_ML/
├─ app.py # Streamlit app
├─ available_fields_and_field_categories.py # Input field definitions
├─ feature_explanations.py # Help text for input fields
├─ src/
│ ├─ artifacts/ # Fitted preprocessing objects
│ ├─ models/ # Trained models (raw + calibrated)
│ └─ pipeline/
│ ├─ preprocess.py # Shared transformation logic
│ ├─ train.py # End-to-end training script
│ ├─ predict.py # Inference entry point
│ └─ monitor_drift.py # Drift monitoring (PSI)
├─ notebooks/ # Exploratory analysis
├─ reports/ # Full technical documentation
├─ tests/ # Smoke tests
└─ requirements.txt

---

## 🧠 Model Summary

- **Dataset:** LendingClub accepted loans, 2007–2018 (~2.26M rows)
- **Model:** XGBoost, trained on 2007–2015 with a temporal train/validation/test split
- **Test AUC:** 0.7106 · **Test KS:** 0.3047
- **Calibration:** Isotonic regression corrected the raw model's probabilities (Brier score improved from 0.2171 → 0.1734)
- **Explainability:** SHAP `TreeExplainer` for per-prediction feature attribution

## 📄 Full Technical Reports

For the complete methodology behind this project, see the reports below:

| Report | Contents |
|---|---|
| [00_project_overview.md](reports/00_project_overview.md) | Project summary — start here |
| [01_feature_engineering.md](reports/01_feature_engineering.md) | Leakage checks, target construction, sentinel handling |
| [02_model_comparison.md](reports/02_model_comparison.md) | LR vs. RF vs. XGBoost vs. LightGBM comparison |
| [03_final_model_selection.md](reports/03_final_model_selection.md) | Test results and rationale for the final model |
| [04_calibration.md](reports/04_calibration.md) | Raw vs. calibrated Brier score and AUC |
| [05_feature_drift_analysis.md](reports/05_feature_drift_analysis.md) | PSI findings and structural data events |
| [06_business_metrics.md](reports/06_business_metrics.md) | Threshold analysis and profit/loss tradeoffs |
| [07_deployment.md](reports/07_deployment.md) | Pipeline architecture, the app, known issues |

---

## ⚙️ Deployment on Streamlit Cloud

**Steps:**
1. Push the project to GitHub (excluding `.venv/`)
2. Make sure `src/artifacts/`, `src/models/`, and `requirements.txt` are included
3. Create a new app on [Streamlit Community Cloud](https://share.streamlit.io/)
4. Select the repo, branch `main`, and main file `app.py`
5. Click **Deploy**

For issues encountered and fixed during deployment (Streamlit state bugs, a SHAP/XGBoost version conflict, unrealistic input ranges, etc.), see **[`reports/07_deployment.md`](reports/07_deployment.md)**.

---

## 🖥️ Demo

**App header and Example borrower profiles:**
<img width="938" height="344" alt="Screenshot 2026-08-03 131107" src="https://github.com/user-attachments/assets/a722c868-053c-4a5e-bac5-bba55b25fc1f" />

**Field selection/search:**
<img width="956" height="410" alt="Screenshot 2026-08-03 131353" src="https://github.com/user-attachments/assets/acd30c12-414e-4b86-9b35-c27dedd518fb" />

**Current input summary:**
<img width="959" height="412" alt="Screenshot 2026-08-03 131417" src="https://github.com/user-attachments/assets/2189affd-c312-4748-b38d-84a4bc5b5474" />

**Risk evaluation results (metrics):**
<img width="959" height="335" alt="Screenshot 2026-08-03 131659" src="https://github.com/user-attachments/assets/c00a598f-cce1-4b5c-a237-2b23ce974d5f" />

**Prediction details / diagnostics:**
<img width="1460" height="902" alt="very_high_risk" src="https://github.com/user-attachments/assets/ec31830b-38c2-4db5-964a-576faab126e8" />

**Prediction details / diagnostics:**
<img width="957" height="413" alt="Screenshot 2026-08-03 132003" src="https://github.com/user-attachments/assets/50a86fee-814c-4c35-903d-845f293cd007" />

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/<your-username>/LendingClub_End_to_End_ML.git
cd LendingClub_End_to_End_ML

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Improvements

- Add authentication for multi-user usage
- Automated drift monitoring alerts in production
- Expand explainability to include counterfactual "what-if" scenarios
- Batch scoring support for multiple applications at once

---

## ⚠️ Limitations

See **[`reports/known_limitations.md`](reports/known_limitations.md)** for the full list of caveats (model accuracy ceiling, calibration requirements, historical data gaps, etc.).
