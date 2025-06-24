# Project – Organ Failure Following Cardiac Surgery  
*The Use of AI for Prediction Modelling*

### Main Focus  
Using AI-based models to predict postoperative organ failure in cardiac surgery patients based on real clinical data from Rigshospitalet. The goal was to explore whether machine learning can outperform traditional risk scores and support better ICU decision-making.

---

## About the Project

This project was completed as part of the BSc course **"Statistical Methods in AI"** at DTU. It investigates whether deep learning and classical machine learning methods can effectively predict serious postoperative complications — specifically organ failure — following cardiac surgery.

Traditional risk scores like EuroSCORE II and STS primarily focus on mortality. Our approach aimed to go further, by predicting complications earlier and enabling better stratification of patients for intensive care.

---

## Methods

The study used clinical data from approximately **1800 cardiac surgery patients** at Rigshospitalet. The dataset included over 100 features across preoperative, perioperative, and postoperative variables.

**Pipeline summary:**
- **Target definition**: Organ failure, based on SOFA score or specific treatments
- **Preprocessing**: Handling missing data, encoding, standardization
- **Models implemented**:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - Multilayer Perceptron (MLP)
- **Evaluation metrics**: ROC-AUC, PR-AUC, accuracy, sensitivity, specificity
- **Model explanation**: SHAP analysis for the XGBoost model

---

## Repository Structure


Organ_Failure_Following_Cardiac_Surgery/
├── Kode/
│ ├── grouped_eval.py
│ ├── main_model_script.py
│ ├── main_script.ipynb
│ └── performance_eval.py
├── Project_Work_Fagprojekt_Organ_Failure.pdf



- `main_script.ipynb`: Main notebook for model training and evaluation  
- `main_model_script.py`: Model definitions and data loaders  
- `performance_eval.py`: Metric calculation and visualization  
- `grouped_eval.py`: Evaluation grouped by surgery type or organ affected

---

## Data Usage

The dataset used in this project is **not included** in the repository due to privacy and ethical restrictions. All analyses were done using de-identified clinical data. No raw or processed patient data is published or shared.

---

## Requirements

Install Python dependencies using:

```bash
pip install -r requirements.txt


Main packages:
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
shap
torch
