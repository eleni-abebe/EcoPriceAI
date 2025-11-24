# EcoPriceAI 🚗💨

**Predict CO₂ Emissions from Vehicle Specifications**  

---

## **Project Overview**
EcoPriceAI predicts **CO₂ emissions (g/km)** using vehicle data like engine size, cylinders, fuel type, and consumption.  
It demonstrates a **full ML workflow**: preprocessing, training, pipeline creation, evaluation, and deployment-ready model export.

---

## **Dataset**
- Source: Kaggle – Fuel Consumption & CO₂ Emissions  
- Rows: 7,385 | Features: 12  
- Selected Features:
  - **Numerical:** Engine Size, Cylinders, Fuel Consumption (City/Hwy/Combined), MPG  
  - **Categorical:** Make, Model, Vehicle Class, Transmission, Fuel Type  

---

## **Workflow**
1. **Data Preprocessing**
   - Missing value check: ✅ No missing data  
   - Numerical scaling (StandardScaler)  
   - Categorical encoding (OneHotEncoder)  

2. **Model Training**
   - Train-test split: 80/20  
   - **Linear Regression** in a Scikit-learn Pipeline  
   - Metrics on test set:
     - MAE: 3.24 g/km  
     - MSE: 29.99 g/km²  
     - RMSE: 5.48 g/km  
     - R²: 0.991  

3. **Model Export**
   - Saved as `model/model.joblib`  
   - Pipeline includes preprocessing + model  
   - Can predict on new inputs without retraining

---
