# Beer Servings & Alcohol Consumption Predictor

A Flask web application that predicts the total litres of pure alcohol consumption based on country, continent, and beverage servings (beer, spirit, and wine), powered by Machine Learning (Ridge Regression / Random Forest).

---

## Prerequisites

1. **Python**: Ensure Python (3.10 - 3.12 recommended) is installed on your computer.
   - When installing Python on Windows, make sure to check **"Add python.exe to PATH"**.

---

## Setup & Running the Project

### 1. Open Terminal / PowerShell in the project directory
```bash
cd c:\Users\CELLFI\beer-servings-prediction
```

### 2. (Optional but Recommended) Create & Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Model
Before starting the web app, train and evaluate the regression models to generate `model.pkl`:
```bash
python train_model.py
```
This script will:
- Load the dataset (`beer-servings.csv`).
- Handle missing values automatically using median imputation.
- Train Ridge Regression and Random Forest Regressor models using 5-fold cross-validation and hyperparameter tuning.
- Evaluate both models on test data using R²-score and RMSE.
- Save the winning model to `model.pkl`.

### 5. Run the Flask Web Application
```bash
python app.py
```

### 6. View the App in your Browser
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

