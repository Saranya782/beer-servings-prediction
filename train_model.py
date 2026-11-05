import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Dataset
csv_file = 'beer-servings.csv' if os.path.exists('beer-servings.csv') else 'beer_servings.csv'
df = pd.read_csv(csv_file)

# Drop rows where target variable is missing
df = df.dropna(subset=['total_litres_of_pure_alcohol'])

# 2. Features and Target
num_cols = ['beer_servings', 'spirit_servings', 'wine_servings']
cat_cols = ['country', 'continent']
X = df[num_cols + cat_cols]
y = df['total_litres_of_pure_alcohol']

# 3. Train-Test Split (80% train, 20% evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Data Preprocessor Pipeline (impute missing values + scale/encode)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# 5. Model 1: Ridge Regression with Hyperparameter Tuning
pipe_ridge = Pipeline([
    ('prep', preprocessor),
    ('reg', Ridge())
])
param_grid_ridge = {'reg__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
grid_ridge = GridSearchCV(pipe_ridge, param_grid_ridge, cv=5, scoring='r2')
grid_ridge.fit(X_train, y_train)

# 6. Model 2: Random Forest Regressor with Hyperparameter Tuning
pipe_rf = Pipeline([
    ('prep', preprocessor),
    ('reg', RandomForestRegressor(random_state=42))
])
param_grid_rf = {
    'reg__n_estimators': [50, 100],
    'reg__max_depth': [5, 10, None],
    'reg__min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='r2')
grid_rf.fit(X_train, y_train)

# 7. Evaluate on Test Data
best_ridge = grid_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("=== Evaluation Criteria (R²-score on Test Set) ===")
print(f"Ridge Regression - Best alpha: {grid_ridge.best_params_['reg__alpha']}, R²: {r2_ridge:.4f}, RMSE: {rmse_ridge:.4f}")
print(f"Random Forest    - Best params: {grid_rf.best_params_}, R²: {r2_rf:.4f}, RMSE: {rmse_rf:.4f}")

# 8. Select and Save Best Model
best_model = best_ridge if r2_ridge >= r2_rf else best_rf
winning_name = 'Ridge' if r2_ridge >= r2_rf else 'Random Forest'
print(f"\nWinning Model Selected: {winning_name}")

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Saved best model to 'model.pkl' successfully.")
