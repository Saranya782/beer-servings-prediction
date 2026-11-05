import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

csv_path = 'beer-servings.csv' if os.path.exists('beer-servings.csv') else 'beer_servings.csv'
df = pd.read_csv(csv_path)
countries = sorted(df['country'].dropna().unique().tolist())
continents = sorted(df['continent'].dropna().unique().tolist())

# Load model if already trained
model = None
if os.path.exists('model.pkl') and os.path.getsize('model.pkl') > 0:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    global model
    prediction = None
    if request.method == 'POST':
        # Re-check model in case it was trained after the app started
        if model is None and os.path.exists('model.pkl') and os.path.getsize('model.pkl') > 0:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)

        if model is None:
            prediction = "Error: 'model.pkl' is empty or missing. Please run 'python train_model.py' first!"
        else:
            input_data = pd.DataFrame([{
                'beer_servings': float(request.form['beer_servings']),
                'spirit_servings': float(request.form['spirit_servings']),
                'wine_servings': float(request.form['wine_servings']),
                'country': request.form['country'],
                'continent': request.form['continent']
            }])
            pred = model.predict(input_data)[0]
            prediction = f"{max(0.0, float(pred)):.2f} Litres"

    # Compute infographic summary stats
    stats = {
        'total_countries': len(df),
        'avg_beer': round(df['beer_servings'].mean(), 1),
        'avg_spirit': round(df['spirit_servings'].mean(), 1),
        'avg_wine': round(df['wine_servings'].mean(), 1)
    }
    return render_template('index.html', countries=countries, continents=continents, stats=stats, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
