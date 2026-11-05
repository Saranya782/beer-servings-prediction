import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'beer-servings.csv')
model_path = os.path.join(BASE_DIR, 'model.pkl')

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

df = pd.read_csv(csv_path)
countries = sorted(df['country'].dropna().unique().tolist())
continents = sorted(df['continent'].dropna().unique().tolist())

def get_model():
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    # Automatically train model if missing or incompatible with local scikit-learn version
    try:
        from train_model import train_and_save
        return train_and_save()
    except Exception as e:
        print(f"Error training model: {e}")
        return None

model = get_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    global model
    prediction = None
    if request.method == 'POST':
        if model is None:
            model = get_model()

        if model is None:
            prediction = "Error: Unable to load or train model."
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
