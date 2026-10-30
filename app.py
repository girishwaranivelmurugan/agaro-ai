import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    for root, dirs, files in os.walk(BASE_DIR):
        if filename in files:
            return os.path.join(root, filename)
    parent = os.path.dirname(BASE_DIR)
    for root, dirs, files in os.walk(parent):
        if filename in files:
            return os.path.join(root, filename)
    return os.path.join(BASE_DIR, filename)

template_path = os.path.dirname(find_file('index.html')) if os.path.exists(find_file('index.html')) else os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=template_path)

# Load Models
try:
    crop_model = joblib.load(find_file('crop_model.pkl'))
    fertilizer_model = joblib.load(find_file('fertilizer_model.pkl'))
    crop_yield_model = joblib.load(find_file('crop_yield_model.pkl'))
    print("✓ All Models Loaded Successfully!")
except Exception as e:
    print(f"⚠ Model Loading Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-crop', methods=['POST'])
def predict_crop_advisory():
    try:
        data = request.json or request.form
        N = float(data.get('N', 50))
        P = float(data.get('P', 40))
        K = float(data.get('K', 40))
        temp = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 60))
        ph = float(data.get('ph', 6.5))
        rainfall = float(data.get('rainfall', 200))

        crop_features = [[N, P, K, temp, humidity, ph, rainfall]]
        predicted_crop = str(crop_model.predict(crop_features)[0])

        try:
            fert_df = pd.DataFrame([{
                'Temparature': temp,
                'Humidity ': humidity,
                'Moisture': 50.0,
                'Soil Type': 'Sandy',
                'Crop Type': predicted_crop.capitalize(),
                'Nitrogen': N,
                'Potassium': K,
                'Phosphorous': P
            }])
            predicted_fert = str(fertilizer_model.predict(fert_df)[0])
        except Exception:
            predicted_fert = "Urea / NPK 14-35-14"

        return jsonify({
            'crop': predicted_crop,
            'fertilizer': predicted_fert
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        data = request.json or request.form
        input_data = pd.DataFrame([{
            'Crop': str(data.get('Crop', 'Rice')),
            'Crop_Year': int(data.get('Crop_Year', 2025)),
            'Season': str(data.get('Season', 'Kharif')),
            'State': str(data.get('State', 'Tamil Nadu')),
            'Area': float(data.get('Area', 2.0)),
            'Annual_Rainfall': float(data.get('Annual_Rainfall', 1200)),
            'Fertilizer': float(data.get('Fertilizer', 500)),
            'Pesticide': float(data.get('Pesticide', 20))
        }])
        prediction = crop_yield_model.predict(input_data)[0]
        return jsonify({'recommendation': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
