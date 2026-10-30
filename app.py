import os
import joblib
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Google Drive Download Setup for Yield Model
YIELD_MODEL_FILE = 'crop_yield_model.pkl'
DRIVE_FILE_ID = '1g7OdaJpDolBOxxhnaDohonQatHQwCjk4'

if not os.path.exists(YIELD_MODEL_FILE):
    try:
        print("Downloading yield model from Google Drive...")
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        gdown.download(url, YIELD_MODEL_FILE, quiet=False)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Failed to download model from Drive: {e}")

# Load Trained Models
crop_model = None
yield_model = None
fertilizer_model = None

try:
    crop_model = joblib.load('crop_model.pkl')
    print("crop_model.pkl loaded successfully!")
except Exception as e:
    print(f"Error loading crop_model.pkl: {e}")

try:
    if os.path.exists(YIELD_MODEL_FILE):
        yield_model = joblib.load(YIELD_MODEL_FILE)
        print("Yield model loaded successfully!")
    else:
        print("Yield model file does not exist locally.")
except Exception as e:
    print(f"Error loading yield_model: {e}")

try:
    fertilizer_model = joblib.load('fertilizer_model.pkl')
    print("fertilizer_model.pkl loaded successfully!")
except Exception as e:
    print(f"Error loading fertilizer_model.pkl: {e}")


@app.route('/')
def home():
    return jsonify({"status": "AgroAI Backend Working Live!"})


@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        n = float(data.get('N', 50))
        p = float(data.get('P', 40))
        k = float(data.get('K', 40))
        temp = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 60))
        ph = float(data.get('ph', 6.5))
        rainfall = float(data.get('rainfall', 200))

        # Crop Prediction using AI Model
        if crop_model is not None:
            crop_pred = crop_model.predict([[n, p, k, temp, humidity, ph, rainfall]])[0]
            recommended_crop = str(crop_pred)
        else:
            recommended_crop = "Rice" if rainfall > 150 else "Wheat"

        # Fertilizer Prediction using AI Model
        if fertilizer_model is not None:
            try:
                fert_pred = fertilizer_model.predict([[n, p, k, temp, humidity, ph, rainfall]])[0]
                recommended_fertilizer = str(fert_pred)
            except Exception:
                try:
                    fert_pred = fertilizer_model.predict([[n, p, k]])[0]
                    recommended_fertilizer = str(fert_pred)
                except Exception:
                    recommended_fertilizer = "NPK 19-19-19"
        else:
            recommended_fertilizer = "NPK 19-19-19"

        return jsonify({
            "crop": recommended_crop,
            "fertilizer": recommended_fertilizer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        data = request.json
        crop_name = str(data.get('crop', 'rice')).lower().strip()
        area = float(data.get('Area', 2.0))
        rainfall = float(data.get('Annual_Rainfall', 1200))
        fert = float(data.get('Fertilizer', 500))
        pest = float(data.get('Pesticide', 20))

        # Simple label encoder map for common crops if model expects numerical input
        crop_map = {
            'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3,
            'pigeonpeas': 4, 'mothbeans': 5, 'mungbean': 6, 'blackgram': 7,
            'lentil': 8, 'pomegranate': 9, 'banana': 10, 'mango': 11,
            'grapes': 12, 'watermelon': 13, 'muskmelon': 14, 'apple': 15,
            'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19,
            'jute': 20, 'coffee': 21, 'wheat': 22
        }
        crop_code = crop_map.get(crop_name, 0)

        if yield_model is not None:
            try:
                # 1st try: With Crop Code + Area + Rainfall + Fert + Pest
                prediction = yield_model.predict([[crop_code, area, rainfall, fert, pest]])[0]
            except Exception:
                try:
                    # 2nd try: With Area + Rainfall + Fert + Pest
                    prediction = yield_model.predict([[area, rainfall, fert, pest]])[0]
                except Exception:
                    # 3rd try: With Area only
                    prediction = yield_model.predict([[area]])[0]
            yield_val = round(float(prediction), 2)
        else:
            yield_val = round(area * 3.8 + (rainfall * 0.001), 2)

        return jsonify({"recommendation": yield_val})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
