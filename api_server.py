from flask import Flask, request, jsonify
import os
from simple_predict import predict_price

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "✅ Flask API is alive!"

@app.route('/predict', methods=['POST'])
def predict():
    print("🔥 /predict route was called")

    data = request.get_json()
    print("Received data:", data)

    if not data or 'features' not in data:
        print("❌ Missing 'features' field")
        return jsonify({'error': 'Missing features field'}), 400

    features_string = data['features']
    print("Parsed features:", features_string)

    prediction = predict_price(features_string)
    print("Prediction result:", prediction)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)




