from flask import Flask, request, jsonify
from simple_predict import predict_price

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features field'}), 400

    features_string = data['features']  # Should be a comma-separated string
    prediction = predict_price(features_string)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)