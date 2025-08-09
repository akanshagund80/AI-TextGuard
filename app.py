from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load("ai_text_detector.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Make prediction
    prediction = model.predict([text])[0]
    
    return jsonify({"AI Text Prediction": "AI-generated" if prediction == 1 else "Human-written"})

if __name__ == '__main__':
    app.run(debug=True)
