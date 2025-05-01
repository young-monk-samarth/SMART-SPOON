# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load models
# try:
#     food_model = joblib.load('food_recognition_model.pkl')
#     taste_model = joblib.load('taste_personalization_model.pkl')
#     category_encoder = joblib.load('category_encoder.pkl')
#     texture_encoder = joblib.load('texture_encoder.pkl')
#     salt_encoder = joblib.load('salt_encoder.pkl')
#     diet_encoder = joblib.load('diet_encoder.pkl')
# except FileNotFoundError as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# @app.route('/')
# def home():
#     return app.send_static_file('index.html')

# @app.route('/predict_food', methods=['POST'])
# def predict_food():
#     try:
#         data = request.get_json()
#         category = data['category']
#         texture = data['texture']
#         category_encoded = category_encoder.transform([category])[0]
#         texture_encoded = texture_encoder.transform([texture])[0]
#         prediction = food_model.predict([[category_encoded, texture_encoded]])
#         salt_level = salt_encoder.inverse_transform([prediction[0]])[0]
#         return jsonify({'salt_level': salt_level})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/predict_taste', methods=['POST'])
# def predict_taste():
#     try:
#         data = request.get_json()
#         age = data['age']
#         diet = data['dietary_restriction']
#         saltiness = data['saltiness_rating']
#         umami = data['umami_rating']
#         diet_encoded = diet_encoder.transform([diet])[0]
#         prediction = taste_model.predict([[age, diet_encoded, saltiness, umami]])
#         return jsonify({'intensity': float(prediction[0])})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/submit_feedback', methods=['POST'])
# def submit_feedback():
#     try:
#         data = request.get_json()
#         feedback = data['feedback']
#         with open('feedback.txt', 'a') as f:
#             f.write(feedback + '\n')
#         return jsonify({'message': 'Feedback saved'})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000"]}})

# Load AI models and encoders
try:
    food_model = joblib.load('food_recognition_model.pkl')
    user_model = joblib.load('taste_personalization_model.pkl')
    label_encoder_category = joblib.load('category_encoder.pkl')
    label_encoder_texture = joblib.load('texture_encoder.pkl')
    label_encoder_salt = joblib.load('salt_encoder.pkl')
    label_encoder_diet = joblib.load('diet_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error: One or more .pkl files are missing in the SmartSpoon folder: {e}")
    exit()

# Route for food recognition
@app.route('/predict_food', methods=['POST'])
def predict_food():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    category = data.get('category')
    texture = data.get('texture')
    saltiness_rating = data.get('saltiness_rating', 3)  # Default to 3 if not provided
    if not category or not texture:
        return jsonify({'error': 'Missing category or texture'}), 400
    try:
        category_encoded = label_encoder_category.transform([category])[0]
        texture_encoded = label_encoder_texture.transform([texture])[0]
        food_input = pd.DataFrame({
            'category_encoded': [category_encoded],
            'texture_encoded': [texture_encoded],
            'saltiness_rating': [saltiness_rating]
        })
        salt_prediction = food_model.predict(food_input)
        salt_level = label_encoder_salt.inverse_transform([int(salt_prediction[0])])[0]
        return jsonify({'salt_level': salt_level})
    except Exception as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400

# Route for taste personalization
@app.route('/predict_taste', methods=['POST'])
def predict_taste():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    try:
        age = int(data['age'])
        dietary_restriction = data['dietary_restriction']
        saltiness_rating = int(data['saltiness_rating'])
        umami_rating = int(data['umami_rating'])
        dietary_restriction_encoded = label_encoder_diet.transform([dietary_restriction])[0]
        user_input = pd.DataFrame({
            'age': [age],
            'dietary_restriction_encoded': [dietary_restriction_encoded],
            'saltiness_rating': [saltiness_rating],
            'umami_rating': [umami_rating]
        })
        intensity = user_model.predict(user_input)[0]
        return jsonify({'intensity': round(intensity, 1)})
    except Exception as e:
        return jsonify({'error': f'Invalid user data: {str(e)}'}), 400

# Route for feedback
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    if not data or 'feedback' not in data:
        return jsonify({'error': 'No feedback provided'}), 400
    feedback = data['feedback']
    try:
        with open('feedback.txt', 'a') as f:
            f.write(feedback + '\n')
        return jsonify({'message': 'Feedback saved'})
    except Exception as e:
        return jsonify({'error': f'Failed to save feedback: {str(e)}'}), 500

# Serve the webpage
@app.route('/')
def serve():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)