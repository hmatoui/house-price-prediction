from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import json
import pickle

# Set the working directory to src/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Load the top features from the JSON file
with open('top_features.json', 'r') as f:
    top_features = json.load(f)

# Load feature ranges from the JSON file
with open("feature_ranges.json", "r") as json_file:
    feature_ranges = json.load(json_file)

# Load feature and target scalers
scaler = pickle.load(open("scaler.pkl", "rb"))
target_scaler = pickle.load(open("target_scaler.pkl", "rb"))

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html', features=top_features)

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    try:
        # Get user inputs as floats and store them in a dictionary
        user_inputs = {
            feature: float(request.form[feature]) for feature in feature_ranges.keys()
        }
        
        # Convert the data into a format that the model can work with (2D array)
        features = list(user_inputs.values())
        feature_array = np.array([features]).reshape(1, -1)
        features_scaled = scaler.transform(feature_array)

        # Predict the house price
        prediction_scaled = model.predict(features_scaled)[0]
        prediction = target_scaler.inverse_transform([[prediction_scaled]])[0][0]


        # Return the predicted house price
        return render_template('index.html', prediction_text= float(prediction), feature_ranges=feature_ranges, user_inputs=user_inputs)
    
    except Exception as e:
        # If there's an error, return it in the response
        return render_template('index.html', prediction_text=f"Error: {str(e)}", feature_ranges=feature_ranges, user_inputs=user_inputs)

if __name__ == "__main__":
    app.run(debug=True)
