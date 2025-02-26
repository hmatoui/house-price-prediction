from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import json

# Set the working directory to src/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Load the top features from the JSON file
with open('top_features.json', 'r') as f:
    top_features = json.load(f)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    try:
        # Extract the input values for the top 5 features dynamically
        features = []
        for feature in top_features.keys():  # Use the top feature names dynamically
            features.append(float(request.form[feature]))
        
        # Convert the data into a format that the model can work with (2D array)
        feature_array = np.array([features])

        # Predict the house price
        prediction = model.predict(features)[0]

        # Return the predicted house price
        return render_template('index.html', prediction_text=f"Predicted House Price: ${prediction:,.2f}", features=top_features)
    
    except Exception as e:
        # If there's an error, return it in the response
        return render_template('index.html', prediction_text=f"Error: {str(e)}", features=top_features)

if __name__ == "__main__":
    app.run(debug=True)
