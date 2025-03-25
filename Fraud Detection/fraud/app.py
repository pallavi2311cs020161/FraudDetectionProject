from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained fraud detection model
model = joblib.load('fraud_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        features = [float(request.form[f'feature{i}']) for i in range(1, 6)]  # Adjust based on your model
    except ValueError:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

    # Make prediction
    prediction = model.predict([features])[0]

    # Map prediction to readable output
    result = "Fraudulent" if prediction == 1 else "Legitimate"
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
