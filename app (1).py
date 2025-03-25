from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained fraud detection model
model = joblib.load('fraud_model.pkl')

def encode_employment_status(status):
    """Encode employment status as numerical values"""
    encoding = {"employed": 0, "self-employed": 1, "unemployed": 2}
    return encoding.get(status, -1)  # Default to -1 if unknown

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = float(request.form['credit_score'])
        employment_status = encode_employment_status(request.form['employment_status'])
        previous_loans = float(request.form['previous_loans'])

        if employment_status == -1:
            return render_template('result.html', prediction="Invalid employment status.")

        features = [income, loan_amount, credit_score, employment_status, previous_loans]
    except ValueError:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

    # Make prediction
    prediction = model.predict([features])[0]

    # Map prediction to readable output
    result = "Fraudulent" if prediction == 1 else "Legitimate"
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=False)
