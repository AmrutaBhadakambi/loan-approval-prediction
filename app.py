from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = int(request.form['name'])
        city = int(request.form['city'])
        income = float(request.form['income'])
        credit_score = float(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        years_employed = float(request.form['years_employed'])
        points = float(request.form['points'])

        data = pd.DataFrame({
            "name": [name],
            "city": [city],
            "income": [income],
            "credit_score": [credit_score],
            "loan_amount": [loan_amount],
            "years_employed": [years_employed],
            "points": [points]
        })

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)