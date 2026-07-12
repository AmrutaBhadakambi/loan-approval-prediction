import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load the saved model and your data to re-create the encoders
model = joblib.load('loan_approval_model.pkl')
df = pd.read_csv('loan_approval.csv')

# 2. Clean up column names to prevent matching errors
df.columns = df.columns.str.strip()

# 3. Fit the encoders exactly like you did in your train script
from sklearn.preprocessing import LabelEncoder
name_encoder = LabelEncoder()
city_encoder = LabelEncoder()

name_encoder.fit(df['name'])
city_encoder.fit(df['city'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get raw text/numbers from the HTML form
        name_input = request.form['name']
        city_input = request.form['city']
        income = float(request.form['income'])
        credit_score = float(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        years_employed = float(request.form['years_employed'])
        points = float(request.form['points'])

        # Transform the text names into numbers safely
        # If the user types a new name not in the dataset, it uses a fallback code [0]
        try:
            name_code = name_encoder.transform([name_input])[0]
        except:
            name_code = 0
            
        try:
            city_code = city_encoder.transform([city_input])[0]
        except:
            city_code = 0

        # Pass all 9 features (including the dummy 0, 0 columns at the end)
        features = [[name_code, city_code, income, credit_score, loan_amount, years_employed, points, 0, 0]]
        prediction = model.predict(features)

        # Generate the final clean output text
        if prediction[0] == 1:
            result = f"Name: {name_input} | City: {city_input} -> Loan Approved"
        else:
            result = f"Name: {name_input} | City: {city_input} -> Loan Rejected"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)