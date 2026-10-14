import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Naive Bayes model safely using absolute pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "naive.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# HTML Template with Embedded Modern Styles
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Performance/Prediction Portal</title>
    <style>
        :root {
            --primary: #4F46E5;
            --primary-hover: #4338CA;
            --bg-color: #F8FAFC;
            --card-bg: #FFFFFF;
            --text-main: #1E293B;
            --text-muted: #64748B;
            --border-color: #E2E8F0;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 2rem 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            color: var(--text-main);
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }
        .grid-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-main);
            margin-bottom: 0.4rem;
        }
        input, select {
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.95rem;
            background-color: #F8FAFC;
            transition: all 0.2s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #FFFFFF;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        .btn-submit {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.85rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
            margin-top: 0.5rem;
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
        }
        .btn-submit:hover {
            background-color: var(--primary-hover);
        }
        .btn-submit:active {
            transform: scale(0.98);
        }
        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #EEF2FF;
            border: 1px solid #C7D2FE;
            border-radius: 12px;
            text-align: center;
        }
        .result-card h3 {
            color: var(--primary);
            font-size: 1.1rem;
            margin-bottom: 0.25rem;
        }
        .result-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #312E81;
        }
        @media (max-width: 600px) {
            .grid-form {
                grid-template-columns: 1fr;
            }
            .form-group.full-width {
                grid-column: span 1;
            }
            .btn-submit {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Employee Prediction Portal</h1>
            <p>Provide profile attributes below to evaluate metrics via Naive Bayes.</p>
        </div>

        <form method="POST" class="grid-form">
            <div class="form-group">
                <label for="Education">Education</label>
                <select name="Education" id="Education" required>
                    <option value="Bachelors">Bachelors</option>
                    <option value="Masters">Masters</option>
                    <option value="PHD">PHD</option>
                </select>
            </div>

            <div class="form-group">
                <label for="JoiningYear">Joining Year</label>
                <input type="number" name="JoiningYear" id="JoiningYear" value="2020" min="2000" max="2026" required>
            </div>

            <div class="form-group">
                <label for="City">City</label>
                <select name="City" id="City" required>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Pune">Pune</option>
                    <option value="New Delhi">New Delhi</option>
                </select>
            </div>

            <div class="form-group">
                <label for="PaymentTier">Payment Tier</label>
                <select name="PaymentTier" id="PaymentTier" required>
                    <option value="1">Tier 1</option>
                    <option value="2" selected>Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>

            <div class="form-group">
                <label for="Age">Age</label>
                <input type="number" name="Age" id="Age" value="26" min="18" max="70" required>
            </div>

            <div class="form-group">
                <label for="Gender">Gender</label>
                <select name="Gender" id="Gender" required>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="EverBenched">Ever Benched</label>
                <select name="EverBenched" id="EverBenched" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="ExperienceInCurrentDomain">Experience in Current Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" value="2" min="0" max="20" required>
            </div>

            <button type="submit" class="btn-submit">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Prediction Result</h3>
            <div class="result-value">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Extract inputs matching the model's feature names
            education = request.form.get("Education")
            joining_year = float(request.form.get("JoiningYear"))
            city = request.form.get("City")
            payment_tier = float(request.form.get("PaymentTier"))
            age = float(request.form.get("Age"))
            gender = request.form.get("Gender")
            ever_benched = request.form.get("EverBenched")
            exp_domain = float(
                request.form.get("ExperienceInCurrentDomain")
            )

            # Construct DataFrame matching the columns order stored in feature_names_in_
            input_data = pd.DataFrame(
                [[
                    education,
                    joining_year,
                    city,
                    payment_tier,
                    age,
                    gender,
                    ever_benched,
                    exp_domain,
                ]],
                columns=[
                    "Education",
                    "JoiningYear",
                    "City",
                    "PaymentTier",
                    "Age",
                    "Gender",
                    "EverBenched",
                    "ExperienceInCurrentDomain",
                ],
            )

            pred = model.predict(input_data)
            prediction = str(pred[0])
        except Exception as e:
            prediction = f"Error during processing: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
