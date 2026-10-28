
import os
import pickle
 
import pandas as pd
from flask import Flask, render_template_string, request
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "naive.pkl")
 
app = Flask(__name__)
 
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
FEATURES = list(model.feature_names_in_)
 
# Label encodings (alphabetical, as produced by sklearn LabelEncoder).
# If your training notebook used different codes, change them here.
EDUCATION = {"Bachelors": 0, "Masters": 1, "PHD": 2}
CITY = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}
GENDER = {"Female": 0, "Male": 1}
BENCHED = {"No": 0, "Yes": 1}
 
DEFAULTS = {
    "education": "Bachelors",
    "joining_year": 2015,
    "city": "Bangalore",
    "payment_tier": 3,
    "age": 28,
    "gender": "Male",
    "ever_benched": "No",
    "experience": 2,
}
 
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retention Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #13234a;
    --ink-soft: #2b3f73;
    --teal: #0f9d8a;
    --teal-soft: #dff5f1;
    --amber: #f2a93b;
    --coral: #e8553f;
    --coral-soft: #fde8e3;
    --paper: #f2f5fb;
    --card: #ffffff;
    --line: #dde3f0;
    --text: #1c2540;
    --muted: #6b7694;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--paper);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.5;
  }
 
  .shell {
    display: grid;
    grid-template-columns: 380px 1fr;
    min-height: 100vh;
  }
 
  /* ---------- left brand panel ---------- */
  .brand {
    background: linear-gradient(165deg, var(--ink) 0%, var(--ink-soft) 100%);
    color: #fff;
    padding: 56px 40px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
  }
  .brand::after {
    content: "";
    position: absolute;
    width: 320px; height: 320px;
    right: -140px; bottom: -120px;
    border-radius: 50%;
    border: 44px solid rgba(15, 157, 138, 0.35);
  }
  .logo {
    display: inline-flex; align-items: center; gap: 10px;
    font-weight: 700; font-size: 15px; letter-spacing: .2px;
  }
  .logo i {
    width: 34px; height: 34px; border-radius: 10px;
    background: var(--teal);
    display: grid; place-items: center;
    font-style: normal; font-weight: 800;
  }
  .brand h1 {
    font-size: 36px; line-height: 1.15; font-weight: 800;
    margin: 56px 0 18px;
  }
  .brand p { color: #c3cdea; font-size: 15px; max-width: 290px; }
  .legend { display: grid; gap: 12px; margin-top: 36px; font-size: 14px; position: relative; z-index: 1; }
  .legend div { display: flex; align-items: center; gap: 10px; color: #dfe6fb; }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .dot.stay { background: var(--teal); }
  .dot.leave { background: var(--coral); }
  .foot { font-size: 12px; color: #93a0c8; position: relative; z-index: 1; }
 
  /* ---------- right content ---------- */
  .main { padding: 56px 64px; max-width: 900px; width: 100%; }
  .main h2 { font-size: 24px; font-weight: 700; color: var(--ink); }
  .main .sub { color: var(--muted); margin: 4px 0 28px; font-size: 14px; }
 
  form.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 32px;
    box-shadow: 0 10px 30px rgba(19, 35, 74, 0.06);
  }
  .group-title {
    font-size: 13px; font-weight: 700; color: var(--teal);
    margin-bottom: 14px;
  }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px 22px; margin-bottom: 28px; }
  .field label { display: block; font-size: 13px; font-weight: 600; color: var(--ink); margin-bottom: 6px; }
  .field input, .field select {
    width: 100%; height: 46px; padding: 0 14px;
    border: 1.5px solid var(--line); border-radius: 12px;
    background: #fafbfe; color: var(--text);
    font: inherit; font-size: 15px;
    transition: border-color .15s, box-shadow .15s, background .15s;
  }
  .field input:focus, .field select:focus {
    outline: none; border-color: var(--teal); background: #fff;
    box-shadow: 0 0 0 4px rgba(15, 157, 138, 0.15);
  }
  .field small { color: var(--muted); font-size: 12px; }
 
  button {
    width: 100%; height: 52px; border: 0; border-radius: 14px;
    background: linear-gradient(90deg, var(--teal), #12b3a0);
    color: #fff; font: inherit; font-weight: 700; font-size: 16px;
    cursor: pointer;
    transition: transform .12s, box-shadow .12s;
    box-shadow: 0 8px 20px rgba(15, 157, 138, 0.28);
  }
  button:hover { transform: translateY(-1px); }
  button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }
 
  /* ---------- result ---------- */
  .result {
    margin-top: 28px; padding: 28px 32px;
    border-radius: 18px; border: 1px solid;
    display: grid; grid-template-columns: auto 1fr; gap: 24px; align-items: center;
  }
  .result.stay  { background: var(--teal-soft);  border-color: #a9e3d9; }
  .result.leave { background: var(--coral-soft); border-color: #f5bfb4; }
  .badge {
    width: 84px; height: 84px; border-radius: 50%;
    display: grid; place-items: center;
    font-size: 30px; font-weight: 800; color: #fff;
  }
  .stay .badge  { background: var(--teal); }
  .leave .badge { background: var(--coral); }
  .result h3 { font-size: 20px; font-weight: 800; color: var(--ink); }
  .result p { color: var(--muted); font-size: 14px; margin-bottom: 14px; }
  .bar { height: 12px; background: #fff; border-radius: 99px; overflow: hidden; display: flex; }
  .bar span:first-child { background: var(--teal); }
  .bar span:last-child  { background: var(--coral); }
  .bar-labels { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-top: 8px; }
  .bar-labels .s { color: var(--teal); }
  .bar-labels .l { color: var(--coral); }
 
  .error {
    margin-top: 20px; padding: 14px 18px; border-radius: 12px;
    background: #fff4e0; border: 1px solid #f7d79a; color: #8a5a00; font-size: 14px;
  }
 
  @media (max-width: 900px) {
    .shell { grid-template-columns: 1fr; }
    .brand { padding: 32px 24px; }
    .brand h1 { margin-top: 28px; font-size: 28px; }
    .legend { margin: 24px 0 0; }
    .foot { margin-top: 24px; }
    .main { padding: 32px 20px; }
  }
  @media (max-width: 560px) {
    .grid { grid-template-columns: 1fr; }
    .result { grid-template-columns: 1fr; text-align: center; justify-items: center; }
  }
</style>
</head>
<body>
<div class="shell">
 
  <aside class="brand">
    <div>
      <span class="logo"><i>R</i> Retention Predictor</span>
      <h1>Will this employee stay?</h1>
      <p>Enter an employee's profile and the Naive Bayes model estimates the chance they leave within two years.</p>
      <div class="legend">
        <div><span class="dot stay"></span> Likely to stay</div>
        <div><span class="dot leave"></span> Likely to leave</div>
      </div>
    </div>
    <div class="foot">Gaussian Naive Bayes &middot; scikit-learn</div>
  </aside>
 
  <main class="main">
    <h2>Employee profile</h2>
    <p class="sub">All fields are required.</p>
 
    <form class="card" method="POST" action="/">
      <div class="group-title">Background</div>
      <div class="grid">
        <div class="field">
          <label for="education">Education</label>
          <select id="education" name="education">
            {% for o in education %}<option value="{{ o }}" {{ 'selected' if v.education == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="age">Age</label>
          <input id="age" type="number" name="age" min="18" max="65" value="{{ v.age }}" required>
        </div>
        <div class="field">
          <label for="gender">Gender</label>
          <select id="gender" name="gender">
            {% for o in gender %}<option value="{{ o }}" {{ 'selected' if v.gender == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="city">City</label>
          <select id="city" name="city">
            {% for o in city %}<option value="{{ o }}" {{ 'selected' if v.city == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
      </div>
 
      <div class="group-title">Role at the company</div>
      <div class="grid">
        <div class="field">
          <label for="joining_year">Joining year</label>
          <input id="joining_year" type="number" name="joining_year" min="2000" max="2030" value="{{ v.joining_year }}" required>
        </div>
        <div class="field">
          <label for="payment_tier">Payment tier</label>
          <select id="payment_tier" name="payment_tier">
            <option value="1" {{ 'selected' if v.payment_tier|string == '1' }}>Tier 1 (highest)</option>
            <option value="2" {{ 'selected' if v.payment_tier|string == '2' }}>Tier 2</option>
            <option value="3" {{ 'selected' if v.payment_tier|string == '3' }}>Tier 3 (lowest)</option>
          </select>
        </div>
        <div class="field">
          <label for="experience">Years of experience in current domain</label>
          <input id="experience" type="number" name="experience" min="0" max="40" value="{{ v.experience }}" required>
        </div>
        <div class="field">
          <label for="ever_benched">Ever benched</label>
          <select id="ever_benched" name="ever_benched">
            {% for o in benched %}<option value="{{ o }}" {{ 'selected' if v.ever_benched == o }}>{{ o }}</option>{% endfor %}
          </select>
          <small>Kept without a project for a month or more.</small>
        </div>
      </div>
 
      <button type="submit">Predict retention</button>
    </form>
 
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
 
    {% if result %}
      <section class="result {{ 'leave' if result.leaves else 'stay' }}">
        <div class="badge">{{ result.headline_pct }}%</div>
        <div>
          <h3>{{ 'Likely to leave' if result.leaves else 'Likely to stay' }}</h3>
          <p>
            {% if result.leaves %}
              The model sees a {{ result.leave_pct }}% chance this employee leaves within two years.
            {% else %}
              The model sees a {{ result.stay_pct }}% chance this employee stays beyond two years.
            {% endif %}
          </p>
          <div class="bar">
            <span style="width: {{ result.stay_pct }}%"></span>
            <span style="width: {{ result.leave_pct }}%"></span>
          </div>
          <div class="bar-labels">
            <span class="s">Stay {{ result.stay_pct }}%</span>
            <span class="l">Leave {{ result.leave_pct }}%</span>
          </div>
        </div>
      </section>
    {% endif %}
  </main>
 
</div>
</body>
</html>
"""
 
 
def render(values, result=None, error=None):
    return render_template_string(
        PAGE,
        v=values,
        result=result,
        error=error,
        education=EDUCATION.keys(),
        city=CITY.keys(),
        gender=GENDER.keys(),
        benched=BENCHED.keys(),
    )
 
 
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render(DEFAULTS)
 
    form = request.form
    values = {k: form.get(k, DEFAULTS[k]) for k in DEFAULTS}
 
    try:
        row = {
            "Education": EDUCATION[values["education"]],
            "JoiningYear": int(values["joining_year"]),
            "City": CITY[values["city"]],
            "PaymentTier": int(values["payment_tier"]),
            "Age": int(values["age"]),
            "Gender": GENDER[values["gender"]],
            "EverBenched": BENCHED[values["ever_benched"]],
            "ExperienceInCurrentDomain": int(values["experience"]),
        }
    except (KeyError, ValueError):
        return render(values, error="Please check your inputs. Every field needs a valid value.")
 
    X = pd.DataFrame([row])[FEATURES]
    proba = model.predict_proba(X)[0]
    leave_p = float(proba[list(model.classes_).index(1)])
    stay_p = 1.0 - leave_p
 
    leave_pct = round(leave_p * 100, 1)
    stay_pct = round(stay_p * 100, 1)
    leaves = leave_p >= 0.5
 
    result = {
        "leaves": leaves,
        "leave_pct": leave_pct,
        "stay_pct": stay_pct,
        "headline_pct": round(leave_pct if leaves else stay_pct),
    }
    return render(values, result=result)
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
