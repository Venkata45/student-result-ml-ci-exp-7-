import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)
model = joblib.load("student_result_model.pkl")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Placement Prediction</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); width: 360px; }
        h2 { text-align: center; color: #38bdf8; margin-top: 0; }
        label { display: block; margin: 12px 0 4px; font-weight: 500; font-size: 14px; }
        input { width: 100%; box-sizing: border-box; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #f8fafc; font-size: 15px; }
        input:focus { outline: none; border-color: #38bdf8; }
        button { width: 100%; margin-top: 20px; padding: 12px; background: #0284c7; border: none; border-radius: 6px; color: white; font-weight: bold; font-size: 16px; cursor: pointer; transition: 0.2s; }
        button:hover { background: #0369a1; }
        .result { margin-top: 20px; padding: 12px; text-align: center; border-radius: 6px; font-weight: bold; font-size: 18px; }
        .placed { background: #065f46; color: #34d399; }
        .not-placed { background: #7f1d1d; color: #f87171; }
        .status-badge { text-align: center; font-size: 12px; color: #94a3b8; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Placement Predictor</h2>
        <form method="POST" action="/">
            <label>CGPA (e.g. 7.54):</label>
            <input type="number" step="0.01" name="cgpa" value="{{ cgpa or '7.54' }}" required>
            <label>Placement Exam Marks (e.g. 40):</label>
            <input type="number" step="0.1" name="marks" value="{{ marks or '40.0' }}" required>
            <button type="submit">Predict Result</button>
        </form>
        {% if prediction %}
        <div class="result {{ 'placed' if prediction == 'PLACED' else 'not-placed' }}">
            Prediction: {{ prediction }}
        </div>
        {% endif %}
        <div class="status-badge">Container API Status: ONLINE</div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    cgpa = None
    marks = None
    if request.method == "POST":
        cgpa = float(request.form.get("cgpa", 0))
        marks = float(request.form.get("marks", 0))
        input_df = pd.DataFrame([[cgpa, marks]], columns=["cgpa", "placement_exam_marks"])
        pred = model.predict(input_df)[0]
        prediction = "PLACED" if int(pred) == 1 else "NOT_PLACED"
    return render_template_string(HTML_TEMPLATE, prediction=prediction, cgpa=cgpa, marks=marks)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    cgpa = float(data.get("cgpa", 0.0))
    marks = float(data.get("placement_exam_marks", 0.0))
    input_df = pd.DataFrame([[cgpa, marks]], columns=["cgpa", "placement_exam_marks"])
    pred = model.predict(input_df)[0]
    result_label = "PLACED" if int(pred) == 1 else "NOT_PLACED"
    return jsonify({"prediction": result_label}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
