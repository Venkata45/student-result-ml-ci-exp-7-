import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_result_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Placement Prediction API is running!"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    cgpa = float(data.get("cgpa", 0.0))
    marks = float(data.get("placement_exam_marks", 0.0))
    
    # Use DataFrame to match training feature names
    input_df = pd.DataFrame([[cgpa, marks]], columns=["cgpa", "placement_exam_marks"])
    prediction = model.predict(input_df)[0]
    result_label = "PLACED" if int(prediction) == 1 else "NOT_PLACED"
    
    return jsonify({"prediction": result_label}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
