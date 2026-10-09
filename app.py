import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_result_model.pkl")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    cgpa = float(data.get("cgpa", 0.0))
    marks = float(data.get("placement_exam_marks", 0.0))
    
    prediction = model.predict([[cgpa, marks]])[0]
    result_label = "PLACED" if int(prediction) == 1 else "NOT_PLACED"
    
    return jsonify({"prediction": result_label}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
