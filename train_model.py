import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv("placement_data.csv")
    X = df[["cgpa", "placement_exam_marks"]]
    y = df["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f)

    joblib.dump(model, "student_result_model.pkl")
    print(f"Model trained successfully. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
