import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def generate_metrics(X_test, y_test_enc, model, encoder):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_enc, y_pred)

    report = classification_report(
        y_test_enc,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test_enc, y_pred).tolist()

    metrics = {
        "overall_accuracy": round(float(accuracy), 4),
        "total_speakers": len(encoder.classes_),
        "classification_report": report,
        "confusion_matrix": cm
    }

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved successfully.")
