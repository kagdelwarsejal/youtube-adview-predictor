import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

# ---- Load model (kept in /artifacts by your training step) ----
MODEL_CANDIDATES = [
    Path("artifacts/model.joblib"),
    Path("model.joblib"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find model file. Expected at artifacts/model.joblib or model.joblib"
    )

model = joblib.load(MODEL_PATH)

# Try to expose expected column names, if the estimator provides them
FEATURE_NAMES = getattr(model, "feature_names_in_", None)

def predict_json(json_str: str):
    """
    Accepts a JSON object (or list of objects) with feature:value pairs.
    Returns a single prediction for one object, or a table for multiple.
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
            y = model.predict(df)
            return float(y[0])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            y = model.predict(df)
            out = df.copy()
            out["prediction"] = y
            return out
        else:
            return gr.update(value="Error: JSON must be an object or a list of objects.")
    except Exception as e:
        return gr.update(value=f"Error: {e}")

def predict_csv(file):
    """
    Batch prediction from CSV upload. CSV must have the same columns used in training.
    """
    try:
        df = pd.read_csv(file.name)
        y = model.predict(df)
        out = df.copy()
        out["prediction"] = y
        # Save a downloadable CSV
        out_path = "predictions.csv"
        out.to_csv(out_path, index=False)
        return out, out_path
    except Exception as e:
        return gr.update(value=f"Error: {e}"), None

def feature_help_text():
    if FEATURE_NAMES is not None:
        cols = "\n".join(f"- {c}" for c in FEATURE_NAMES)
        return f"**Model expects columns:**\n{cols}"
    return (
        "Model didn’t expose column names. Use the same feature names you used in training.\n"
        "Tip: open your training notebook and copy the final feature column list."
    )

with gr.Blocks(title="YouTube AdView Predictor") as demo:
    gr.Markdown("# 📈 YouTube AdView Predictor\nEnter features and get the predicted AdViews.")

    with gr.Tab("Single / Multiple via JSON"):
        gr.Markdown(
            "Paste a **JSON object** for one prediction or a **JSON list** for multiple rows.\n\n"
            "Example (one row):\n"
            "```json\n"
            "{\"feature_a\": 1.0, \"feature_b\": 2, \"feature_c\": \"some_value\"}\n"
            "```\n"
            "Example (multiple rows):\n"
            "```json\n"
            "[{\"feature_a\": 1.0, \"feature_b\": 2, \"feature_c\": \"x\"},\n"
            " {\"feature_a\": 3.5, \"feature_b\": 5, \"feature_c\": \"y\"}]\n"
            "```"
        )
        json_in = gr.Textbox(label="Features (JSON)", lines=10, value="")
        json_btn = gr.Button("Predict")
        json_out = gr.Component()  # auto-detects number/table
        json_btn.click(predict_json, inputs=json_in, outputs=json_out)

    with gr.Tab("Batch via CSV"):
        gr.Markdown("Upload a CSV with the **same columns used in training**.")
        csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
        csv_btn = gr.Button("Predict CSV")
        csv_table = gr.Dataframe(label="Predictions")
        csv_file_out = gr.File(label="Download predictions.csv")
        csv_btn.click(predict_csv, inputs=csv_in, outputs=[csv_table, csv_file_out])

    with gr.Accordion("Model input schema (help)", open=False):
        gr.Markdown(feature_help_text())

# Hugging Face Spaces will pick up `demo` automatically
if __name__ == "__main__":
    demo.launch()
