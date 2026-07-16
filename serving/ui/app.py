from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import gradio as gr
import pandas as pd
import requests


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def _get_json(path: str) -> dict[str, Any]:
    r = requests.get(f"{BACKEND_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_model_info() -> dict[str, Any]:
    return _get_json("/model")


def predict_single(
    age,
    sex,
    race_ethnicity,
    height_cm,
    weight_kg,
    bmi,
    pack_years,
    smoking_status,
    fev1_ml,
    fvc_ml,
) -> tuple[str, pd.DataFrame]:
    payload = {
        "age": age,
        "sex": sex,
        "race_ethnicity": race_ethnicity,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "pack_years": pack_years,
        "smoking_status": smoking_status,
        "fev1_ml": fev1_ml,
        "fvc_ml": fvc_ml,
    }

    res = _post_json("/predict", payload)

    copd_pred = res.get("copd_pred")
    copd_proba = res.get("copd_proba")
    gold = res.get("gold_stage_pred")

    summary = (
        f"COPD prediction: **{copd_pred}**  (P(COPD)={copd_proba:.3f})\n\n"
        f"GOLD stage prediction: **{gold}** (0 means no COPD)"
    )

    champ = res.get("champion", {})
    champ_df = pd.DataFrame(
        [{
            "champion_id": champ.get("champion_id"),
            "mlflow_run_id": champ.get("mlflow_run_id"),
            "registered_at": champ.get("registered_at"),
            "metric_name": champ.get("metric_name"),
            "metric_value": champ.get("metric_value"),
            "diagnosis_threshold": champ.get("diagnosis_threshold"),
        }]
    )

    return summary, champ_df


def predict_csv(file_obj) -> tuple[pd.DataFrame, str]:
    if file_obj is None:
        return pd.DataFrame(), "Please upload a CSV file."

    with open(file_obj, "rb") as fh:
        files = {"file": (os.path.basename(file_obj), fh, "text/csv")}
        r = requests.post(f"{BACKEND_URL}/predict_csv", files=files, timeout=120)
        r.raise_for_status()
        content = r.content

    df = pd.read_csv(io.BytesIO(content))

    # Write to a temp file so the user can download.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="copd_predictions_")
    tmp.write(content)
    tmp.flush()
    tmp.close()

    return df.head(25), tmp.name


CSS = """
#title-block {
  background: linear-gradient(90deg, #0f172a 0%, #1e293b 60%, #0b1220 100%);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 20px 22px;
  border-radius: 14px;
}
#title-block h1 { color: #e2e8f0; margin: 0; }
#title-block p { color: #cbd5e1; margin-top: 8px; }
.card {
  border: 1px solid rgba(15, 23, 42, 0.15);
  border-radius: 12px;
  padding: 14px;
  background: rgba(248,250,252,0.9);
}
"""


with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div id='title-block'>
          <h1>COPD Predictor</h1>
          <p>Binary COPD diagnosis + GOLD stage (if COPD). Powered by the latest champion model from the offline Airflow/MLflow pipeline.</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Welcome"):
            gr.Markdown(
                """
                ### How it works
                - The training pipeline selects a **champion** model and registers it in PostgreSQL.
                - This app always uses the **currently active champion**.

                ### What you can do
                - **Single prediction**: fill the form and predict COPD + GOLD stage.
                - **Batch prediction**: upload a CSV with columns like `age`, `sex`, `race_ethnicity`, `height_cm`, `weight_kg`, `bmi`, `pack_years`, `smoking_status`, optionally `fev1_ml`, `fvc_ml`.
                """
            )
            info_btn = gr.Button("Show current champion")
            info_out = gr.JSON(label="Champion metadata")
            info_btn.click(fetch_model_info, outputs=info_out)

        with gr.Tab("Predict"):
            with gr.Tabs():
                with gr.Tab("Single"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            age = gr.Number(label="Age", value=None)
                            sex = gr.Dropdown(
                                label="Sex (numeric encoding)",
                                choices=[0, 1],
                                value=None,
                                info="Use the same encoding as the dataset (example: 0/1).",
                            )
                            race_ethnicity = gr.Number(label="Race/ethnicity (numeric encoding)", value=None)
                            smoking_status = gr.Dropdown(
                                label="Smoking status",
                                choices=["never", "former", "current", "unknown"],
                                value=None,
                            )
                            pack_years = gr.Number(label="Pack-years", value=None)
                        with gr.Column(scale=1):
                            height_cm = gr.Number(label="Height (cm)", value=None)
                            weight_kg = gr.Number(label="Weight (kg)", value=None)
                            bmi = gr.Number(label="BMI", value=None)
                            fev1_ml = gr.Number(label="FEV1 (ml) [optional]", value=None)
                            fvc_ml = gr.Number(label="FVC (ml) [optional]", value=None)

                    predict_btn = gr.Button("Predict")
                    summary = gr.Markdown()
                    champ_df = gr.Dataframe(label="Champion", interactive=False)

                    predict_btn.click(
                        predict_single,
                        inputs=[
                            age,
                            sex,
                            race_ethnicity,
                            height_cm,
                            weight_kg,
                            bmi,
                            pack_years,
                            smoking_status,
                            fev1_ml,
                            fvc_ml,
                        ],
                        outputs=[summary, champ_df],
                    )

                with gr.Tab("Batch (CSV)"):
                    gr.Markdown(
                        "Upload a CSV with patient rows. The app will return a new CSV with prediction columns appended."
                    )
                    csv_in = gr.File(label="CSV file")
                    run_btn = gr.Button("Run batch prediction")
                    preview = gr.Dataframe(label="Preview (first 25 rows)")
                    csv_out = gr.File(label="Download predictions CSV")
                    status = gr.Markdown()

                    run_btn.click(predict_csv, inputs=[csv_in], outputs=[preview, csv_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
