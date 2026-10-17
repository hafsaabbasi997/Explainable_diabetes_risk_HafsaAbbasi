"""
app.py - Explainable Diabetes Risk Assistant | 2026 Edition (Streamlit)

Works with the output of train.py:
    assets/model.json            best XGBoost model (native format) - loaded FIRST
    assets/model.pkl             same model as a pickle - fallback only
    assets/imputer_medians.json  training medians (default slider values)
    assets/metrics.json          CV / test metrics (shown in the header)

Run locally :  streamlit run app.py
Deploy      :  push app.py, requirements.txt and assets/ to GitHub, then pick app.py
               on share.streamlit.io

NOTE: research / educational prototype. NOT a medical device.
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
from groq import Groq
groq_api=st.secrets["GROQ_API_KEY"]
client=Groq(api_key=groq_api)
from xgboost import XGBClassifier

ASSETS = Path(__file__).parent / "assets"
HIGH_RISK_THRESHOLD = 0.50  # probability at/above which the label is "High risk"

# Must match train.py exactly (same names, same ORDER) - all 8 features
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

st.set_page_config(page_title="Explainable Diabetes Risk", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- #
# Load model + explainer once per server process
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = None

    # 1) model.json first (XGBoost native format, works across XGBoost versions)
    json_path = ASSETS / "model.json"
    if json_path.exists():
        try:
            model = XGBClassifier()
            model.load_model(str(json_path))
        except Exception:
            model = None

    # 2) fallback: model.pkl
    pkl_path = ASSETS / "model.pkl"
    if model is None and pkl_path.exists():
        try:
            obj = joblib.load(pkl_path)
            model = obj.named_steps["xgb"] if hasattr(obj, "named_steps") else obj
        except Exception:
            model = None

    if model is None:
        return None, None, {}, {}

    # Explainable AI: TreeExplainer on the XGBoost model
    explainer = shap.TreeExplainer(model)

    metrics_path = ASSETS / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    med_path = ASSETS / "imputer_medians.json"
    medians = json.loads(med_path.read_text()) if med_path.exists() else {}
    return model, explainer, metrics, medians


model, explainer, metrics, medians = load_artifacts()
if model is None:
    st.error(
        "Could not load the model. Run `python train.py` and keep the `assets/` folder "
        "(with model.json) next to app.py."
    )
    st.stop()

# Clinical context for each feature (used to write the reasoning text)
INFO = {
    "Glucose": ("plasma glucose", lambda v: v >= 140,
                "≥140 mg/dL after a 2-hour glucose tolerance test suggests impaired glucose handling; ≥200 is in the diabetic range."),
    "BMI": ("BMI", lambda v: v >= 30,
            "BMI ≥30 is classed as obesity and is linked to insulin resistance."),
    "Age": ("age", lambda v: v >= 45,
            "Type 2 diabetes risk rises with age; screening is commonly advised from the mid-30s to 40s."),
    "BloodPressure": ("diastolic blood pressure", lambda v: v >= 90,
                      "Diastolic ≥90 mm Hg is hypertensive and often clusters with metabolic syndrome."),
    "Insulin": ("2-hour serum insulin", lambda v: v >= 166,
                "Markedly raised insulin can signal insulin resistance (the pancreas compensating)."),
    "DiabetesPedigreeFunction": ("family-history score", lambda v: v >= 0.8,
                                 "Higher values indicate a stronger genetic / family-history contribution."),
    "Pregnancies": ("number of pregnancies", lambda v: v >= 6,
                    "A higher pregnancy count is associated with higher type 2 diabetes risk in this cohort."),
    "SkinThickness": ("triceps skin-fold thickness", lambda v: v >= 35,
                      "Skin-fold thickness is a proxy for body fat."),
}
UNIT_LABELS = {
    "Pregnancies": "Pregnancies", "SkinThickness": "Skin thickness",
    "Glucose": "Glucose", "BMI": "BMI", "Age": "Age",
    "BloodPressure": "Blood pressure", "Insulin": "Insulin",
    "DiabetesPedigreeFunction": "Pedigree",
}


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def predict(user_values: dict):
    """All 8 features come from the sliders (no imputation needed).
    Returns: risk probability, shap.Explanation for this patient."""
    X = np.array([[user_values[f] for f in FEATURES]], dtype=float)
    risk = float(model.predict_proba(X)[0, 1])

    sv = explainer(X)
    explanation = shap.Explanation(
        values=sv.values[0],
        base_values=float(np.ravel(sv.base_values)[0]),
        data=X[0],
        feature_names=FEATURES,
    )
    return risk, explanation


def clinical_reasoning(user_values: dict, explanation, top_k: int = 3) -> list:
    """Turn the largest SHAP contributions into plain-language statements."""
    lines = []
    order = np.argsort(-np.abs(explanation.values))
    for i in order:
        name = FEATURES[i]
        label, is_high, reference = INFO[name]
        v, s = user_values[name], float(explanation.values[i])
        direction = "raises" if s > 0 else "lowers"
        side = "in the concern range" if is_high(v) else "outside the usual concern range"
        lines.append(
            f"**{label.capitalize()} = {v:g}** {direction} the predicted risk "
            f"(SHAP {s:+.2f} log-odds) and is {side}. _{reference}_"
        )
        if len(lines) == top_k:
            break
    return lines


def waterfall_png(explanation) -> bytes:
    """Render the SHAP waterfall to a PNG without cropping.
    figsize=(10, 6) is set AFTER shap draws (shap resizes the figure itself) and the image
    is saved with bbox_inches='tight' + padding, so labels and axes are fully visible."""
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Sidebar: 8 inputs (defaults = training medians when available)
# --------------------------------------------------------------------------- #
def default(name: str, fallback: float) -> float:
    return float(medians.get(name, fallback))


with st.sidebar:
    st.header("Patient inputs")
    user_values = {
        "Pregnancies": st.slider("Pregnancies (count)", 0, 17, int(round(default("Pregnancies", 3)))),
        "Glucose": st.slider("Glucose (mg/dL, 2-h OGTT)", 40, 250, int(round(default("Glucose", 117)))),
        "BloodPressure": st.slider("Diastolic blood pressure (mm Hg)", 40, 140,
                                   int(round(default("BloodPressure", 72)))),
        "SkinThickness": st.slider("Skin thickness (mm)", 5, 100, int(round(default("SkinThickness", 29)))),
        "Insulin": st.slider("2-h serum insulin (µU/mL)", 10, 850, int(round(default("Insulin", 126)))),
        "BMI": st.slider("BMI (kg/m²)", 15.0, 60.0, float(round(default("BMI", 32.4), 1)), 0.1),
        "DiabetesPedigreeFunction": st.slider("Diabetes pedigree function", 0.05, 2.50,
                                              float(round(default("DiabetesPedigreeFunction", 0.37), 2)), 0.01),
        "Age": st.slider("Age (years)", 18, 90, int(round(default("Age", 29)))),
    }

risk, explanation = predict(user_values)
is_high = risk >= HIGH_RISK_THRESHOLD

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🩺 Explainable Diabetes Risk Assistant")
st.caption("XGBoost · SMOTE class balancing · SHAP explanations - research prototype, not a diagnostic tool.")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted diabetes risk", f"{risk:.0%}")
c2.metric("Result", "HIGH RISK" if is_high else "LOW RISK")
if "test_roc_auc" in metrics:
    c3.metric("Model test ROC-AUC", f"{metrics['test_roc_auc']:.2f}")
st.progress(float(np.clip(risk, 0.0, 1.0)))

if is_high:
    st.error("High risk - this profile resembles patients who had diabetes in the training data. Clinical follow-up is advised.")
else:
    st.success("Low risk - this profile resembles patients without diabetes in the training data.")

# ---- Explainable AI --------------------------------------------------------- #
st.subheader("Why this prediction? (SHAP waterfall)")
# Full-width image: nothing is cropped or squeezed into a narrow column
st.image(waterfall_png(explanation), width='stretch')
st.caption(
    "Red bars push the risk up, blue bars push it down (log-odds). Bars start at the average "
    "patient, E[f(x)], and add up to this patient's score, f(x)."
)

st.markdown("#### Clinical reasoning")
for line in clinical_reasoning(user_values, explanation):
    st.markdown(f"- {line}")

# ---- Chat with your report (LLM-ready placeholder) ------------------------- #
st.divider()
st.subheader("💬 Chat with your report (LLM-ready)")
st.caption(
    "Placeholder for a 2026-style grounded assistant. The JSON report below is exactly the "
    "context an LLM would receive."
)

report = {
    "risk_probability": round(risk, 3),
    "label": "high" if is_high else "low",
    "inputs": {k: float(v) for k, v in user_values.items()},
    "shap_contributions_logodds": {f: round(float(v), 3) for f, v in zip(FEATURES, explanation.values)},
    "baseline_logodds": round(float(explanation.base_values), 3),
}


def build_llm_prompt(question: str) -> str:
    """Grounded prompt an LLM would receive (plug in any provider)."""
    return (
        "You are a careful clinical-education assistant. Answer ONLY from the report. "
        "Do not diagnose; advise seeing a clinician.\n"
        f"REPORT: {report}\nQUESTION: {question}"
    )

def answer_question(question: str) -> str:
    try:
        prompt = build_llm_prompt(question)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {e}"


if "chat" not in st.session_state:
    st.session_state.chat = []
for role, text in st.session_state.chat:
    st.chat_message(role).write(text)

if prompt := st.chat_input("Ask about your report, e.g. 'Which factor matters most?'"):
    st.session_state.chat.append(("user", prompt))
    st.chat_message("user").write(prompt)
    reply = answer_question(prompt)
    st.session_state.chat.append(("assistant", reply))
    st.chat_message("assistant").write(reply)

with st.expander("Show LLM-ready structured report (JSON)"):
    st.json(report, width='stretch')

# ---- Footer ----------------------------------------------------------------- #
st.divider()
st.caption("Note: Trained on Pima Indian Women dataset only.")
st.caption(
    "⚠️ Educational prototype, not validated for clinical use. "
    "Trained on 768 historical records of women aged 21+."
)