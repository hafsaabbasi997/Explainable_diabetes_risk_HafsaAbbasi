"""
app.py - Explainable Diabetes Risk Assistant | 2026 Edition (Streamlit)
Fixed: LLM now answers what-if + actually recalculates risk
"""
import io
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
from xgboost import XGBClassifier

# --- Groq client safe load ---
try:
    from groq import Groq
    groq_api = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api)
except Exception:
    client = None

ASSETS = Path(__file__).parent / "assets"
HIGH_RISK_THRESHOLD = 0.50

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

st.set_page_config(page_title="Explainable Diabetes Risk", page_icon="🩺", layout="wide")

@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = None
    json_path = ASSETS / "model.json"
    if json_path.exists():
        try:
            model = XGBClassifier()
            model.load_model(str(json_path))
        except Exception:
            model = None
    pkl_path = ASSETS / "model.pkl"
    if model is None and pkl_path.exists():
        try:
            obj = joblib.load(pkl_path)
            model = obj.named_steps["xgb"] if hasattr(obj, "named_steps") else obj
        except Exception:
            model = None
    if model is None:
        return None, None, {}, {}
    explainer = shap.TreeExplainer(model)
    metrics_path = ASSETS / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    med_path = ASSETS / "imputer_medians.json"
    medians = json.loads(med_path.read_text()) if med_path.exists() else {}
    return model, explainer, metrics, medians

model, explainer, metrics, medians = load_artifacts()
if model is None:
    st.error("Could not load model. Run train.py and keep assets/ next to app.py.")
    st.stop()

INFO = {
    "Glucose": ("plasma glucose", lambda v: v >= 140, "≥140 mg/dL suggests impaired glucose handling; ≥200 is diabetic range."),
    "BMI": ("BMI", lambda v: v >= 30, "BMI ≥30 is obesity linked to insulin resistance."),
    "Age": ("age", lambda v: v >= 45, "Risk rises with age."),
    "BloodPressure": ("diastolic blood pressure", lambda v: v >= 90, "Diastolic ≥90 is hypertensive."),
    "Insulin": ("2-hour serum insulin", lambda v: v >= 166, "High insulin signals insulin resistance."),
    "DiabetesPedigreeFunction": ("family-history score", lambda v: v >= 0.8, "Higher = stronger family history."),
    "Pregnancies": ("number of pregnancies", lambda v: v >= 6, "Higher count linked to higher risk in this cohort."),
    "SkinThickness": ("triceps skin-fold", lambda v: v >= 35, "Proxy for body fat."),
}

def default(name: str, fallback: float) -> float:
    return float(medians.get(name, fallback))

with st.sidebar:
    st.header("Patient inputs")
    user_values = {
        "Pregnancies": st.slider("Pregnancies (count)", 0, 17, int(round(default("Pregnancies", 3)))),
        "Glucose": st.slider("Glucose (mg/dL, 2-h OGTT)", 40, 250, int(round(default("Glucose", 117)))),
        "BloodPressure": st.slider("Diastolic blood pressure (mm Hg)", 40, 140, int(round(default("BloodPressure", 72)))),
        "SkinThickness": st.slider("Skin thickness (mm)", 5, 100, int(round(default("SkinThickness", 29)))),
        "Insulin": st.slider("2-h serum insulin (µU/mL)", 10, 850, int(round(default("Insulin", 126)))),
        "BMI": st.slider("BMI (kg/m²)", 15.0, 60.0, float(round(default("BMI", 32.4), 1)), 0.1),
        "DiabetesPedigreeFunction": st.slider("Diabetes pedigree function", 0.05, 2.50, float(round(default("DiabetesPedigreeFunction", 0.37), 2)), 0.01),
        "Age": st.slider("Age (years)", 18, 90, int(round(default("Age", 29)))),
    }

def predict(user_vals: dict):
    X = np.array([[user_vals[f] for f in FEATURES]], dtype=float)
    risk = float(model.predict_proba(X)[0, 1])
    sv = explainer(X)
    explanation = shap.Explanation(values=sv.values[0], base_values=float(np.ravel(sv.base_values)[0]), data=X[0], feature_names=FEATURES)
    return risk, explanation

def clinical_reasoning(user_vals: dict, explanation, top_k: int = 3) -> list:
    lines = []
    order = np.argsort(-np.abs(explanation.values))
    for i in order:
        name = FEATURES[i]
        label, is_high, reference = INFO[name]
        v, s = user_vals[name], float(explanation.values[i])
        direction = "raises" if s > 0 else "lowers"
        side = "in the concern range" if is_high(v) else "outside concern range"
        lines.append(f"**{label.capitalize()} = {v:g}** {direction} risk (SHAP {s:+.2f}) and is {side}. _{reference}_")
        if len(lines) == top_k: break
    return lines

def waterfall_png(explanation) -> bytes:
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return buf.getvalue()

risk, explanation = predict(user_values)
is_high = risk >= HIGH_RISK_THRESHOLD

st.title("🩺 Explainable Diabetes Risk Assistant")
st.caption("XGBoost · SMOTE · SHAP - research prototype, not a diagnostic tool.")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted diabetes risk", f"{risk:.0%}")
c2.metric("Result", "HIGH RISK" if is_high else "LOW RISK")
if "test_roc_auc" in metrics:
    c3.metric("Model test ROC-AUC", f"{metrics['test_roc_auc']:.2f}")
st.progress(float(np.clip(risk, 0.0, 1.0)))
if is_high:
    st.error("High risk - resembles patients who had diabetes. Clinical follow-up advised.")
else:
    st.success("Low risk - resembles patients without diabetes.")

st.subheader("Why this prediction? (SHAP waterfall)")
st.image(waterfall_png(explanation), width='stretch')
st.caption("Red pushes risk up, blue pushes down. Starts at average patient E[f(x)] to f(x).")
st.markdown("#### Clinical reasoning")
for line in clinical_reasoning(user_values, explanation):
    st.markdown(f"- {line}")

# ---- SMART CHAT ----
st.divider()
st.subheader("💬 Chat with your report (LLM-ready)")
report = {
    "risk_probability": round(risk, 3),
    "label": "high" if is_high else "low",
    "inputs": {k: float(v) for k, v in user_values.items()},
    "shap_contributions_logodds": {f: round(float(v), 3) for f, v in zip(FEATURES, explanation.values)},
    "baseline_logodds": round(float(explanation.base_values), 3),
}

def try_what_if(question: str):
    """If user says 'what if glucose 100', actually recalculate risk"""
    q = question.lower()
    new_vals = user_values.copy()
    changed = {}
    # regex for each feature
    patterns = {
        "Glucose": r"glucose.*?(\d+\.?\d*)",
        "BMI": r"bmi.*?(\d+\.?\d*)",
        "Age": r"age.*?(\d+\.?\d*)",
        "BloodPressure": r"(blood pressure|bp).*?(\d+\.?\d*)",
        "Insulin": r"insulin.*?(\d+\.?\d*)",
    }
    for feat, pat in patterns.items():
        m = re.search(pat, q)
        if m:
            try:
                val = float(m.groups()[-1])
                if feat == "Glucose" and 40 <= val <= 250:
                    new_vals[feat] = val
                    changed[feat] = val
                elif feat == "BMI" and 10 <= val <= 70:
                    new_vals[feat] = val
                    changed[feat] = val
                elif feat in new_vals:
                    new_vals[feat] = val
                    changed[feat] = val
            except: pass
    if changed:
        new_risk, new_exp = predict(new_vals)
        return changed, new_risk, new_exp
    return None, None, None

def build_llm_prompt(question: str, what_if_info: str = "") -> str:
    return (
        "You are a helpful clinical-education assistant for an explainable diabetes risk tool.\n"
        "You HAVE report + you CAN reason about what-if.\n"
        "RULES:\n"
        "1. Explain current risk using SHAP.\n"
        "2. If WHAT-IF RESULT is provided, use it to tell exact change in risk.\n"
        "3. If no exact result, you can still say generally lower glucose/BMI usually lowers risk.\n"
        "4. Never claim to be a doctor. Always add: 'Educational only, consult a clinician.'\n"
        f"REPORT: {report}\n"
        f"{what_if_info}\n"
        f"QUESTION: {question}"
    )

def answer_question(question: str) -> str:
    if client is None:
        return "⚠️ GROQ_API_KEY missing. Add it in Streamlit secrets, then reboot app."
    changed, new_risk, new_exp = try_what_if(question)
    what_if_info = ""
    if changed:
        what_if_info = f"WHAT-IF RESULT: User changed {changed} -> New predicted risk = {new_risk:.1%} (was {risk:.1%}). New SHAP: {dict(zip(FEATURES, [round(float(v),2) for v in new_exp.values]))}"

    prompt = build_llm_prompt(question, what_if_info)
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {e}"

if "chat" not in st.session_state:
    st.session_state.chat = []
for role, text in st.session_state.chat:
    st.chat_message(role).write(text)

if prompt := st.chat_input("Ask e.g., 'What if glucose was 100?' or 'Which factor matters most?'"):
    st.session_state.chat.append(("user", prompt))
    st.chat_message("user").write(prompt)
    reply = answer_question(prompt)
    st.session_state.chat.append(("assistant", reply))
    st.chat_message("assistant").write(reply)

with st.expander("Show LLM-ready structured report (JSON)"):
    st.json(report, width='stretch')

st.divider()
st.caption("⚠️ Educational prototype, not validated for clinical use. Trained on 768 records.")
