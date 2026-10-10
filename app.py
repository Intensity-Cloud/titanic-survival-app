"""
Titanic Passenger Survival Predictor  -  Streamlit web app
AIC354 Machine Learning Fundamentals, Assignment 2 (Task 1)

What this file does
-------------------
1. Loads the trained SVC model (svc_trained_model.pkl, saved by the notebook).
2. Shows a form with four dropdowns (PClass, Gender, Sibling, Embarked).
3. Checks that every choice is valid, converts the choices to numbers with the
   SAME label-encoding codes used in the notebook, and shows the prediction.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------
# Settings you may want to edit
# --------------------------------------------------------------------------
BUILT_BY = "Qasim Saeed / FA24-BSE-149 (B)"      # shown in the footer - replace with your details
MODEL_FILE = "svc_trained_model.pkl"   # must sit in the SAME folder as this file
TEST_ACCURACY = "80%"                  # accuracy the notebook measured on its 20-passenger test set

# Paths are built from this file's own location, so the app works on any computer
# or cloud server without editing (no hard-coded C:\ or /home/... paths).
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / MODEL_FILE

# The four inputs. The ORDER matters: it must match the column order used when the
# model was trained (PClass, Gender, Sibling, Embarked).
FEATURES = {
    "PClass": {
        "label": "Passenger class",
        "options": ["First", "Second", "Third"],
        "help": "The class of ticket the passenger held.",
    },
    "Gender": {
        "label": "Gender",
        "options": ["Male", "Female"],
        "help": "The passenger's gender.",
    },
    "Sibling": {
        "label": "Siblings / spouses aboard",
        "options": ["Zero", "One", "Two", "Three"],
        "help": "How many siblings or spouses travelled with the passenger.",
    },
    "Embarked": {
        "label": "Port of embarkation",
        "options": ["Southampton", "Cherbourg", "Queenstown"],
        "help": "The port where the passenger boarded.",
    },
}
FEATURE_ORDER = list(FEATURES)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource
def build_encoders() -> dict[str, LabelEncoder]:
    """Recreate the notebook's LabelEncoders.

    LabelEncoder gives codes in alphabetical order of the categories it is fitted
    on, so fitting on the same fixed lists always gives the same codes as in the
    notebook. That is why no separate encoder file has to be shipped.
    """
    return {name: LabelEncoder().fit(spec["options"]) for name, spec in FEATURES.items()}


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    """Load the pickled model. Returns (model, list_of_version_warnings).

    A pickle saved with one scikit-learn version and opened with another raises
    InconsistentVersionWarning. We catch it here so the app can show a clear
    banner instead of silently risking wrong predictions.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
    notes = [str(w.message) for w in caught if issubclass(w.category, InconsistentVersionWarning)]
    return model, notes


def find_problems(choices: dict[str, str | None]) -> list[str]:
    """Return a list of friendly messages; an empty list means the input is valid."""
    problems = []
    for name, spec in FEATURES.items():
        value = choices.get(name)
        if value is None:
            problems.append(f"Please choose a value for **{spec['label']}**.")
        elif value not in spec["options"]:
            problems.append(f"'{value}' is not a valid choice for **{spec['label']}**.")
    return problems


def encode(choices: dict[str, str]) -> pd.DataFrame:
    """Turn the text choices into a one-row table of numeric codes (columns in training order)."""
    encoders = build_encoders()
    return pd.DataFrame(
        {name: [int(encoders[name].transform([choices[name]])[0])] for name in FEATURE_ORDER}
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Passenger Survival Predictor")
st.write(
    "Choose a passenger's details below. A trained **Support Vector Classifier** "
    "will predict whether that passenger would have survived the Titanic disaster."
)

# --- Load the model, and stop with a helpful message if that is not possible ---
try:
    model, version_notes = load_model()
except FileNotFoundError:
    st.error(
        f"The model file **{MODEL_FILE}** was not found. It must be in the same folder "
        "as `app.py` in your GitHub repository (check the file name and that it was uploaded)."
    )
    st.stop()
except Exception as error:  # broken/incompatible pickle
    st.error(
        "The model file exists but could not be loaded. The usual cause is a "
        "**scikit-learn version mismatch** between the notebook that saved the model and "
        "`requirements.txt`. Details below."
    )
    st.code(f"{type(error).__name__}: {error}")
    st.stop()

if getattr(model, "n_features_in_", len(FEATURE_ORDER)) != len(FEATURE_ORDER):
    st.error(
        f"This model expects {model.n_features_in_} inputs but the app collects "
        f"{len(FEATURE_ORDER)}. Upload the model that was trained for these four inputs."
    )
    st.stop()

if version_notes:
    st.warning(
        "⚠️ This model was saved with a different scikit-learn version than the one running "
        "this app, so predictions may be unreliable. Pin the matching version in "
        "`requirements.txt` (see README).\n\n" + version_notes[0]
    )

# --- Input form ---
with st.form("passenger_form"):
    left, right = st.columns(2)
    choices: dict[str, str | None] = {}
    for position, (name, spec) in enumerate(FEATURES.items()):
        column = left if position % 2 == 0 else right
        with column:
            choices[name] = st.selectbox(
                spec["label"],
                spec["options"],
                index=None,                    # nothing pre-selected, so the user must choose
                placeholder="Choose an option",
                help=spec["help"],
            )
    submitted = st.form_submit_button("Predict survival", type="primary", width="stretch")

# --- Validate, predict, show ---
if submitted:
    problems = find_problems(choices)
    if problems:
        for message in problems:
            st.warning(message)
    else:
        features = encode(choices)
        prediction = int(model.predict(features)[0])   # 1 = survived, 0 = did not survive

        st.divider()
        if prediction == 1:
            st.success("### ✅ SURVIVED")
        else:
            st.error("### ❌ NOT SURVIVED")

        with st.expander("See the encoded feature vector"):
            st.write("The model works with numbers, so each choice was converted to a code:")
            st.dataframe(features, hide_index=True)

# --- Sidebar and footer ---
with st.sidebar:
    st.header("About")
    st.write(
        "Model: Support Vector Classifier (scikit-learn).\n\n"
        "Data: a small, simplified 100-passenger Titanic sample.\n\n"
        f"Accuracy on its held-out test set: **{TEST_ACCURACY}** (only 20 passengers, so treat it as a rough guide)."
    )
    st.caption("Educational demo. Predictions reflect patterns in a tiny toy dataset, not a rigorous historical analysis.")

st.divider()
st.caption(f"AIC354 Machine Learning Fundamentals - Assignment 2  |  Built by {BUILT_BY}  |  scikit-learn {sklearn.__version__}")
