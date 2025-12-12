import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
import warnings

# --- 0. SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Date Decision Simulator", layout="centered")
st.title("❤️ Will I Say 'Yes'?")
st.markdown("Based on your ratings and preferences, this tool predicts if you will want to see this person again.")

# --- 2. LOAD RESOURCES ---
@st.cache_data
def load_resources():
    model = joblib.load('dating_model.joblib')
    baseline = pd.read_csv('baseline.csv')
    return model, baseline

try:
    rf_model, baseline_df = load_resources()
except FileNotFoundError:
    st.error("Error: Missing files! Make sure 'dating_model.joblib' and 'baseline.csv' are in this folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. SIDEBAR: USER INPUTS ---

st.sidebar.header("1. Rate the Partner")
st.sidebar.caption("How do you perceive them? (1-10)")

attr_p = st.sidebar.slider("Attractive", 1, 10, 5)
sinc_p = st.sidebar.slider("Sincere", 1, 10, 5)
intel_p = st.sidebar.slider("Intelligent", 1, 10, 5)
fun_p = st.sidebar.slider("Funny", 1, 10, 5)
amb_p = st.sidebar.slider("Ambitious", 1, 10, 5)

st.sidebar.divider()

st.sidebar.header("2. Your Preferences")
st.sidebar.caption("How important is this attribute to you? (1-10)")

attr_imp = st.sidebar.slider("Importance: Looks", 1, 10, 5)
sinc_imp = st.sidebar.slider("Importance: Sincerity", 1, 10, 5)
intel_imp = st.sidebar.slider("Importance: Intelligence", 1, 10, 5)
fun_imp = st.sidebar.slider("Importance: Humor", 1, 10, 5)
amb_imp = st.sidebar.slider("Importance: Ambition", 1, 10, 5)
share_imp = st.sidebar.slider("Importance: Shared Interests", 1, 10, 5)

st.sidebar.divider()

# --- UPDATED SECTION: Clean & Centered Labels ---
st.sidebar.header("3. Interest Correlation")

# We use 3 columns to align the text perfectly above the slider ends and middle
col_a, col_b, col_c = st.sidebar.columns([1, 1, 1])

# Using HTML to force 'gray' color and 'center' alignment
with col_a:
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>-1<br>Opposite</div>", unsafe_allow_html=True)
with col_b:
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>0<br>None</div>", unsafe_allow_html=True)
with col_c:
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>1<br>Match</div>", unsafe_allow_html=True)

int_corr = st.sidebar.slider("Interest Correlation", -1.0, 1.0, 0.00, label_visibility="collapsed")


# --- 4. PREPARE DATA ---
input_df = baseline_df.copy()

# Overwrite with User Inputs
input_df['attractive_partner'] = attr_p
input_df['sincere_partner'] = sinc_p
input_df['intelligence_partner'] = intel_p
input_df['funny_partner'] = fun_p
input_df['ambition_partner'] = amb_p

input_df['attractive_important'] = attr_imp
input_df['sincere_important'] = sinc_imp
input_df['intellicence_important'] = intel_imp
input_df['funny_important'] = fun_imp
input_df['ambtition_important'] = amb_imp
input_df['shared_interests_important'] = share_imp

input_df['interests_correlate'] = int_corr

# --- 5. PREDICTION ---
st.subheader("The Verdict")

prob = rf_model.predict_proba(input_df)[0][1]

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Likelihood to Accept", f"{prob:.1%}")

with col2:
    if prob > 0.5:
        st.success("Verdict: **YES** 🥂")
        st.write("You would likely want to see this person again.")
        st.balloons()
    else:
        st.error("Verdict: **NO** 🙅")
        st.write("You would likely reject this person.")

# --- 6. SENSITIVITY ANALYSIS ---
st.divider()
st.subheader("Sensitivity Analysis")
st.markdown("How much would your decision change if you rated the partner **+1 point higher** on a specific trait?")

features_to_bump = [
    'attractive_partner', 'sincere_partner', 'intelligence_partner', 
    'funny_partner', 'ambition_partner'
]

label_map = {
    'attractive_partner': "Partner's Attractiveness",
    'sincere_partner': "Partner's Sincerity",
    'intelligence_partner': "Partner's Intelligence",
    'funny_partner': "Partner's Humor",
    'ambition_partner': "Partner's Ambition"
}

data_for_chart = []

for feat in features_to_bump:
    temp_df = input_df.copy()
    current_val = temp_df[feat].values[0]
    temp_df[feat] = min(current_val + 1, 10)
    
    new_prob = rf_model.predict_proba(temp_df)[0][1]
    
    data_for_chart.append({
        "Trait": label_map[feat],
        "Probability": new_prob,
        "Label": f"{new_prob:.1%}" 
    })

chart_df = pd.DataFrame(data_for_chart)

bars = alt.Chart(chart_df).mark_bar(color='#4c78a8').encode(
    x=alt.X('Trait', sort='-y', axis=alt.Axis(labelAngle=-45)), 
    y=alt.Y('Probability', title='Total Likelihood to Say Yes', axis=alt.Axis(format='%'))
)

text = bars.mark_text(
    align='center',
    baseline='bottom',
    dy=-5, 
    fontSize=12,
    fontWeight='bold'
).encode(
    text='Label'
)

rule = alt.Chart(pd.DataFrame({'y': [prob]})).mark_rule(color='red', strokeDash=[5, 5]).encode(
    y='y'
)

final_chart = (bars + text + rule).properties(height=400)

st.altair_chart(final_chart, width="stretch")

st.caption("The red dashed line represents your CURRENT likelihood. The bars show your NEW likelihood if that trait improves by +1.")