import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("sleep_model.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Sleep Quality Predictor",
    page_icon="💤",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7ff;
}
h1 {
    color: #4B6CB7;
    text-align: center;
}
.result-good {
    background-color: #d4f8e8;
    padding: 15px;
    border-radius: 10px;
}
.result-average {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 10px;
}
.result-poor {
    background-color: #f8d7da;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("😴 Sleep Quality Predictor")
st.caption("A smart way to monitor and improve your sleep")

st.divider()

# Input Section
st.subheader("📝 Enter Your Daily Details")

sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0)
physical_activity = st.slider("Physical Activity (minutes)", 0, 120, 30)
stress_level = st.slider("Stress Level (0–10)", 0, 10, 5)
screen_time = st.slider("Screen Time Before Bed (minutes)", 0, 180, 60)

if st.button("🔍 Predict Sleep Quality"):
    features = np.array([[sleep_duration, physical_activity, stress_level, screen_time]])
    prediction = model.predict(features)[0]

    st.subheader("📊 Predicted Sleep Quality")

    if prediction == "Good":
        st.markdown(f"<div class='result-good'><b>{prediction}</b></div>", unsafe_allow_html=True)
        st.success("✅ Great sleep habits! Keep it up.")
    elif prediction == "Average":
        st.markdown(f"<div class='result-average'><b>{prediction}</b></div>", unsafe_allow_html=True)
        st.warning("⚠️ Sleep is okay, but improvements are possible.")
    else:
        st.markdown(f"<div class='result-poor'><b>{prediction}</b></div>", unsafe_allow_html=True)
        st.error("❌ Poor sleep detected.")

    # Suggestions
    st.subheader("💡 Suggestions")
    if stress_level > 6:
        st.write("• Reduce stress using meditation or breathing exercises.")
    if sleep_duration < 7:
        st.write("• Aim for 7–8 hours of sleep.")
    if physical_activity < 30:
        st.write("• Increase daily physical activity.")
    if screen_time > 90:
        st.write("• Reduce screen time before bed.")
    st.write("• Consider consulting a healthcare professional.")

st.divider()
st.caption("Sleep Quality Predictor | Machine Learning Project")
