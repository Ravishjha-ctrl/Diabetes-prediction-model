import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1{
    color:#0E76A8;
    text-align:center;
}
.card{
    padding:20px;
    border-radius:15px;
    background:white;
    box-shadow:0px 0px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏥 Hospital Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🩺 Prediction", "ℹ About"]
)

# =============================
# HOME PAGE
# =============================
if page == "🏠 Home":

    st.title("🩺 Diabetes Prediction System")

    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=900",
        use_container_width=True
    )

    st.markdown("""
### Welcome!

This application predicts whether a patient is diabetic using Machine Learning.

### Features

- 🤖 AI Prediction
- 📊 Risk Percentage
- 💡 Health Recommendation
- ⚡ Random Forest Model
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Random Forest")
    col2.metric("Features", "8")
    col3.metric("Prediction", "Instant")

# =============================
# PREDICTION PAGE
# =============================
elif page == "🩺 Prediction":

    st.title("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input("Pregnancies", 0, 20)

        glucose = st.number_input("Glucose", 0, 300)

        bloodpressure = st.number_input("Blood Pressure", 0, 200)

        skin = st.number_input("Skin Thickness", 0, 100)

    with col2:

        insulin = st.number_input("Insulin", 0, 900)

        bmi = st.number_input("BMI", 0.0, 70.0)

        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)

        age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):

        data = np.array([[

            pregnancies,
            glucose,
            bloodpressure,
            skin,
            insulin,
            bmi,
            dpf,
            age

        ]])

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(scaled)[0][1]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠ High Risk of Diabetes")

        else:

            st.success("✅ Low Risk of Diabetes")

        st.write("### Risk Percentage")

        st.progress(int(probability * 100))

        st.metric(
            "Risk Score",
            f"{probability*100:.2f}%"
        )

        st.subheader("Health Recommendation")

        if prediction == 1:

            st.warning("""
✔ Reduce Sugar Intake

✔ Exercise Daily

✔ Drink Plenty of Water

✔ Visit a Doctor

✔ Monitor Blood Glucose
""")

        else:

            st.success("""
✔ Maintain Healthy Diet

✔ Continue Exercise

✔ Annual Health Checkup

✔ Stay Hydrated
""")

# =============================
# ABOUT PAGE
# =============================
else:

    st.title("About")

    st.write("""
## Diabetes Prediction System

This project predicts diabetes using Machine Learning.

### Algorithm Used

- Random Forest Classifier

### Libraries

- Streamlit
- Scikit-Learn
- NumPy
- Pandas

### Developed By

7th Semester Machine Learning Project
""")
