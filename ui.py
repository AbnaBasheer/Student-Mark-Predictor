

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Student Predictor", layout="centered")
# Load dataset
data = pd.read_csv("student_data.csv")

X = data[['study_hours', 'attendance', 'previous_marks', 'sleep_hours']]
y = data['final_marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🎓 Student Performance Predictor")

study_hours = st.slider("Study Hours", 0, 10)
attendance = st.slider("Attendance (%)", 0, 100)
previous_marks = st.slider("Previous Marks", 0, 100)
sleep_hours = st.slider("Sleep Hours", 0, 10)
if st.button("Predict"):
    result = model.predict([[study_hours, attendance, previous_marks, sleep_hours]])
    st.success(f"Predicted Marks: {round(result[0], 2)}")
    predicted =round(result[0], 2)

    # Smart suggestion
    if predicted < 60:
        st.warning("⚠️ You need improvement! Try increasing study hours.")
    elif predicted < 80:
        st.info("👍 Good, but you can score higher with more consistency.")
    else:
        st.success("🔥 Excellent performance expected!")
import matplotlib.pyplot as plt

if st.button("Show Graph"):
    plt.scatter(y, model.predict(X))
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.title("Model Performance")
    st.pyplot(plt)