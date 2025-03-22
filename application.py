import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Function to load models safely
def load_model(filepath):
    try:
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            st.error(f"⚠️ Error: {filepath} not found. Please check the file path.")
            return None
    except Exception as e:
        st.error(f"⚠️ Model loading error: {e}")
        return None

# Load Models Safely
models = {
    "Diabetes": load_model("SAV_File\diabetes_model.sav"),
    "Heart Disease": load_model("SAV_File\heart_disease_model.sav"),
    "Lung Disease": load_model("SAV_File\Lungs_model.sav"),
    "Parkinson's Disease": load_model("SAV_File\parkinson.sav"),
    "Preprocessed Lungs Disease": load_model("SAV_File\prepocessed_lungs_model.sav")
}


# Hide Streamlit Default UI Elements
st.markdown("""
    <style>
        #MainMenu, header, footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Default Light Theme */
        @media (prefers-color-scheme: light) {
            div[data-baseweb="select"] {
                border: 3px solid #FDFCDC !important; 
                border-radius: 6px !important;  
                background-color: #f4f4f4 !important;
                transition: all 0.3s ease-in-out;
            }
            div[data-baseweb="select"]:hover {
                border-color: #FFC107 !important;
                background-color: #fff9c4 !important;
            }
            div[data-baseweb="select"] * {
                color: black !important;  /* Dark text for light mode */
                font-weight: bold !important;
            }
        }

        /* Dark Theme */
        @media (prefers-color-scheme: dark) {
            div[data-baseweb="select"] {
                border: 3px solid #333 !important; 
                border-radius: 6px !important;  
                background-color: #222 !important;
                transition: all 0.3s ease-in-out;
            }
            div[data-baseweb="select"]:hover {
                border-color: #FFC107 !important;
                background-color: #444 !important;
            }
            div[data-baseweb="select"] * {
                color: white !important;  /* Light text for dark mode */
                font-weight: bold !important;
            }
        }

        [data-testid="stSidebar"] {
            background-color: #1976D2 !important;
            padding: 20px !important;
            border-radius: 0px 2px 2px 0px;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
            font-weight: 600 !important;
        }

        .highlight-red {
            color: red !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("🏥 AI Medical Diagnosis")
option = st.sidebar.selectbox("Select Diagnosis Type", [
    "Home", "Diabetes", "Heart Disease","Preprocessed Lungs Disease","Lung Disease", "Parkinson's Disease"
])

# Home Page
if option == "Home":
    st.markdown("""
        <div style="text-align: center; font-family: Arial, sans-serif;">
            <h1 style="color: #1976D2;">⚕️ AI-Powered Medical Diagnosis</h1>
        </div>
    """, unsafe_allow_html=True)

    # Display Image
    image_url = "https://www.just-medical.ch/hubfs/AI-diagnostic_855591230.jpeg"
    st.image(image_url, caption="AI in Healthcare", use_container_width=True)

    st.write("""
        ## 🏥 Why Your Health Matters  
        - **More Energy** ⚡: Stay productive all day.  
        - **Stronger Immunity** 🛡️: Reduce illness risks.  
        - **Better Mental Health** 😊: Manage stress effectively.  
        - **Longevity** 🎉: Live a healthier, longer life.  
        
        ### 🔹 Essential Health Tips  
        - 🥗 **Eat Healthy**: Nutrient-rich foods fuel your body.  
        - 🏃‍♂️ **Stay Active**: Exercise boosts physical & mental health.  
        - 💧 **Hydrate Well**: Drink plenty of water daily.  
        - 😴 **Prioritize Sleep**: Quality sleep is crucial.  
        - 🧘‍♀️ **Manage Stress**: Meditation and mindfulness help.  
        - 🏥 **Routine Checkups**: Early detection saves lives.  
    """)

    st.markdown("""
        <div style="background-color:#FFC107; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold;">
            ⚠️ AI-generated insights. Always consult a medical professional!
        </div>
    """, unsafe_allow_html=True)

# Prediction Function
def predict_disease(model, input_data, disease_name, positive_label, negative_label):
    if model:
        result = model.predict(input_data)
        st.write(f"Result: {positive_label if result[0] == 1 else negative_label}")
    else:
        st.error(f"⚠️ {disease_name} model is missing. Please check the model files.")

def predict_diabetes():
    st.subheader("Diabetes Diagnosis")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=1000)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    age = st.number_input("Age", min_value=0, max_value=120)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, bmi, age, diabetes_pedigree_function, skin_thickness, insulin]])
        predict_disease(models["Diabetes"], input_data, "Diabetes", "Diabetic", "Non-Diabetic")

def predict_heart_disease():
    st.subheader("Heart Disease Prediction")
    
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=250)
    chol = st.number_input("Cholesterol Level", min_value=0, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, step=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, step=1)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", min_value=0, max_value=4, step=1)
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, step=1)
    
    # Convert categorical variables to numeric
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], dtype=float)
        predict_disease(models["Heart Disease"], input_data, "Heart Disease", "Positive", "Negative")

def predict_lung_disease():
    st.subheader("Lung Disease Prediction")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    alcohol_consuming = st.selectbox("Alcohol Consuming", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    
    gender = 1 if gender == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0
    yellow_fingers = 1 if yellow_fingers == "Yes" else 0
    anxiety = 1 if anxiety == "Yes" else 0
    peer_pressure = 1 if peer_pressure == "Yes" else 0
    chronic_disease = 1 if chronic_disease == "Yes" else 0
    fatigue = 1 if fatigue == "Yes" else 0
    allergy = 1 if allergy == "Yes" else 0
    wheezing = 1 if wheezing == "Yes" else 0
    alcohol_consuming = 1 if alcohol_consuming == "Yes" else 0
    coughing = 1 if coughing == "Yes" else 0
    shortness_of_breath = 1 if shortness_of_breath == "Yes" else 0
    swallowing_difficulty = 1 if swallowing_difficulty == "Yes" else 0
    chest_pain = 1 if chest_pain == "Yes" else 0
    
    if st.button("Predict Lung Disease"):
        input_data = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, wheezing, chest_pain]])
        predict_disease(models["Lung Disease"], input_data, "Lung Disease", "Positive", "Negative")

def predict_parkinson():
    st.subheader("Parkinson's Disease Prediction")
    features = [st.number_input(label, min_value=0.0) for label in [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE",
        "Age"]]
    if st.button("Predict Parkinson's Disease"):
        predict_disease(models["Parkinson's Disease"], np.array([features]), "Parkinson's Disease", "Positive", "Negative")

def predict_preprocessed_lung_disease():
    st.subheader("Preprocessed Lung Disease Prediction")
    categorical_labels = ["Gender", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease",
                          "Fatigue", "Allergy", "Wheezing", "Alcohol Consuming", "Coughing", "Shortness of Breath",
                          "Swallowing Difficulty", "Chest Pain"]
    numerical_labels = ["Age"]
    numerical_values = [st.number_input(label, min_value=0.0) for label in numerical_labels]
    categorical_values = [1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0 for label in categorical_labels]
    input_data = np.array([numerical_values + categorical_values])
    if st.button("Predict Preprocessed Lung Disease"):
        predict_disease(models["Preprocessed Lungs Disease"], input_data, "Preprocessed Lung Disease", "Positive", "Negative")


# Disease Diagnosis Routes
if option == "Diabetes":
    predict_diabetes()
elif option == "Heart Disease":
    predict_heart_disease()
elif option == "Lung Disease":
    predict_lung_disease()
elif option == "Parkinson's Disease":
    predict_parkinson()
elif option == "Preprocessed Lungs Disease":
    predict_preprocessed_lung_disease()

