import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Models
diabetes_model = joblib.load("SAV File/diabetes_model.sav")
heart_model = joblib.load("SAV File/heart_disease_model.sav")
fetal_model = joblib.load("SAV File/fetal_health.sav")  
lungs_model = joblib.load("SAV File/Lungs_model.sav")
hypothyroid_model = joblib.load("SAV File/hypothyroid.sav")
migraine_model = joblib.load("SAV File/migraine_model.sav")
parkinson_model = joblib.load("SAV File/parkinson.sav")
PHypothyroid_model = joblib.load("SAV File/prepocessed_hypothyroid_model.sav")
PLungs_model = joblib.load("SAV File/prepocessed_lungs_model.sav")

# Hide Streamlit Default UI Elements
st.markdown("""
    <style>
        #MainMenu, header, footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Custom CSS Styling
st.markdown("""
    <style>
        div[data-baseweb="select"] {
            border: 3px solid #FDFCDC !important; 
            border-radius: 12px !important;  
            padding: 5px !important;
            background-color: #f4f4f4 !important;
            transition: all 0.3s ease-in-out;
        }
        div[data-baseweb="select"]:hover {
            border-color: #FFC107 !important;
            background-color: #fff9c4 !important;
        }
        div[data-baseweb="select"] * {
            color: white !important;
            font-weight: bold !important;
        }
        [data-testid="stSidebar"] {
            background-color: #1976D2 !important;
            padding: 20px !important;
            border-radius: 0px 15px 15px 0px;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"]::before {
            content: "⚙️ Settings";
            display: block;
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.sidebar.title("AI-Powered Medical Diagnosis System")

# Sidebar Navigation
option = st.sidebar.selectbox("Select Diagnosis Type", [
    "Home", "Diabetes", "Heart Disease", "Fetal Health",
    "Lung Disease", "Parkinson's Disease", "Hypothyroid",
    "Migraine", "Preprocessed Hypothyroid Disease", "Preprocessed Lungs Disease"
])  # Fixed capitalization

# Home Page
if option == "Home":
    st.markdown("""
        <div style="text-align: center;">
            <h1>⚕️ Welcome to the AI-Powered Medical Diagnosis System</h1>
        </div>
    """, unsafe_allow_html=True)

    # Display Image from URL
    image_url = "https://services.brieflands.com/cdn/serve/316a3/d822ee1d2d9bfa6c9ed8e7c36ae61a1d8705918c/Explore%20how%20AI%20is%20revolutionizing%20healthcare,%20from%20disease%20tracking%20and%20pathology%20to%20psychiatric%20care,%20promising%20a%20transformative%20impact%20on%20medicine..jpeg"
    st.image(image_url, caption="AI in Healthcare", use_container_width=True)

    st.write("""
        ## 🏥 The Importance of Good Health  

        A healthy lifestyle is the foundation of a happy and active life. Follow these key principles to improve your overall well-being.  

        ### ✅ Why Prioritize Your Health?  
        - **More Energy** ⚡: Stay active and productive throughout the day.  
        - **Stronger Immunity** 🛡️: Reduce the risk of infections and illnesses.  
        - **Better Mental Health** 😊: Manage stress and maintain a positive mindset.  
        - **Longevity** 🎉: Live a longer, healthier life with fewer complications.  

        ### 🛠 Essential Health Tips  
        - **Eat a Balanced Diet** 🥗: Include a variety of nutrient-rich foods like fruits, vegetables, and proteins.  
        - **Exercise Regularly** 🏃‍♂️: Aim for at least 30 minutes of physical activity daily.  
        - **Stay Hydrated** 💧: Drink plenty of water to support vital body functions.  
        - **Get Quality Sleep** 😴: Sleep 7-9 hours per night for optimal recovery and focus.  
        - **Take Care of Your Mind** 🧘‍♀️: Practice mindfulness, meditation, and stress management.  
        - **Schedule Routine Checkups** 🏥: Regular doctor visits help with early detection and prevention.  
    """)

    st.warning("⚠️ This is AI-generated information. Always consult a medical professional for personalized health advice.")

# Improved Input Function
def display_input(label, tooltip, key, input_type="text"):
    if input_type == "number":
        return st.number_input(label, key=key, help=tooltip, step=1)
    return st.text_input(label, key=key, help=tooltip)


# Prediction Logic for Different Diseases
def predict_diabetes():
    st.subheader("Diabetes Diagnosis")
    
    # Input fields for diabetes risk factors
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=1000)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120)
    
    if st.button("Predict Diabetes"):
        # Prepare input data for the model
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        
        # Predict using the diabetes model
        result = diabetes_model.predict(input_data)
        
        # Display the result
        st.write("Result:", "Diabetic" if result[0] == 1 else "Non-Diabetic")

def predict_heart_disease():
    st.subheader("Heart Disease Prediction")
    
    # Input fields for heart disease risk factors
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
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0
    
    if st.button("Predict Heart Disease"):
        # Prepare input data for the model
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Predict using the heart disease model
        result = heart_model.predict(input_data)
        
        # Display the result
        st.write("Result:", "Heart Disease Detected" if result[0] == 1 else "No Heart Disease")
def predict_fetal_health():
    st.subheader("Fetal Health Prediction")
    
    # Input fields for fetal health risk factors
    baseline_value = st.number_input("Baseline Value", min_value=0.0, max_value=200.0, step=0.1)
    accelerations = st.number_input("Accelerations", min_value=0.0, max_value=1.0, step=0.001)
    fetal_movement = st.number_input("Fetal Movement", min_value=0.0, max_value=1.0, step=0.001)
    uterine_contractions = st.number_input("Uterine Contractions", min_value=0.0, max_value=1.0, step=0.001)
    light_decelerations = st.number_input("Light Decelerations", min_value=0.0, max_value=1.0, step=0.001)
    severe_decelerations = st.number_input("Severe Decelerations", min_value=0.0, max_value=1.0, step=0.001)
    prolongued_decelerations = st.number_input("Prolongued Decelerations", min_value=0.0, max_value=1.0, step=0.001)
    abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", min_value=0.0, max_value=100.0, step=0.1)
    mean_value_of_short_term_variability = st.number_input("Mean Short Term Variability", min_value=0.0, max_value=10.0, step=0.1)
    percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percentage of Time with Abnormal Long Term Variability", min_value=0.0, max_value=100.0, step=0.1)
    mean_value_of_long_term_variability = st.number_input("Mean Long Term Variability", min_value=0.0, max_value=50.0, step=0.1)
    histogram_width = st.number_input("Histogram Width", min_value=0.0, max_value=300.0, step=0.1)
    histogram_min = st.number_input("Histogram Min", min_value=0.0, max_value=200.0, step=0.1)
    histogram_max = st.number_input("Histogram Max", min_value=0.0, max_value=300.0, step=0.1)
    histogram_number_of_peaks = st.number_input("Histogram Number of Peaks", min_value=0, max_value=20, step=1)
    histogram_number_of_zeroes = st.number_input("Histogram Number of Zeroes", min_value=0, max_value=20, step=1)
    histogram_mode = st.number_input("Histogram Mode", min_value=0.0, max_value=200.0, step=0.1)
    histogram_mean = st.number_input("Histogram Mean", min_value=0.0, max_value=200.0, step=0.1)
    histogram_median = st.number_input("Histogram Median", min_value=0.0, max_value=200.0, step=0.1)
    histogram_variance = st.number_input("Histogram Variance", min_value=0.0, max_value=200.0, step=0.1)
    histogram_tendency = st.number_input("Histogram Tendency", min_value=0.0, max_value=1.0, step=0.1)
    
    if st.button("Predict Fetal Health"):
        # Prepare input data for the model
        input_data = np.array([[baseline_value, accelerations, fetal_movement, uterine_contractions, light_decelerations,
                                severe_decelerations, prolongued_decelerations, abnormal_short_term_variability,
                                mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability,
                                mean_value_of_long_term_variability, histogram_width, histogram_min, histogram_max,
                                histogram_number_of_peaks, histogram_number_of_zeroes, histogram_mode, histogram_mean,
                                histogram_median, histogram_variance, histogram_tendency]])
        
        # Predict using the fetal health model
        result = fetal_model.predict(input_data)
        
        # Display the result
        st.write("Result:", "Normal" if result[0] == 1 else "Suspect" if result[0] == 2 else "Pathological")

def predict_lung_cancer():
    st.subheader("Lung Cancer Diagnosis")
    
    # Input fields for lung cancer risk factors
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
    
    # Convert categorical inputs to numerical values
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
    
    if st.button("Predict Lung Cancer"):
        # Prepare input data for the model
        input_data = [[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]]
        
        # Predict using the lung cancer model (replace `lung_cancer_model` with your actual model)
        result = lungs_model.predict(input_data)
        
        # Display the result
        st.write("Result:", "Positive" if result[0] == 1 else "Negative")


def predict_parkinson():
    st.subheader("Parkinson's Disease Prediction")
    
    # Input fields for Parkinson's disease features
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=300.0, step=0.1)
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=300.0, step=0.1)
    mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=300.0, step=0.1)
    mdvp_jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=1.0, step=0.0001)
    mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.1, step=0.00001)
    mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1, step=0.0001)
    mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1, step=0.0001)
    jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.1, step=0.0001)
    mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, step=0.0001)
    mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=10.0, step=0.1)
    shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.1, step=0.0001)
    shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.1, step=0.0001)
    mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.1, step=0.0001)
    shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.1, step=0.0001)
    nhr = st.number_input("NHR", min_value=0.0, max_value=1.0, step=0.0001)
    hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, step=0.1)
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, step=0.0001)
    dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, step=0.0001)
    spread1 = st.number_input("Spread1", min_value=-10.0, max_value=10.0, step=0.0001)
    spread2 = st.number_input("Spread2", min_value=-5.0, max_value=5.0, step=0.0001)
    d2 = st.number_input("D2", min_value=0.0, max_value=5.0, step=0.0001)
    ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, step=0.0001)
    
    if st.button("Predict Parkinson's Disease"):
        # Prepare input data for the model
        input_data = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs, mdvp_rap,
                                mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
                                shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa,
                                spread1, spread2, d2, ppe]])
        
        # Predict using the Parkinson's disease model
        result = parkinson_model.predict(input_data)
        
        # Display the result
        st.write("Result:", "Parkinson's Positive" if result[0] == 1 else "Parkinson's Negative")


def predict_hypothyroid():
    st.subheader("Hypothyroid Prediction")
    
    # Input fields for hypothyroid risk factors
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    on_thyroxine = st.selectbox("On Thyroxine", ["No", "Yes"])
    query_on_thyroxine = st.selectbox("Query on Thyroxine", ["No", "Yes"])
    on_antithyroid_medication = st.selectbox("On Antithyroid Medication", ["No", "Yes"])
    sick = st.selectbox("Sick", ["No", "Yes"])
    pregnant = st.selectbox("Pregnant", ["No", "Yes"])
    thyroid_surgery = st.selectbox("Thyroid Surgery", ["No", "Yes"])
    i131_treatment = st.selectbox("I131 Treatment", ["No", "Yes"])
    query_hypothyroid = st.selectbox("Query Hypothyroid", ["No", "Yes"])
    query_hyperthyroid = st.selectbox("Query Hyperthyroid", ["No", "Yes"])
    lithium = st.selectbox("Lithium", ["No", "Yes"])
    goitre = st.selectbox("Goitre", ["No", "Yes"])
    tumor = st.selectbox("Tumor", ["No", "Yes"])
    hypopituitary = st.selectbox("Hypopituitary", ["No", "Yes"])
    psych = st.selectbox("Psych", ["No", "Yes"])
    TSH = st.number_input("TSH", min_value=0.0, max_value=100.0, step=0.1)
    T3 = st.number_input("T3", min_value=0.0, max_value=10.0, step=0.1)
    TT4 = st.number_input("TT4", min_value=0.0, max_value=300.0, step=0.1)
    T4U = st.number_input("T4U", min_value=0.0, max_value=2.0, step=0.01)
    FTI = st.number_input("FTI", min_value=0.0, max_value=300.0, step=0.1)
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    on_thyroxine = 1 if on_thyroxine == "Yes" else 0
    query_on_thyroxine = 1 if query_on_thyroxine == "Yes" else 0
    on_antithyroid_medication = 1 if on_antithyroid_medication == "Yes" else 0
    sick = 1 if sick == "Yes" else 0
    pregnant = 1 if pregnant == "Yes" else 0
    thyroid_surgery = 1 if thyroid_surgery == "Yes" else 0
    i131_treatment = 1 if i131_treatment == "Yes" else 0
    query_hypothyroid = 1 if query_hypothyroid == "Yes" else 0
    query_hyperthyroid = 1 if query_hyperthyroid == "Yes" else 0
    lithium = 1 if lithium == "Yes" else 0
    goitre = 1 if goitre == "Yes" else 0
    tumor = 1 if tumor == "Yes" else 0
    hypopituitary = 1 if hypopituitary == "Yes" else 0
    psych = 1 if psych == "Yes" else 0
    
    if st.button("Predict Hypothyroid"):
        # Prepare input data for the model
        input_data = np.array([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication,
                                sick, pregnant, thyroid_surgery, i131_treatment, query_hypothyroid,
                                query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych,
                                TSH, T3, TT4, T4U, FTI]])
        
        # Predict using the hypothyroid model
        result = hypothyroid_model.predict(input_data)
        
        # Display the result
        st.write("Result:", "Hypothyroid Positive" if result[0] == 1 else "Hypothyroid Negative")

def predict_migraine():
    st.subheader("Migraine Prediction")
    
    # Input fields for migraine risk factors
    age = st.number_input("Age", min_value=0, max_value=120)
    duration = st.number_input("Duration of headache (hours)", min_value=0, max_value=48)
    frequency = st.number_input("Frequency per month", min_value=0, max_value=30)
    location = st.selectbox("Pain Location", ["Unilateral", "Bilateral"])
    character = st.selectbox("Pain Character", ["Throbbing", "Pressing"])
    intensity = st.number_input("Pain Intensity (1-5)", min_value=1, max_value=5)
    nausea = st.selectbox("Nausea", ["No", "Yes"])
    vomit = st.selectbox("Vomit", ["No", "Yes"])
    phonophobia = st.selectbox("Phonophobia", ["No", "Yes"])
    photophobia = st.selectbox("Photophobia", ["No", "Yes"])
    visual = st.selectbox("Visual Disturbances", ["No", "Yes"])
    sensory = st.selectbox("Sensory Disturbances", ["No", "Yes"])
    dysphasia = st.selectbox("Dysphasia", ["No", "Yes"])
    dysarthria = st.selectbox("Dysarthria", ["No", "Yes"])
    vertigo = st.selectbox("Vertigo", ["No", "Yes"])
    tinnitus = st.selectbox("Tinnitus", ["No", "Yes"])
    hypoacusis = st.selectbox("Hypoacusis", ["No", "Yes"])
    diplopia = st.selectbox("Diplopia", ["No", "Yes"])
    defect = st.selectbox("Visual Field Defect", ["No", "Yes"])
    ataxia = st.selectbox("Ataxia", ["No", "Yes"])
    conscience = st.selectbox("Conscience Disorder", ["No", "Yes"])
    paresthesia = st.selectbox("Paresthesia", ["No", "Yes"])
    dpf = st.selectbox("DPF", ["No", "Yes"])
    
    # Convert categorical inputs to numerical values
    location = 1 if location == "Unilateral" else 0
    character = 1 if character == "Throbbing" else 0
    nausea = 1 if nausea == "Yes" else 0
    vomit = 1 if vomit == "Yes" else 0
    phonophobia = 1 if phonophobia == "Yes" else 0
    photophobia = 1 if photophobia == "Yes" else 0
    visual = 1 if visual == "Yes" else 0
    sensory = 1 if sensory == "Yes" else 0
    dysphasia = 1 if dysphasia == "Yes" else 0
    dysarthria = 1 if dysarthria == "Yes" else 0
    vertigo = 1 if vertigo == "Yes" else 0
    tinnitus = 1 if tinnitus == "Yes" else 0
    hypoacusis = 1 if hypoacusis == "Yes" else 0
    diplopia = 1 if diplopia == "Yes" else 0
    defect = 1 if defect == "Yes" else 0
    ataxia = 1 if ataxia == "Yes" else 0
    conscience = 1 if conscience == "Yes" else 0
    paresthesia = 1 if paresthesia == "Yes" else 0
    dpf = 1 if dpf == "Yes" else 0
    
    if st.button("Predict Migraine Type"):
        # Prepare input data for the model
        input_data = np.array([[age, duration, frequency, location, character, intensity,
                                nausea, vomit, phonophobia, photophobia, visual, sensory,
                                dysphasia, dysarthria, vertigo, tinnitus, hypoacusis, diplopia,
                                defect, ataxia, conscience, paresthesia, dpf]])
        
        # Predict using the migraine model
        result = migraine_model.predict(input_data)
        
        # Display the result
        st.write("Migraine Type:", result[0])

def predict_Preprocessed_hypothyroid():
    st.subheader("Preprocessed Hypothyroid Prediction")
    
    # Input fields for hypothyroid risk factors
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    on_thyroxine = st.selectbox("On Thyroxine", ["No", "Yes"])
    TSH = st.number_input("TSH Level", min_value=0.0, max_value=10.0, format="%.2f")
    T3_measured = st.selectbox("T3 Measured", ["No", "Yes"])
    T3 = st.number_input("T3 Level", min_value=0.0, max_value=10.0, format="%.2f")
    TT4 = st.number_input("TT4 Level", min_value=0.0, max_value=300.0, format="%.1f")
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    on_thyroxine = 1 if on_thyroxine == "Yes" else 0
    T3_measured = 1 if T3_measured == "Yes" else 0
    
    if st.button("Predict Preprocessed Hypothyroid"):
        # Prepare input data for the model
        input_data = np.array([[age, sex, on_thyroxine, TSH, T3_measured, T3, TT4]])
        
        # Predict using the hypothyroid model
        result = hypothyroid_model.predict(input_data)
        
        # Display the result
        st.write("Preprocessed Hypothyroid Diagnosis:", "Positive" if result[0] == 1 else "Negative")

def predict_preprocessed_lung_disease():
    st.subheader("Lung Disease Prediction")
    
    # Input fields for lung disease risk factors
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
    
    # Convert categorical inputs to numerical values
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
        # Prepare input data for the model
        input_data = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, 
                                fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, 
                                swallowing_difficulty, chest_pain]])
        
        # Predict using the lung disease model
        result = lungs_model.predict(input_data)
        
        # Display the result
        st.write("Lung Disease Diagnosis:", "Positive" if result[0] == 1 else "Negative")


# Disease Diagnosis Routes
if option == "Diabetes":
    predict_diabetes()
elif option == "Heart Disease":
    predict_heart_disease()
elif option == "Fetal Health Disease":
    predict_fetal_health()
elif option == "Lung Disease":
    predict_lung_cancer()
elif option == "Parkinson's Disease":
    predict_parkinson()
elif option == "Hypothyroid Disease":
    predict_hypothyroid()
elif option == "Migrane Disease":
    predict_migraine()
elif option == "preprocessed Hypothyroid Disease":
    predict_Preprocessed_hypothyroid()
elif option == "Preprocessed Lungs Disease":
    predict_preprocessed_lung_disease()

