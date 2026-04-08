from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, '..', 'frontend', 'dist')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'SAV_File')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)

MODEL_CONFIGS = {
    'diabetes': {
        'name': 'Diabetes',
        'filename': 'diabetes_model.sav',
        'fields': [
            {'name': 'pregnancies', 'label': 'Pregnancies', 'type': 'number'},
            {'name': 'glucose', 'label': 'Glucose Level', 'type': 'number'},
            {'name': 'blood_pressure', 'label': 'Blood Pressure', 'type': 'number'},
            {'name': 'skin_thickness', 'label': 'Skin Thickness', 'type': 'number'},
            {'name': 'insulin', 'label': 'Insulin Level', 'type': 'number'},
            {'name': 'bmi', 'label': 'BMI', 'type': 'number'},
            {'name': 'diabetes_pedigree_function', 'label': 'Diabetes Pedigree Function', 'type': 'number'},
            {'name': 'age', 'label': 'Age', 'type': 'number'},
        ],
        'labels': ['Non-Diabetic', 'Diabetic'],
    },
    'heart-disease': {
        'name': 'Heart Disease',
        'filename': 'heart_disease_model.sav',
        'fields': [
            {'name': 'age', 'label': 'Age', 'type': 'number'},
            {'name': 'sex', 'label': 'Sex', 'type': 'select', 'options': [{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}]},
            {'name': 'cp', 'label': 'Chest Pain Type', 'type': 'number'},
            {'name': 'trestbps', 'label': 'Resting Blood Pressure', 'type': 'number'},
            {'name': 'chol', 'label': 'Cholesterol Level', 'type': 'number'},
            {'name': 'fbs', 'label': 'Fasting Blood Sugar > 120 mg/dl', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'restecg', 'label': 'Resting ECG Results', 'type': 'number'},
            {'name': 'thalach', 'label': 'Maximum Heart Rate', 'type': 'number'},
            {'name': 'exang', 'label': 'Exercise Induced Angina', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'oldpeak', 'label': 'ST Depression Induced by Exercise', 'type': 'number'},
            {'name': 'slope', 'label': 'Slope of Peak Exercise ST Segment', 'type': 'number'},
            {'name': 'ca', 'label': 'Number of Major Vessels', 'type': 'number'},
            {'name': 'thal', 'label': 'Thalassemia', 'type': 'number'},
        ],
        'labels': ['No Heart Disease', 'Heart Disease Detected'],
    },
    'fetal-health': {
        'name': 'Fetal Health',
        'filename': 'fetal_health.sav',
        'fields': [
            {'name': 'baseline_value', 'label': 'Baseline Value', 'type': 'number'},
            {'name': 'accelerations', 'label': 'Accelerations', 'type': 'number'},
            {'name': 'fetal_movement', 'label': 'Fetal Movement', 'type': 'number'},
            {'name': 'uterine_contractions', 'label': 'Uterine Contractions', 'type': 'number'},
            {'name': 'light_decelerations', 'label': 'Light Decelerations', 'type': 'number'},
            {'name': 'severe_decelerations', 'label': 'Severe Decelerations', 'type': 'number'},
            {'name': 'prolongued_decelerations', 'label': 'Prolongued Decelerations', 'type': 'number'},
            {'name': 'abnormal_short_term_variability', 'label': 'Abnormal Short Term Variability', 'type': 'number'},
            {'name': 'mean_value_of_short_term_variability', 'label': 'Mean Short Term Variability', 'type': 'number'},
            {'name': 'percentage_of_time_with_abnormal_long_term_variability', 'label': 'Percentage of Time with Abnormal Long Term Variability', 'type': 'number'},
            {'name': 'mean_value_of_long_term_variability', 'label': 'Mean Long Term Variability', 'type': 'number'},
            {'name': 'histogram_width', 'label': 'Histogram Width', 'type': 'number'},
            {'name': 'histogram_min', 'label': 'Histogram Min', 'type': 'number'},
            {'name': 'histogram_max', 'label': 'Histogram Max', 'type': 'number'},
            {'name': 'histogram_number_of_peaks', 'label': 'Histogram Number of Peaks', 'type': 'number'},
            {'name': 'histogram_number_of_zeroes', 'label': 'Histogram Number of Zeroes', 'type': 'number'},
            {'name': 'histogram_mode', 'label': 'Histogram Mode', 'type': 'number'},
            {'name': 'histogram_mean', 'label': 'Histogram Mean', 'type': 'number'},
            {'name': 'histogram_median', 'label': 'Histogram Median', 'type': 'number'},
            {'name': 'histogram_variance', 'label': 'Histogram Variance', 'type': 'number'},
            {'name': 'histogram_tendency', 'label': 'Histogram Tendency', 'type': 'number'},
        ],
        'labels': {1: 'Normal', 2: 'Suspect', 3: 'Pathological'},
    },
    'lung-disease': {
        'name': 'Lung Disease',
        'filename': 'Lungs_model.sav',
        'fields': [
            {'name': 'gender', 'label': 'Gender', 'type': 'select', 'options': [{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}]},
            {'name': 'age', 'label': 'Age', 'type': 'number'},
            {'name': 'smoking', 'label': 'Smoking', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'yellow_fingers', 'label': 'Yellow Fingers', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'anxiety', 'label': 'Anxiety', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'peer_pressure', 'label': 'Peer Pressure', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'chronic_disease', 'label': 'Chronic Disease', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'fatigue', 'label': 'Fatigue', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'allergy', 'label': 'Allergy', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'wheezing', 'label': 'Wheezing', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'alcohol_consuming', 'label': 'Alcohol Consuming', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'coughing', 'label': 'Coughing', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'shortness_of_breath', 'label': 'Shortness of Breath', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'swallowing_difficulty', 'label': 'Swallowing Difficulty', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'chest_pain', 'label': 'Chest Pain', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
        ],
        'labels': ['Negative', 'Positive'],
    },
    'parkinson': {
        'name': "Parkinson's Disease",
        'filename': 'parkinson.sav',
        'fields': [
            {'name': 'mdvp_fo', 'label': 'MDVP:Fo(Hz)', 'type': 'number'},
            {'name': 'mdvp_fhi', 'label': 'MDVP:Fhi(Hz)', 'type': 'number'},
            {'name': 'mdvp_flo', 'label': 'MDVP:Flo(Hz)', 'type': 'number'},
            {'name': 'mdvp_jitter', 'label': 'MDVP:Jitter(%)', 'type': 'number'},
            {'name': 'mdvp_jitter_abs', 'label': 'MDVP:Jitter(Abs)', 'type': 'number'},
            {'name': 'mdvp_rap', 'label': 'MDVP:RAP', 'type': 'number'},
            {'name': 'mdvp_ppq', 'label': 'MDVP:PPQ', 'type': 'number'},
            {'name': 'jitter_ddp', 'label': 'Jitter:DDP', 'type': 'number'},
            {'name': 'mdvp_shimmer', 'label': 'MDVP:Shimmer', 'type': 'number'},
            {'name': 'mdvp_shimmer_db', 'label': 'MDVP:Shimmer(dB)', 'type': 'number'},
            {'name': 'shimmer_apq3', 'label': 'Shimmer:APQ3', 'type': 'number'},
            {'name': 'shimmer_apq5', 'label': 'Shimmer:APQ5', 'type': 'number'},
            {'name': 'mdvp_apq', 'label': 'MDVP:APQ', 'type': 'number'},
            {'name': 'shimmer_dda', 'label': 'Shimmer:DDA', 'type': 'number'},
            {'name': 'nhr', 'label': 'NHR', 'type': 'number'},
            {'name': 'hnr', 'label': 'HNR', 'type': 'number'},
            {'name': 'rpde', 'label': 'RPDE', 'type': 'number'},
            {'name': 'dfa', 'label': 'DFA', 'type': 'number'},
            {'name': 'spread1', 'label': 'Spread1', 'type': 'number'},
            {'name': 'spread2', 'label': 'Spread2', 'type': 'number'},
            {'name': 'd2', 'label': 'D2', 'type': 'number'},
            {'name': 'ppe', 'label': 'PPE', 'type': 'number'},
        ],
        'labels': ['Negative', 'Positive'],
    },
    'hypothyroid': {
        'name': 'Hypothyroid',
        'filename': 'hypothyroid.sav',
        'fields': [
            {'name': 'age', 'label': 'Age', 'type': 'number'},
            {'name': 'sex', 'label': 'Sex', 'type': 'select', 'options': [{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}]},
            {'name': 'on_thyroxine', 'label': 'On Thyroxine', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'query_on_thyroxine', 'label': 'Query on Thyroxine', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'on_antithyroid_medication', 'label': 'On Antithyroid Medication', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'sick', 'label': 'Sick', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'pregnant', 'label': 'Pregnant', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'thyroid_surgery', 'label': 'Thyroid Surgery', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'i131_treatment', 'label': 'I131 Treatment', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'query_hypothyroid', 'label': 'Query Hypothyroid', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'query_hyperthyroid', 'label': 'Query Hyperthyroid', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'lithium', 'label': 'Lithium', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'goitre', 'label': 'Goitre', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'tumor', 'label': 'Tumor', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'hypopituitary', 'label': 'Hypopituitary', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'psych', 'label': 'Psych', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'TSH', 'label': 'TSH', 'type': 'number'},
            {'name': 'T3', 'label': 'T3', 'type': 'number'},
            {'name': 'TT4', 'label': 'TT4', 'type': 'number'},
            {'name': 'T4U', 'label': 'T4U', 'type': 'number'},
            {'name': 'FTI', 'label': 'FTI', 'type': 'number'},
        ],
        'labels': ['Negative', 'Positive'],
    },
    'migraine': {
        'name': 'Migraine',
        'filename': 'migraine_model.sav',
        'fields': [
            {'name': 'age', 'label': 'Age', 'type': 'number'},
            {'name': 'duration', 'label': 'Duration of headache (hours)', 'type': 'number'},
            {'name': 'frequency', 'label': 'Frequency per month', 'type': 'number'},
            {'name': 'location', 'label': 'Pain Location', 'type': 'select', 'options': [{'label': 'Unilateral', 'value': 1}, {'label': 'Bilateral', 'value': 0}]},
            {'name': 'character', 'label': 'Pain Character', 'type': 'select', 'options': [{'label': 'Throbbing', 'value': 1}, {'label': 'Pressing', 'value': 0}]},
            {'name': 'intensity', 'label': 'Pain Intensity (1-5)', 'type': 'number'},
            {'name': 'nausea', 'label': 'Nausea', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'vomit', 'label': 'Vomit', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'phonophobia', 'label': 'Phonophobia', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'photophobia', 'label': 'Photophobia', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'visual', 'label': 'Visual Disturbances', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'sensory', 'label': 'Sensory Disturbances', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'dysphasia', 'label': 'Dysphasia', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'dysarthria', 'label': 'Dysarthria', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'vertigo', 'label': 'Vertigo', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'tinnitus', 'label': 'Tinnitus', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'hypoacusis', 'label': 'Hypoacusis', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'diplopia', 'label': 'Diplopia', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'defect', 'label': 'Visual Field Defect', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'ataxia', 'label': 'Ataxia', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'conscience', 'label': 'Conscience Disorder', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'paresthesia', 'label': 'Paresthesia', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'dpf', 'label': 'DPF', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
        ],
    },
    'preprocessed-hypothyroid': {
        'name': 'Preprocessed Hypothyroid',
        'filename': 'prepocessed_hypothyroid_model.sav',
        'fields': [
            {'name': 'age', 'label': 'Age', 'type': 'number'},
            {'name': 'sex', 'label': 'Sex', 'type': 'select', 'options': [{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}]},
            {'name': 'on_thyroxine', 'label': 'On Thyroxine', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'TSH', 'label': 'TSH Level', 'type': 'number'},
            {'name': 'T3_measured', 'label': 'T3 Measured', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'T3', 'label': 'T3 Level', 'type': 'number'},
            {'name': 'TT4', 'label': 'TT4 Level', 'type': 'number'},
        ],
        'labels': ['Negative', 'Positive'],
    },
    'preprocessed-lungs-disease': {
        'name': 'Preprocessed Lungs Disease',
        'filename': 'prepocessed_lungs_model.sav',
        'fields': [
            {'name': 'gender', 'label': 'Gender', 'type': 'select', 'options': [{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}]},
            {'name': 'age', 'label': 'Age', 'type': 'number'},
            {'name': 'smoking', 'label': 'Smoking', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'yellow_fingers', 'label': 'Yellow Fingers', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'anxiety', 'label': 'Anxiety', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'peer_pressure', 'label': 'Peer Pressure', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'chronic_disease', 'label': 'Chronic Disease', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'fatigue', 'label': 'Fatigue', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'allergy', 'label': 'Allergy', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'wheezing', 'label': 'Wheezing', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'alcohol_consuming', 'label': 'Alcohol Consuming', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'coughing', 'label': 'Coughing', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'shortness_of_breath', 'label': 'Shortness of Breath', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'swallowing_difficulty', 'label': 'Swallowing Difficulty', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
            {'name': 'chest_pain', 'label': 'Chest Pain', 'type': 'select', 'options': [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]},
        ],
        'labels': ['Negative', 'Positive'],
    },
}

MODEL_CACHE = {}

for key, config in MODEL_CONFIGS.items():
    model_path = os.path.join(MODEL_DIR, config['filename'])
    if os.path.exists(model_path):
        try:
            MODEL_CACHE[key] = joblib.load(model_path)
        except Exception as exc:
            print(f'Failed to load {key}: {exc}')
    else:
        print(f'Missing model file: {model_path}')


@app.route('/api/metadata', methods=['GET'])
def metadata():
    return jsonify({
        key: {'name': config['name'], 'fields': config['fields']}
        for key, config in MODEL_CONFIGS.items()
    })


def format_prediction(config, raw_value):
    labels = config.get('labels')
    if labels is None:
        return str(raw_value)
    if isinstance(labels, dict):
        return labels.get(raw_value, str(raw_value))
    if isinstance(labels, list):
        return labels[raw_value] if 0 <= raw_value < len(labels) else str(raw_value)
    return str(raw_value)


def parse_value(field, value):
    if value is None:
        raise ValueError(f'Missing value for {field["label"]}')
    if field['type'] == 'number' or field['type'] == 'select':
        return float(value)
    raise ValueError(f'Unsupported field type: {field["type"]}')


@app.route('/api/predict/<model_key>', methods=['POST'])
def predict(model_key):
    config = MODEL_CONFIGS.get(model_key)
    if config is None:
        return jsonify({'error': 'Unknown model key'}), 404

    model = MODEL_CACHE.get(model_key)
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    payload = request.get_json(silent=True)
    if not payload or 'inputs' not in payload:
        return jsonify({'error': 'Request body must include inputs'}), 400

    inputs = payload['inputs']
    try:
        values = [parse_value(field, inputs.get(field['name'])) for field in config['fields']]
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    prediction = model.predict(np.array([values]))
    raw_value = int(prediction[0]) if hasattr(prediction[0], '__int__') else prediction[0]
    label = format_prediction(config, raw_value)

    return jsonify({'prediction': raw_value, 'label': label, 'model': config['name']})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
