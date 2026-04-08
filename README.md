# AI Powered Medical Diagnosis System

An interactive web application for medical diagnosis using machine learning models. Supports diagnosis for diabetes, heart disease, fetal health, lung disease, Parkinson's, hypothyroid, migraine, and more.

## Features

- Interactive React frontend with Tailwind CSS
- Flask backend API serving 9 different medical diagnosis models
- Dynamic form generation based on model requirements
- Responsive design for desktop and mobile

## Supported Diagnoses

- Diabetes
- Heart Disease
- Fetal Health
- Lung Disease (Cancer)
- Parkinson's Disease
- Hypothyroid
- Migraine
- Preprocessed Hypothyroid
- Preprocessed Lungs Disease

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Arkadeep01/AI_Powered_Medical_Diagnosis_System.git
   cd AI_Powered_Medical_Diagnosis_System
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. Start the backend:
   ```bash
   python backend/api.py
   ```

5. In another terminal, start the frontend:
   ```bash
   cd frontend
   npm run dev
   ```

6. Open http://localhost:5173 in your browser.

## Deployment

### Using Docker (Recommended)

1. Build the frontend:
   ```bash
   ./build.sh
   ```

2. Build the Docker image:
   ```bash
   docker build -t medical-diagnosis-app .
   ```

3. Run the container:
   ```bash
   docker run -p 5000:5000 medical-diagnosis-app
   ```

4. Access the application at http://localhost:5000

### Manual Deployment

1. Build the application:
   ```bash
   ./build.sh
   ```

2. Start the Flask app:
   ```bash
   python backend/api.py
   ```

3. The app will be available at http://localhost:5000

### Cloud Deployment

The application can be deployed to any cloud platform that supports Docker containers or Python applications:

- **Heroku**: Push the Docker image or use the Python buildpack
- **Railway**: Deploy the Docker image
- **Render**: Use the Dockerfile
- **AWS/GCP/Azure**: Deploy the container to their container services

For production, set the `PORT` environment variable to the port assigned by the platform.

## API Endpoints

- `GET /metadata` - Get model configurations
- `POST /predict/<model_key>` - Make a prediction for a specific model
- `GET /` - Serve the frontend application

## Model Files

All trained models are stored in the `SAV_File/` directory as `.sav` files using joblib.

## Technologies Used

- **Backend**: Flask, scikit-learn, joblib, numpy
- **Frontend**: React, Vite, Tailwind CSS
- **Deployment**: Docker

## License

[Add your license here]
