# 🛡️ PhishGuard — Adaptive Explainable Phishing Detection System

A machine learning-powered phishing website detection system with explainable AI,
WHOIS domain analysis, and a real-time React frontend.

## 📁 Project Structure

frontend/        → React.js UI
backend/         → FastAPI REST API
notebooks/       → EDA, training, evaluation
data/            → UCI Phishing Dataset

## ⚙️ Local Setup

### Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

### Frontend
cd frontend
npm install
npm start

## 🧠 Models
- Logistic Regression
- Random Forest
- XGBoost (primary)

## 📌 Features
- URL feature extraction (30+ features)
- SHAP explainability
- WHOIS domain analysis
- Trust score output
- Adaptive retraining via user feedback

## 👩‍💻 Team
- Anshita Sugandhi
- Farhana Tabassum
- Gangisetti Himasree

## 🎓 Under guidance of Dr. M. Shabana, CSE — NGIT