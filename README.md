# 🛡️ PhishLens — Adaptive Explainable Phishing Detection System

A machine learning-powered phishing website detection system with explainable AI,
WHOIS domain analysis, and a real-time React frontend.

## 📁 Project Structure

frontend/        → React.js UI<br>
backend/         → FastAPI REST API<br>
notebooks/       → EDA, training, evaluation<br>
data/            → UCI Phishing Dataset<br>

## ⚙️ Local Setup

### Backend
cd backend<br>
python -m venv venv<br>
venv\Scripts\activate<br>
pip install -r requirements.txt<br>
uvicorn main:app --reload<br>

### Frontend
cd frontend<br>
npm install<br>
npm start<br>

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