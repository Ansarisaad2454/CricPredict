---
title: CricPredict
emoji: 🏏
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# CricPredict – AI-Based IPL Match Prediction System 🏏

CricPredict is an AI-powered web application developed to analyze **Indian Premier League (IPL)** cricket data and provide intelligent match insights.  
The system combines **Machine Learning** and **Generative AI** to predict match outcomes, estimate scores, and answer cricket-related questions in natural language.

---

## 🚀 Key Features

- 🏆 **Win Probability Predictor**  
  Predicts the chances of a team winning based on current match conditions.

- 📊 **Score Prediction**  
  Estimates the final score of an innings using historical IPL data and live match features.

- 🤖 **AI Chatbot (RAG-based)**  
  Allows users to ask IPL-related questions in natural language and receive accurate, data-driven answers.

- 📈 **Interactive Analytics Dashboard**  
  Displays team and player performance using charts and visualizations.

- 🌐 **User-Friendly Web Interface**  
  Simple, fast, and responsive UI with real-time interaction.

---

## 🧠 Technologies Used

### 🔹 Machine Learning & AI
- XGBoost (Primary prediction model)
- Logistic Regression
- Decision Tree
- TF-IDF Vectorization
- FAISS (Vector similarity search)
- RAG (Retrieval-Augmented Generation)

### 🔹 Backend
- Python
- FastAPI
- Pandas, NumPy
- Pickle

### 🔹 Frontend
- HTML
- CSS
- JavaScript
- HTMX

---

## 📊 Dataset

- Historical IPL match data (2008–2025)
- Ball-by-ball data
- Player and team statistics
- Data sourced from publicly available cricket datasets

---

## ⚙️ System Workflow

1. Data Collection  
2. Data Preprocessing & Feature Engineering  
3. Model Training (Logistic Regression, Decision Tree, XGBoost)  
4. Model Evaluation & Selection  
5. Backend Integration using FastAPI  
6. Frontend Interaction via HTMX  
7. Prediction Output (Win Probability / Score)  
8. Chatbot Query Processing using RAG  

---

## Deploy On Hugging Face Spaces

This project is configured for **Hugging Face Spaces (Docker SDK)**.

### 1. Push this code to GitHub
- Make sure these files are in your repo root:
  - `main.py`
  - `requirements.txt`
  - `Dockerfile`
  - `.dockerignore`
  - `templates/`, `static/`, `data/`
  - `pipe.joblib`, `score_pipe.joblib`, `faiss_index.idx`, `rag_texts.pkl`

### 2. Create a Space
- Go to Hugging Face -> **New Space**
- Choose:
  - **Owner**: your account
  - **Space name**: e.g. `cricpredict`
  - **SDK**: `Docker`
  - **Visibility**: Public or Private

### 3. Add your repository to the Space
- Option A: Connect GitHub repo directly
- Option B: Upload files manually

### 4. Configure Space secrets (Settings -> Variables and secrets)
- Add these keys:
  - `SECRET_KEY` (required)
  - `GEMINI_API_KEY` (optional, for AI analysis)
  - `NEWSDATA_API_KEY` (optional, for live news)

If optional keys are missing, related features may show fallback/error text but the app can still run.

### 5. Build and run
- Hugging Face will auto-build using `Dockerfile`.
- Your app must listen on port `7860` inside the container (already handled).

### 6. Open your Space URL
- After build success, open:
  - `https://huggingface.co/spaces/<username>/<space-name>`

---

## 🧪 Model Performance

- Multiple models were tested for prediction accuracy
- **XGBoost** provided the best performance and stability
- Handles both balanced and unbalanced datasets effectively

---

## 🎯 Project Objectives

- Simplify complex cricket statistics for normal users
- Provide real-time predictive insights instead of only historical data
- Enable human-like interaction using an AI chatbot
- Combine predictive analytics with conversational intelligence

---

## 👨‍💻 Developer

**Saad Ansari**  
MSc Artificial Intelligence & Machine Learning  
Department of Computer Science, Gujarat University

---

## ⭐ If you like this project
Give it a ⭐ on GitHub and feel free to explore or improve it!
