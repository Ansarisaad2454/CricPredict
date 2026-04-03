
# 🏏 CricPredict  
## An AI-Based IPL Match Prediction and Analytics System

---

## 1. Introduction

CricPredict is an AI-powered web application developed to analyze Indian Premier League (IPL) cricket data and provide match predictions, statistical insights, and conversational answers through an intelligent chatbot.

The system integrates Machine Learning, Retrieval-Augmented Generation (RAG), and modern web technologies to simplify complex cricket analytics for both technical and non-technical users.

The application is trained on historical IPL data from 2008 to 2025 and offers real-time-like predictions through a clean, interactive interface.

---

## 2. Problem Statement

- IPL data is distributed across multiple websites  
- Raw statistics are difficult for normal users to understand  
- Existing platforms focus mainly on historical data  
- No system allows users to ask cricket-related questions in natural language  
- Data is often displayed in complex tables instead of visual insights  

---

## 3. Objectives of the Project

- Predict win probability during live matches  
- Estimate final innings scores  
- Provide team and player analytics  
- Build an AI chatbot for IPL-related queries  
- Present cricket data in a simple and interactive format  

---

## 4. System Features

### 4.1 AI Chatbot
- Answers IPL-related questions using natural language  
- Powered by RAG and vector search  

### 4.2 Win Probability Predictor
- Calculates winning chances during a match  
- Uses XGBoost for high accuracy  

### 4.3 Score Prediction
- Predicts final score based on current innings state  

### 4.4 Interactive Dashboard
- Visual representation of player and team performance  

---

## 5. Technologies Used

### Backend
- Python  
- FastAPI  
- Pandas  
- NumPy  

### Machine Learning & AI
- Logistic Regression  
- Decision Tree  
- XGBoost  
- FAISS  
- RAG  

### Frontend
- HTML  
- CSS  
- JavaScript  
- HTMX  

---

## 6. Dataset

- IPL data from 2008–2025  
- Match, player, team, and venue statistics  

---

## 7. Methodology

1. Data Collection  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation  
6. Backend Integration  
7. Frontend Interaction  
8. Prediction Output  

---

## 8. Installation & Setup

### Prerequisites
- Python 3.9+  
- Git  
- pip  

### Clone Repository
```bash
git clone https://github.com/your-username/CricPredict.git
cd CricPredict
```

### Create Virtual Environment
```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. Run Application

```bash
uvicorn main:app --reload
```

Access:
- http://127.0.0.1:8000  
- http://127.0.0.1:8000/docs  

---

## 10. Free Deployment (Render Recommended)

This project is now deployment-ready for Render free tier.

### Files added for deployment
- `render.yaml` (Blueprint for Render)
- `Procfile` (ASGI start command)
- `runtime.txt` (Python version)

### A) Deploy on Render (Free)
1. Push this project to GitHub.
2. Create a free account at Render.
3. Click **New +** -> **Blueprint**.
4. Connect your GitHub repo and select this project.
5. Render auto-detects `render.yaml` and creates the web service.
6. Set/verify environment variables:
	- `SECRET_KEY` (required, random string)
	- `GEMINI_API_KEY` (optional, enables AI analysis)
	- `NEWSDATA_API_KEY` (optional, enables live news)
7. Deploy and open the generated Render URL.

Default start command used:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### B) Deploy on Railway (Alternative Free Credits)
1. Create a Railway project and link your GitHub repo.
2. Set the start command to:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```
3. Add environment variables:
	- `SECRET_KEY`
	- `GEMINI_API_KEY` (optional)
	- `NEWSDATA_API_KEY` (optional)
4. Deploy.

### Notes for free plans
- First request may be slow due to cold starts.
- If optional API keys are missing, app still runs; AI analysis/news features return fallback responses.
- Keep model/data files in repo or storage accessible at boot.

---

## 11. Model Evaluation

XGBoost performed best compared to Logistic Regression and Decision Tree.

---

## 12. Conclusion

CricPredict successfully combines Machine Learning and Generative AI to provide accurate IPL predictions and analytics in a user-friendly manner.

---

## 13. Future Enhancements

- Live match data  
- Mobile application  
- Fantasy team recommendation  
- Multilingual chatbot  

---

## 14. Developer

**Saad Ansari**  
M.Sc. AI & ML  
Gujarat University  

---

## 15. License

For academic and educational purposes only.
