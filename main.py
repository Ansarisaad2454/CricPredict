# backend/main.py

# --- CRITICAL FIX FOR MAC OS SEGFAULT ---
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
# ----------------------------------------

import pandas as pd
import pickle
import joblib 
import re
from fastapi import FastAPI, Request, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from news_utils import fetch_cricket_news 
from embeddings import load_embedding_model, load_faiss_index
from retriever import IPLRetriever
from chatbot_logic import process_query_v2
from gemini_utils import get_gemini_analysis
from stats_utils import StatsEngine

# --- 1. SETUP FASTAPI & TEMPLATES ---
app = FastAPI(title="CricPredict API")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SECRET_KEY = os.environ.get("SECRET_KEY", "a-very-weak-default-key-for-dev-only")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- 2. LOAD MODELS ---
models = {}

@app.on_event("startup")
def load_models():
    print("Loading models...")
    try:
        # Load Retriever with master squads
        models["chatbot_retriever"] = IPLRetriever(
            csv_path="data/matches.csv",
            master_squads_path="data/master_squads.csv"
        )

        # Load Venue Stats
        try:
            with open("data/venue_stats.pkl", 'rb') as f:
                models["venue_stats"] = pickle.load(f)
                print("Venue Stats Loaded!")
        except: print("Warning: Venue stats missing.")
        
        # Load Win Predictor
        try:
            models["win_predictor_pipe"] = joblib.load("pipe.joblib")
            print("Win Predictor (XGBoost) Loaded!")
        except FileNotFoundError: print("Warning: pipe.joblib not found.")

        # Load Score Predictor
        try:
            models["score_predictor_pipe"] = joblib.load("score_pipe.joblib")
            print("Score Predictor (XGBoost) Loaded!")
        except Exception as e:
            print(f"Warning: Score predictor failed to load: {e}")

        # Load Player Data for Strength Calc
        try:
            with open("data/player_ratings.pkl", 'rb') as f:
                models["player_ratings"] = pickle.load(f)
            with open("data/player_teams.pkl", 'rb') as f:
                models["player_teams"] = pickle.load(f)
            print("Player Ratings & Teams Loaded!")
        except Exception as e:
            print(f"Warning: Player data for strength calculation missing: {e}")

        models["stats_engine"] = StatsEngine() 
        
        load_faiss_index() 
        load_embedding_model() 
        print("Models loaded successfully!")
    
    except Exception as e:
        print(f"Error loading models: {e}")

# --- HELPER: CALCULATE CURRENT STRENGTH ---
def get_current_team_strength(team_name: str) -> float:
    ratings = models.get("player_ratings", {})
    p_teams = models.get("player_teams", {})
    if not ratings or not p_teams: return 50.0 
    squad_members = [p for p, t in p_teams.items() if t == team_name]
    if not squad_members: return 50.0 
    squad_ratings = sorted([ratings.get(p, 5.0) for p in squad_members], reverse=True)
    top_11 = squad_ratings[:11]
    return float(sum(top_11))

# --- HELPER: NORMALIZE CITIES ---
def get_clean_city_list(retriever):
    """Returns a sorted list of cities with 'Bangalore' merged into 'Bengaluru'."""
    if not retriever: return []
    
    raw_cities = retriever.city_to_venues_map.keys()
    normalized_cities = set()
    
    for city in raw_cities:
        # Title case (e.g. "delhi" -> "Delhi")
        c_title = city.strip().title()
        
        # Merge Duplicates
        if c_title == "Bangalore":
            normalized_cities.add("Bengaluru")
        else:
            normalized_cities.add(c_title)
            
    return sorted(list(normalized_cities))

# --- 3. FORMATTING FUNCTION ---
def format_chatbot_response_for_html(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    text = text.replace("\\n", "<br>").replace("\n", "<br>")
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"((?:•\s.*(?:<br>|$))+)", r"<ul class='list-disc pl-5 mt-2 mb-2'>\1</ul>", text, flags=re.MULTILINE)
    text = re.sub(r"•\s(.*)", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<br>\s*)+", "<br>", text)
    if text.startswith("<br>"): text = text[4:]
    if text.endswith("<br>"): text = text[:-4]
    return text.strip()

# --- 4. CHATBOT ENDPOINTS ---
@app.get("/")
def read_root(request: Request):
    request.session.pop('chat_context', None)
    initial_message = format_chatbot_response_for_html(
        "Hi! I'm CricPredict.\nI can help with:\n\n"
        "• **Match Predictions** (Win & Score)\n"
        "• **Player & Team Stats**\n"
        "• **IPL History & Records**"
    )
    return templates.TemplateResponse("index.html", {"request": request, "initial_message": initial_message})

@app.post("/chat", response_class=HTMLResponse) 
async def handle_chat(request: Request, query: str = Form(...)):
    if "chatbot_retriever" not in models: return HTMLResponse(content="Error: Model not loaded.")
    retriever = models["chatbot_retriever"]
    session_context = request.session.get('chat_context', {})
    answer, updated_context = process_query_v2(query, retriever, session_context)
    request.session['chat_context'] = updated_context
    formatted_answer = format_chatbot_response_for_html(answer)
    template = templates.env.get_template("partials/chat_message.html")
    return HTMLResponse(content=template.render({"request": request, "role": "User", "message": query}) + 
                                template.render({"request": request, "role": "Assistant", "message": formatted_answer}))

# --- 5. PREDICTOR ENDPOINTS (FIXED ROUTES) ---

@app.get("/predictor")
def get_predictor_hub(request: Request):
    """Serves the Predictor Hub (Card Menu)."""
    return templates.TemplateResponse("predictor.html", {"request": request})

@app.get("/predictor/win")
def get_win_predictor_page(request: Request):
    """Serves the Win Predictor Page."""
    retriever = models.get("chatbot_retriever")
    if not retriever: return HTMLResponse("Error: Model data not loaded.")
    
    active_teams = [
        'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
        'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
        'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
    ]
    
    # Use the helper to get clean cities
    all_cities = get_clean_city_list(retriever)
    
    return templates.TemplateResponse("win.html", {
        "request": request, 
        "teams": sorted(active_teams), 
        "cities": all_cities
    })

@app.get("/predictor/score")
def get_score_predictor_page(request: Request):
    """Serves the Score Predictor Page."""
    retriever = models.get("chatbot_retriever")
    if not retriever: return HTMLResponse("Error: Model data not loaded.")
    
    active_teams = [
        'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
        'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
        'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
    ]
    
    # Use the helper to get clean cities
    all_cities = get_clean_city_list(retriever)
    
    return templates.TemplateResponse("score.html", {
        "request": request, 
        "teams": sorted(active_teams), 
        "cities": all_cities
    })

@app.post("/predict", response_class=HTMLResponse)
async def handle_prediction(
    request: Request,
    batting_team: str = Form(...),
    bowling_team: str = Form(...),
    city: str = Form(...),
    target: float = Form(...),
    score: float = Form(...),
    overs_done: float = Form(...),
    wickets: float = Form(...)
):
    # --- VALIDATION ---
    if batting_team == bowling_team:
        return HTMLResponse(content="<div class='text-red-500 font-bold p-4 bg-red-500/10 rounded-lg'>Error: Batting and Bowling teams cannot be the same.</div>")
    
    if wickets < 0 or wickets > 10:
        return HTMLResponse(content="<div class='text-red-500 font-bold p-4 bg-red-500/10 rounded-lg'>Error: Wickets must be between 0 and 10.</div>")
        
    if "win_predictor_pipe" not in models:
        return HTMLResponse(content="<div class='text-red-500 font-bold p-4 bg-red-500/10 rounded-lg'>Error: Model not loaded.</div>")
        
    pipe = models["win_predictor_pipe"]

    # --- NORMALIZATION ---
    # Ensure city name matches training data (e.g. Bangalore -> Bengaluru)
    city = city.strip().title()
    if city == "Bangalore":
        city = "Bengaluru"

    runs_left = target - score
    balls_left = 120 - (overs_done * 6)
    wickets_left = 10 - wickets
    balls_bowled = 120 - balls_left
    crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    if runs_left <= 0: rrr = 0

    bat_strength = get_current_team_strength(batting_team)
    bowl_strength = get_current_team_strength(bowling_team)

    input_df = pd.DataFrame({
        'batting_strength': [bat_strength],
        'bowling_strength': [bowl_strength],
        'city': [city],
        'runs_left': [max(0, runs_left)],
        'balls_left': [max(0, balls_left)],
        'wickets_left': [max(0, wickets_left)],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })
    
    try:
        # Check if city exists in trained categories
        trained_cities = pipe.named_steps['step1'].transformers_[0][1].categories_[0]
        if city not in trained_cities:
            input_df['city'] = 'Other' 
    except:
        pass

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    template = templates.env.get_template("partials/prediction_result.html")
    return HTMLResponse(content=template.render({
        "request": request,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "batting_strength": round(bat_strength, 1), 
        "bowling_strength": round(bowl_strength, 1), 
        "batting_win_prob": round(win * 100, 1),
        "bowling_win_prob": round(loss * 100, 1),
        "city": city,
        "target": target,
        "runs_left": round(runs_left, 0),
        "balls_left": round(balls_left, 0),
        "wickets_left": round(wickets_left, 0),
        "rrr": round(rrr, 2)
    }))

@app.post("/predict-score", response_class=HTMLResponse)
async def handle_score_prediction(
    request: Request,
    batting_team: str = Form(...),
    bowling_team: str = Form(...),
    city: str = Form(...),
    current_score: int = Form(...),
    wickets: int = Form(...),
    overs: float = Form(...),
    runs_last_5: int = Form(...)
):
    if "score_predictor_pipe" not in models: 
        return HTMLResponse(content="<div class='p-4 bg-red-500/20 text-red-200 border border-red-500/50 rounded-lg'>⚠️ Error: Model not loaded.</div>")
    
    pipe = models["score_predictor_pipe"]
    
    bat_strength = get_current_team_strength(batting_team)
    bowl_strength = get_current_team_strength(bowling_team)
    
    # Normalize City
    city = city.strip().title()
    if city == "Bangalore": city = "Bengaluru"
    
    venue_stats = models.get("venue_stats", {})
    venue_avg = venue_stats.get(city, 185.0) 
    
    wickets_last_5 = 1 if wickets > 0 else 0
    
    if overs > 0:
        crr = current_score / overs
        projected_simple = crr * 20
    else:
        projected_simple = 180

    input_df = pd.DataFrame({
        'batting_strength': [bat_strength],
        'bowling_strength': [bowl_strength],
        'venue_avg': [venue_avg],
        'current_score': [current_score],
        'wickets': [wickets],
        'overs': [overs],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5],
        'projected_score_simple': [projected_simple]
    })
    
    try:
        predicted_score = pipe.predict(input_df)[0]
        predicted_score = max(predicted_score, current_score)
        
        template = templates.env.get_template("partials/score_result.html")
        return HTMLResponse(content=template.render({
            "request": request, 
            "predicted_score": int(predicted_score), 
            "upper_bound": int(predicted_score + 10)
        }))
    except Exception as e:
        return HTMLResponse(content=f"<div class='p-4 bg-red-500/20 text-red-200 border border-red-500/50 rounded-lg'>⚠️ Prediction failed: {str(e)}</div>")

# --- 6. NEWS ENDPOINTS ---
@app.get("/news", response_class=HTMLResponse)
def get_news_page(request: Request):
    return templates.TemplateResponse("news.html", {"request": request})

@app.get("/get-news-articles", response_class=HTMLResponse)
def get_news_articles(request: Request):
    articles = fetch_cricket_news()
    if not articles: return HTMLResponse(content="<p class='text-white/50 col-span-full text-center'>No news found.</p>")
    html_content = ""
    for article in articles: html_content += templates.get_template("partials/news_article.html").render({"request": request, "article": article})
    return HTMLResponse(content=html_content)

# --- 7. STATS ROUTES (FIXED FOR 10 TEAMS) ---

@app.get("/stats", response_class=HTMLResponse)
def get_stats_hub(request: Request):
    return templates.TemplateResponse("stats.html", {"request": request})

@app.get("/stats/teams", response_class=HTMLResponse)
def get_team_stats_page(request: Request):
    """
    Serves the Team Stats Page with ONLY the 10 Active IPL Teams.
    """
    active_teams = [
        'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
        'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
        'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
    ]
    return templates.TemplateResponse("team_stats.html", {"request": request, "teams": sorted(active_teams)})

@app.get("/stats/players", response_class=HTMLResponse)
def get_player_stats_page(request: Request):
    retriever = models.get("chatbot_retriever")
    all_players = sorted(retriever.get_all_player_names())[:500] if retriever else []
    return templates.TemplateResponse("player_stats.html", {"request": request, "players": all_players})

@app.post("/get-team-stats", response_class=HTMLResponse)
def get_team_stats_chart(request: Request, team_name: str = Form(...)):
    stats = models["stats_engine"].get_team_stats(team_name)
    if not stats: return "<div class='text-red-400 p-4'>No data found.</div>"
    return templates.TemplateResponse("partials/team_charts.html", {"request": request, "team": team_name, "stats": stats})

@app.post("/get-player-stats", response_class=HTMLResponse)
def get_player_stats_chart(request: Request, player_name: str = Form(...)):
    stats = models["stats_engine"].get_player_stats(player_name)
    if not stats: return "<div class='text-red-400 p-4'>No data found.</div>"
    return templates.TemplateResponse("partials/player_charts.html", {"request": request, "player": player_name, "stats": stats})

# --- 8. GEMINI AI ---
@app.get("/get-gemini-analysis", response_class=HTMLResponse)
async def handle_gemini_analysis(request: Request, batting_team: str = Query(...), bowling_team: str = Query(...), batting_win_prob: float = Query(...), bowling_win_prob: float = Query(...), city: str = Query(...), target: float = Query(...), runs_left: float = Query(...), balls_left: float = Query(...), wickets_left: float = Query(...), rrr: float = Query(...)):
    winning_team = batting_team if batting_win_prob > bowling_win_prob else bowling_team
    win_prob = max(batting_win_prob, bowling_win_prob)
    analysis_prompt = (f"Match: {batting_team} vs {bowling_team} at {city}. Target: {target}, Runs Left: {runs_left}, Balls: {balls_left}, Wickets: {wickets_left}, RRR: {rrr}. Model: {winning_team} wins ({win_prob}%). Explain why in 1 sentence.")
    analysis_text = await get_gemini_analysis(analysis_prompt)
    return HTMLResponse(content=templates.get_template("partials/analysis_result.html").render({"request": request, "analysis_text": analysis_text}))