# backend/train.py

import pandas as pd
import numpy as np
import joblib  # <-- CHANGED: Using joblib instead of pickle
import os
import traceback
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# --- Global Settings ---
MODEL_FILE = 'pipe.joblib'  # <-- CHANGED: Extension to .joblib
DATA_DIR = './data'
MATCHES_FILE = os.path.join(DATA_DIR, "OGDATA.csv")
DELIVERIES_FILE = os.path.join(DATA_DIR, "deliveries.csv")
PLAYER_RATINGS_FILE = os.path.join(DATA_DIR, "player_ratings.pkl")
TEAMS_LIST_FILE = 'teams_list.pkl'
CITIES_LIST_FILE = 'cities_list.pkl'

# --- Configuration for 2025 ---
ACTIVE_TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Titans',
    'Pune Warriors': 'Rising Pune Supergiant'
}

def train_and_save_model():
    """
    Trains the Win Predictor using XGBoost and saves using Joblib.
    """
    global TEAMS, CITIES

    print("--- Training Win Predictor (XGBoost) ---")

    try:
        # 1. LOAD DATA
        if not os.path.exists(MATCHES_FILE) or not os.path.exists(DELIVERIES_FILE):
            print(f"ERROR: Data files not found in {DATA_DIR}"); return None

        match = pd.read_csv(MATCHES_FILE)
        delivery = pd.read_csv(DELIVERIES_FILE)
        
        if not os.path.exists(PLAYER_RATINGS_FILE):
            print(f"ERROR: '{PLAYER_RATINGS_FILE}' not found. Run 'build_squads.py' first!"); return None
            
        with open(PLAYER_RATINGS_FILE, 'rb') as f:
            import pickle # Keep pickle just for loading the ratings dict
            player_ratings = pickle.load(f)
            print(f"Loaded ratings for {len(player_ratings)} players.")

        # 2. DATA CLEANING & NORMALIZATION
        bat_col = 'batter' if 'batter' in delivery.columns else 'batsman'
        bowl_col = 'bowler'
        
        match['team1'] = match['team1'].replace(TEAM_MAPPING)
        match['team2'] = match['team2'].replace(TEAM_MAPPING)
        match['winner'] = match['winner'].replace(TEAM_MAPPING)
        delivery['batting_team'] = delivery['batting_team'].replace(TEAM_MAPPING)
        delivery['bowling_team'] = delivery['bowling_team'].replace(TEAM_MAPPING)

        match = match[match['team1'].isin(ACTIVE_TEAMS)]
        match = match[match['team2'].isin(ACTIVE_TEAMS)]
        
        valid_match_ids = match['id'].unique()
        delivery = delivery[delivery['match_id'].isin(valid_match_ids)]

        match['city'] = match['city'].replace({'Bangalore': 'Bengaluru'}).fillna('Unknown')
        CITIES = sorted(match['city'].unique().tolist())
        TEAMS = sorted(ACTIVE_TEAMS)

        # 3. FEATURE ENGINEERING
        print("Calculating match situation metrics...")
        
        total_score_df = delivery[delivery['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
        match_df = match.merge(total_score_df[['match_id', 'total_runs']].rename(columns={'total_runs':'target_runs'}),
                               left_on='id', right_on='match_id')
        
        match_df = match_df[['match_id', 'city', 'winner', 'target_runs']]
        delivery_df = match_df.merge(delivery, on='match_id')
        delivery_df = delivery_df[delivery_df['inning'] == 2].copy()
        
        total_runs_col = 'total_runs_y' if 'total_runs_y' in delivery_df.columns else 'total_runs'
        delivery_df['current_score'] = delivery_df.groupby('match_id')[total_runs_col].cumsum()
        delivery_df['runs_left'] = delivery_df['target_runs'] - delivery_df['current_score']
        
        delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])
        delivery_df['balls_left'] = delivery_df['balls_left'].apply(lambda x: max(0, x))
        
        delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
        delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 1 if str(x) != "0" else 0)
        delivery_df['wickets_fallen'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
        delivery_df['wickets_left'] = 10 - delivery_df['wickets_fallen']
        
        balls_bowled = 120 - delivery_df['balls_left']
        delivery_df['crr'] = np.where(balls_bowled > 0, (delivery_df['current_score']*6)/balls_bowled, 0)
        delivery_df['rrr'] = np.where(delivery_df['balls_left'] > 0, (delivery_df['runs_left']*6)/delivery_df['balls_left'], 999)
        delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)

        # 4. TEAM STRENGTH
        print("Calculating historical team strengths...")
        match_strength_map = {}
        deliveries_grouped = delivery.groupby('match_id')

        for mid, match_delivery_data in deliveries_grouped:
            batters = match_delivery_data[bat_col].unique()
            bowlers = match_delivery_data[bowl_col].unique()
            
            def calculate_strength(player_list):
                return sum([player_ratings.get(p.strip(), 5.0) for p in player_list])

            match_strength_map[mid] = {
                match_delivery_data.iloc[0]['batting_team']: calculate_strength(batters),
                match_delivery_data.iloc[0]['bowling_team']: calculate_strength(bowlers)
            }

        def get_row_strength(row, is_batting):
            m_data = match_strength_map.get(row['match_id'], {})
            team_name = row['batting_team'] if is_batting else row['bowling_team']
            return m_data.get(team_name, 50.0)

        delivery_df['batting_strength'] = delivery_df.apply(lambda x: get_row_strength(x, True), axis=1)
        delivery_df['bowling_strength'] = delivery_df.apply(lambda x: get_row_strength(x, False), axis=1)

        # 5. TRAINING
        final_df = delivery_df[[
            'batting_strength', 'bowling_strength', 'city', 
            'runs_left', 'balls_left', 'wickets_left', 'target_runs', 'crr', 'rrr', 'result'
        ]].copy()
        
        final_df.rename(columns={'target_runs': 'total_runs_x'}, inplace=True)
        final_df.dropna(inplace=True)
        final_df = final_df[final_df['balls_left'] != 0]
        
        final_df['city'] = final_df['city'].apply(lambda x: x if x in CITIES else 'Other')
        if 'Other' not in CITIES: CITIES.append('Other')

        X = final_df.drop('result', axis=1)
        y = final_df['result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        trf = ColumnTransformer([
            ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['city'])
        ], remainder='passthrough')

        # REMOVED 'use_label_encoder' to fix warning
        pipe = Pipeline(steps=[
            ('step1', trf),
            ('step2', XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=6, eval_metric='logloss'))
        ])

        print("Fitting XGBoost model...")
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"--- Model Trained! Accuracy: {acc*100:.2f}% ---")

        # 6. SAVE ARTIFACTS
        joblib.dump(pipe, MODEL_FILE) # <-- USING JOBLIB
        
        import pickle 
        with open(TEAMS_LIST_FILE, 'wb') as f: pickle.dump(TEAMS, f)
        with open(CITIES_LIST_FILE, 'wb') as f: pickle.dump(CITIES, f)
        
        print(f"✅ XGBoost Model saved to {MODEL_FILE}")
        return pipe

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_and_save_model()