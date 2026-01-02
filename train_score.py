# backend/train_score.py

import pandas as pd
import numpy as np
import joblib
import pickle
import os
import traceback
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --- Global Settings ---
MODEL_FILE = 'score_pipe.joblib'
DATA_DIR = './data'
IPL_FILE = os.path.join(DATA_DIR, "IPL.csv")
PLAYER_RATINGS_FILE = os.path.join(DATA_DIR, "player_ratings.pkl")
VENUE_STATS_FILE = os.path.join(DATA_DIR, "venue_stats.pkl")

# --- Configuration ---
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
    'Pune Warriors': 'Rising Pune Supergiant',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}

def train_score_predictor():
    print("--- Training AGGRESSIVE High-Score Predictor (2018-2025) ---")
    
    try:
        # 1. LOAD DATA
        if not os.path.exists(IPL_FILE):
            print(f"ERROR: '{IPL_FILE}' not found."); return

        print(f"Loading {IPL_FILE}...")
        df = pd.read_csv(IPL_FILE, low_memory=False)

        if not os.path.exists(PLAYER_RATINGS_FILE):
            print(f"ERROR: Ratings file missing. Run build_squads.py first."); return
            
        with open(PLAYER_RATINGS_FILE, 'rb') as f:
            player_ratings = pickle.load(f)

        # 2. PREPROCESS COLUMNS
        if 'innings' in df.columns: df.rename(columns={'innings': 'inning'}, inplace=True)
        if 'runs_total' in df.columns: df.rename(columns={'runs_total': 'total_runs'}, inplace=True)
        if 'player_out' in df.columns: df.rename(columns={'player_out': 'player_dismissed'}, inplace=True)
            
        if 'year' not in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year

        # 3. AGGRESSIVE FILTERING
        # Drop old teams
        df['batting_team'] = df['batting_team'].replace(TEAM_MAPPING)
        df['bowling_team'] = df['bowling_team'].replace(TEAM_MAPPING)
        df = df[df['batting_team'].isin(ACTIVE_TEAMS)]
        df = df[df['bowling_team'].isin(ACTIVE_TEAMS)]

        # --- CRITICAL CHANGE: Drop Pre-2018 Data ---
        # Old cricket drags the prediction down. We strictly want modern trends.
        print(f"Original Data Size: {len(df)}")
        df = df[df['year'] >= 2018]
        print(f"Modern Data Size (2018+): {len(df)}")

        # City Normalization
        df['city'] = df['city'].replace({'Bangalore': 'Bengaluru'}).fillna('Unknown')

        # 4. CALCULATE VENUE AVERAGES (Strictly Modern)
        print("Calculating venue biases based on 2023-2025 High Scoring Era...")
        
        match_groups = df[df['inning'] == 1].groupby(['match_id', 'year'])
        match_totals = match_groups['total_runs'].sum().reset_index()
        match_cities = match_groups['city'].first().reset_index()
        
        venue_df = match_totals.merge(match_cities[['match_id', 'city']], on='match_id')
        
        # Calculate stats ONLY on 2023+ data (Impact Player Era)
        recent_venue_df = venue_df[venue_df['year'] >= 2023]
        
        if len(recent_venue_df) > 50:
            print("Using strict 2023+ Venue Stats (Batting Highways)")
            venue_stats = recent_venue_df.groupby('city')['total_runs'].mean().to_dict()
        else:
            print("Not enough 2023 data, falling back to 2018+")
            venue_stats = venue_df.groupby('city')['total_runs'].mean().to_dict()

        # Manual Boost for known highways if data is lagging
        if 'Bengaluru' in venue_stats: venue_stats['Bengaluru'] = max(venue_stats['Bengaluru'], 200)
        if 'Hyderabad' in venue_stats: venue_stats['Hyderabad'] = max(venue_stats['Hyderabad'], 200)
        if 'Mumbai' in venue_stats: venue_stats['Mumbai'] = max(venue_stats['Mumbai'], 190)
        if 'Kolkata' in venue_stats: venue_stats['Kolkata'] = max(venue_stats['Kolkata'], 190)

        with open(VENUE_STATS_FILE, 'wb') as f:
            pickle.dump(venue_stats, f)

        # 5. PREPARE TRAINING DATA
        train_df = df[df['inning'] == 1].copy()
        
        # Map total scores
        match_score_map = venue_df.set_index('match_id')['total_runs'].to_dict()
        train_df['total_runs_x'] = train_df['match_id'].map(match_score_map)
        train_df.dropna(subset=['total_runs_x'], inplace=True)

        # Map Venue Avg
        train_df['venue_avg'] = train_df['city'].map(venue_stats).fillna(175.0)

        # Features
        train_df['current_score'] = train_df.groupby('match_id')['total_runs'].cumsum()
        train_df['is_wicket'] = train_df['player_dismissed'].notna().astype(int)
        train_df['wickets'] = train_df.groupby('match_id')['is_wicket'].cumsum()
        train_df['overs'] = train_df['over'] + (train_df['ball'] / 6)

        # Rolling Features
        groups = train_df.groupby('match_id')
        train_df['runs_last_5'] = groups['total_runs'].rolling(window=30, min_periods=1).sum().reset_index(0, drop=True).fillna(0)
        train_df['wickets_last_5'] = groups['is_wicket'].rolling(window=30, min_periods=1).sum().reset_index(0, drop=True).fillna(0)
        
        # Projected Score Simple
        train_df['projected_score_simple'] = (train_df['current_score'] / train_df['overs']) * 20
        train_df['projected_score_simple'] = train_df['projected_score_simple'].replace([np.inf, -np.inf], 0).fillna(0)

        # 6. TEAM STRENGTH
        print("Calculating team strengths...")
        match_strength_map = {}
        for mid, grp in train_df.groupby('match_id'):
            batters = grp['batter'].unique()
            bowlers = grp['bowler'].unique()
            s_bat = sum([player_ratings.get(p.strip(), 5.0) for p in batters])
            s_bowl = sum([player_ratings.get(p.strip(), 5.0) for p in bowlers])
            match_strength_map[mid] = {'bat': s_bat, 'bowl': s_bowl}

        train_df['batting_strength'] = train_df['match_id'].apply(lambda m: match_strength_map.get(m, {}).get('bat', 50))
        train_df['bowling_strength'] = train_df['match_id'].apply(lambda m: match_strength_map.get(m, {}).get('bowl', 50))

        # 7. FINAL DATASET
        final_df = train_df[[
            'batting_strength', 'bowling_strength', 'venue_avg', 
            'current_score', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 
            'projected_score_simple', 'year', 'total_runs_x'
        ]].copy()

        final_df = final_df[final_df['overs'] >= 5]
        
        # --- AGGRESSIVE WEIGHTING ---
        # 2024-2025 matches are 10x more important than 2018 matches
        def get_weight(year):
            if year >= 2024: return 10.0
            if year == 2023: return 5.0
            if year >= 2021: return 2.0
            return 1.0

        sample_weights = final_df['year'].apply(get_weight)

        X = final_df.drop(['total_runs_x', 'year'], axis=1)
        y = final_df['total_runs_x']

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.1, random_state=42
        )

        # 8. PIPELINE (XGBoost Tuned for High Variance)
        trf = ColumnTransformer([('pass', 'passthrough', X.columns)], remainder='drop')

        pipe = Pipeline(steps=[
            ('step1', trf),
            ('step2', XGBRegressor(
                n_estimators=2000, 
                learning_rate=0.01, 
                max_depth=8, 
                min_child_weight=3, # Allows fitting to outliers (high scores)
                eval_metric='mae',
                n_jobs=1
            ))
        ])

        print("Fitting XGBoost Model (Aggressive Mode)...")
        pipe.fit(X_train, y_train, step2__sample_weight=w_train)
        
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"--- Model Trained! MAE: {mae:.2f} runs ---")

        joblib.dump(pipe, MODEL_FILE)
        print(f"✅ Saved to {MODEL_FILE}")

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    train_score_predictor()