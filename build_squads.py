# backend/build_squads.py
import pandas as pd
import pickle
import os

# --- CONFIG ---
ACTIVE_TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiant': 'Rising Pune Supergiant', # Keep to map, then filter
    'Gujarat Lions': 'Gujarat Lions' # Keep to map, then filter
}

def build_squad_data():
    print("--- Building Player Ratings & Squads ---")
    
    # 1. Load Data
    try:
        p_stats = pd.read_csv("data/player_stats.csv")
        matches = pd.read_csv("data/matches.csv") 
        deliveries = pd.read_csv("data/deliveries.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- FIX: Detect Column Names ---
    print(f"Deliveries columns: {deliveries.columns.tolist()[:5]}...") # Debug print
    
    # Check for 'batter' vs 'batsman'
    bat_col = 'batter' if 'batter' in deliveries.columns else 'batsman'
    bowl_col = 'bowler' # This is usually consistent, but good to know
    
    print(f"Using batter column: '{bat_col}'")

    # 2. Generate Player Ratings (0-10 Scale)
    p_stats.fillna(0, inplace=True)
    
    ratings_dict = {}
    for _, row in p_stats.iterrows():
        # Heuristic score
        bat_score = (row.get('Runs', 0) * 0.05) + (row.get('Avg', 0) * 1.5)
        bowl_score = (row.get('Wickets_taken', 0) * 10) + (50 / (row.get('Economy', 10) + 1))
        
        total_score = max(bat_score, bowl_score) 
        rating = min(10.0, total_score / 30.0) 
        ratings_dict[row['Player'].strip()] = round(rating, 2)

    # 3. Find "Current Team" for every player
    matches['id'] = pd.to_numeric(matches['id'], errors='coerce')
    matches.sort_values('id', ascending=True, inplace=True)
    
    player_team_map = {}
    
    print("Scanning deliveries to find latest teams...")
    
    # Group by player and get max match_id to find their latest appearance
    # Batter's latest team
    latest_bat = deliveries.groupby(bat_col).agg({'match_id': 'max', 'batting_team': 'last'}).reset_index()
    # Bowler's latest team
    latest_bowl = deliveries.groupby(bowl_col).agg({'match_id': 'max', 'bowling_team': 'last'}).reset_index()
    
    # Merge findings
    for _, row in latest_bat.iterrows():
        team = row['batting_team']
        team = TEAM_MAPPING.get(team, team) # Normalize
        if team in ACTIVE_TEAMS:
            player_team_map[row[bat_col]] = team
            
    for _, row in latest_bowl.iterrows():
        team = row['bowling_team']
        team = TEAM_MAPPING.get(team, team) # Normalize
        if team in ACTIVE_TEAMS:
            player_team_map[row[bowl_col]] = team

    # 4. Save the artifacts
    print(f"Identified {len(player_team_map)} active players in 10 teams.")
    
    # Ensure data dir exists
    if not os.path.exists("data"):
        os.makedirs("data")

    with open("data/player_ratings.pkl", "wb") as f:
        pickle.dump(ratings_dict, f)
        
    with open("data/player_teams.pkl", "wb") as f:
        pickle.dump(player_team_map, f)
        
    print("✅ Created 'data/player_ratings.pkl' and 'data/player_teams.pkl'")

if __name__ == "__main__":
    build_squad_data()