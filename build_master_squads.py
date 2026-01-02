# backend/build_master_squads.py
import pandas as pd
import os
import pickle

# --- CONFIG ---
OUTPUT_FILE = "data/master_squads.csv"
MATCHES_FILE = "data/matches.csv"
DELIVERIES_FILE = "data/deliveries.csv"
PLAYER_TEAMS_PKL = "data/player_teams.pkl" 

def build_master_csv():
    print(f"--- Building Master Squads (Force Merge Mode) ---")
    
    if not (os.path.exists(MATCHES_FILE) and os.path.exists(DELIVERIES_FILE)):
        print("❌ Error: CSV files not found.")
        return

    # 1. Load Data
    print("Loading CSVs...")
    matches = pd.read_csv(MATCHES_FILE)
    deliveries = pd.read_csv(DELIVERIES_FILE)
    
    # Clean headers
    matches.columns = [c.strip() for c in matches.columns]
    deliveries.columns = [c.strip() for c in deliveries.columns]
    
    # 2. Sort both files to align them
    # We assume the 1st match in matches.csv corresponds to Match ID 1 in deliveries.csv
    
    # Sort matches by date or ID
    if 'date' in matches.columns:
        matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
        matches.sort_values(by=['date', 'id'], inplace=True)
    else:
        matches.sort_values(by=['id'], inplace=True)
        
    # Get columns
    year_col = 'season' if 'season' in matches.columns else 'year'
    d_id_col = 'match_id' if 'match_id' in deliveries.columns else 'id'
    
    # Get unique delivery match IDs (sorted numerically)
    # This handles IDs like 1, 2, 3...
    delivery_ids = sorted(deliveries[d_id_col].unique(), key=lambda x: int(x) if str(x).replace('.','').isdigit() else str(x))
    
    print(f"Found {len(matches)} matches in matches.csv")
    print(f"Found {len(delivery_ids)} matches in deliveries.csv")
    
    # 3. Create the Mapping (Force Link)
    # Map the i-th delivery ID to the i-th match year
    limit = min(len(matches), len(delivery_ids))
    id_to_year = {}
    
    for i in range(limit):
        d_id = delivery_ids[i]
        m_row = matches.iloc[i]
        year = m_row.get(year_col)
        # Fallback to date if season is missing
        if pd.isna(year) and 'date' in m_row and pd.notna(m_row['date']):
            year = m_row['date'].year
            
        id_to_year[d_id] = int(year) if pd.notna(year) else 0
        
    print(f"✅ Mapped {len(id_to_year)} matches successfully.")

    # 4. Extract Squads
    all_records = []
    
    bat_col = 'batter' if 'batter' in deliveries.columns else 'batsman'
    
    # Process Batters
    print("Extracting players...")
    temp_df = deliveries[[d_id_col, 'batting_team', bat_col]].drop_duplicates()
    for _, row in temp_df.iterrows():
        yr = id_to_year.get(row[d_id_col])
        if yr:
            all_records.append({'Year': yr, 'Team': row['batting_team'], 'Player': row[bat_col]})
            
    # Process Bowlers
    if 'bowler' in deliveries.columns:
        temp_df = deliveries[[d_id_col, 'bowling_team', 'bowler']].drop_duplicates()
        for _, row in temp_df.iterrows():
            yr = id_to_year.get(row[d_id_col])
            if yr:
                all_records.append({'Year': yr, 'Team': row['bowling_team'], 'Player': row['bowler']})

    # 5. Add 2025/Latest Data (if available)
    if os.path.exists(PLAYER_TEAMS_PKL):
        try:
            with open(PLAYER_TEAMS_PKL, "rb") as f:
                latest = pickle.load(f)
            # Find max year so we don't overlap
            max_yr = max([x['Year'] for x in all_records]) if all_records else 2024
            target_yr = max_yr + 1
            
            # Map legacy names for current squad
            std_names = {
                'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings',
                'Deccan Chargers': 'Sunrisers Hyderabad', 'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
            }
            
            for player, team in latest.items():
                norm_team = std_names.get(team, team)
                all_records.append({'Year': target_yr, 'Team': norm_team, 'Player': player})
            print(f"✅ Added latest squad data as Year {target_yr}")
        except: pass

    # 6. Save
    if all_records:
        final_df = pd.DataFrame(all_records)
        final_df.drop_duplicates(inplace=True)
        final_df.sort_values(by=['Year', 'Team', 'Player'], ascending=[False, True, True], inplace=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ SUCCESS! Saved to {OUTPUT_FILE}")
        print(final_df.head())
    else:
        print("❌ Failed to build squad list.")

if __name__ == "__main__":
    build_master_csv()