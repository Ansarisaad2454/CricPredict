# backend/build_ratings.py
import pandas as pd
import pickle

def calculate_player_ratings():
    # 1. Load Data
    df = pd.read_csv("data/player_stats.csv")
    
    # 2. Normalize Columns (Handle missing values)
    df.fillna(0, inplace=True)
    
    # 3. Define Rating Formula
    # This is a heuristic formula. You can tweak the weights.
    # Batting Score: Runs + (Strike Rate / 2) + (Avg * 10)
    df['bat_rating'] = (df['Runs'] * 0.05) + (df['SR'] * 0.5) + (df['Avg'] * 2)
    
    # Bowling Score: (Wickets * 20) + (Dots * 0.5) + (Economy * -10 to penalize high economy)
    # We invert Economy (12 - Economy) so lower is better.
    df['bowl_rating'] = (df['Wickets_taken'] * 15) + ((12 - df['Economy']) * 5)
    
    # Combined Rating (Take the higher skill or sum them for all-rounders)
    df['total_rating'] = df[['bat_rating', 'bowl_rating']].max(axis=1)
    
    # Normalize to 0-10 scale for easier interpretability
    max_score = df['total_rating'].max()
    df['final_score'] = (df['total_rating'] / max_score) * 10
    
    # 4. Create Dictionary: { "Virat Kohli": 9.8, "Jasprit Bumrah": 9.5, ... }
    rating_dict = pd.Series(df.final_score.values, index=df.Player).to_dict()
    
    # 5. Save
    with open("player_ratings.pkl", "wb") as f:
        pickle.dump(rating_dict, f)
        
    print(f"Ratings generated for {len(rating_dict)} players.")
    return rating_dict

if __name__ == "__main__":
    calculate_player_ratings()