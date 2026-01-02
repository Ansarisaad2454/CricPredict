# backend/retriever.py
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz
import numpy as np
import warnings
import re
import pickle
from typing import List, Dict, Any, Optional, Tuple, Set

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Helper to find the best column name match."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # Fallback fuzzy match
    for col in df.columns:
        for cand in candidates:
            if fuzz.partial_ratio(col.lower(), cand.lower()) > 85:
                return col
    return None

class IPLRetriever:
    def __init__(self, csv_path: str = "data/matches.csv", 
                 hattrick_path: str = "data/hatrick.csv", 
                 player_stats_path: str = "data/player_stats.csv",
                 master_squads_path: str = "data/master_squads.csv", 
                 deliveries_path: str = "data/deliveries.csv",
                 player_teams_path: str = "data/player_teams.pkl",
                 model_name: str = "all-MiniLM-L6-v2"):
        
        print("--- Initializing IPLRetriever ---")

        # 1. LOAD MASTER SQUADS (The fix)
        self.squads_df = pd.DataFrame()
        if Path(master_squads_path).exists():
            self.squads_df = pd.read_csv(master_squads_path)
            self.squads_df['Team'] = self.squads_df['Team'].astype(str)
            self.squads_df['Player'] = self.squads_df['Player'].astype(str)
        else:
            print(f"⚠️ {master_squads_path} not found. Please run 'python backend/build_master_squads.py'")
        
        # 1. Load Matches (Essential)
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Data file not found at {csv_path}")

        self.df = pd.read_csv(csv_file)
        self.df.columns = [c.strip() for c in self.df.columns]

        # 2. Load Hattricks
        hattrick_file = Path(hattrick_path)
        if hattrick_file.exists():
            try:
                self.hattricks_df = pd.read_csv(hattrick_file)
                self.hattricks_df.columns = ['bowler', 'match', 'venue', 'wickets', 'year']
            except: self.hattricks_df = pd.DataFrame()
        else:
            self.hattricks_df = pd.DataFrame()
            
        # 3. Load Player Career Stats
        self.player_stats_df = None
        player_stats_file = Path(player_stats_path)
        if player_stats_file.exists():
            try:
                self.player_stats_df = pd.read_csv(player_stats_file)
                if 'Player' in self.player_stats_df.columns:
                    self.player_stats_df['Player'] = self.player_stats_df['Player'].astype(str)
            except Exception as e:
                print(f"Error loading player_stats.csv: {e}")
                self.player_stats_df = None
        
        # 4. Load Deliveries (for Year-Wise Squads)
        self.deliveries_df = pd.DataFrame()
        if Path(deliveries_path).exists():
            print(f"Loading {deliveries_path} for squad building...")
            try:
                self.deliveries_df = pd.read_csv(deliveries_path)
                self.deliveries_df.columns = [c.strip() for c in self.deliveries_df.columns]
                # Ensure match_id is string
                if 'match_id' in self.deliveries_df.columns:
                    self.deliveries_df['match_id'] = self.deliveries_df['match_id'].astype(str)
            except Exception as e:
                print(f"Error loading deliveries.csv: {e}")
        
        # 5. Load Latest Player Teams (for 2025/Current Squads)
        self.latest_player_teams = {}
        if Path(player_teams_path).exists():
            try:
                with open(player_teams_path, "rb") as f:
                    self.latest_player_teams = pickle.load(f)
            except Exception as e: print(f"Error loading player_teams.pkl: {e}")

        # --- Detect Columns in Matches ---
        self.col_date = find_col(self.df, ["date", "match_date"])
        self.col_year = find_col(self.df, ["season", "year"])
        self.col_team1 = find_col(self.df, ["team1", "team_1"])
        self.col_team2 = find_col(self.df, ["team2", "team_2"])
        self.col_winner = find_col(self.df, ["winner", "match_winner"])
        self.col_pom = find_col(self.df, ["player_of_match", "man of the match"])
        self.col_stage = find_col(self.df, ["match_type", "stage", "final"])
        self.col_ump1 = find_col(self.df, ["umpire1", "umpire_1"])
        self.col_ump2 = find_col(self.df, ["umpire2", "umpire_2"])
        self.col_toss_win = find_col(self.df, ["toss_winner"])
        self.col_toss_dec = find_col(self.df, ["toss_decision"])
        self.col_result = find_col(self.df, ["result"])
        self.col_margin = find_col(self.df, ["result_margin"])
        self.col_method = find_col(self.df, ["method"])
        self.col_super_over = find_col(self.df, ["super_over"])
        self.col_venue = find_col(self.df, ["venue"])
        self.col_city = find_col(self.df, ["city"]) 
        
        self.df = self._normalize_dataframe(self.df)
        # Ensure ID is string for joining
        if 'id' in self.df.columns:
            self.df['id'] = self.df['id'].astype(str)

        # --- Standard Caches ---
        team1_set = set(self.df["team1"].dropna().tolist())
        team2_set = set(self.df["team2"].dropna().tolist())
        self.team_full_names: List[str] = sorted(list(team1_set.union(team2_set)))
        self.team_full_names = [t for t in self.team_full_names if t] 
        self.team_map: Dict[str, str] = self._build_team_map()
        self.all_venue_names: List[str] = self.df['venue'].dropna().unique().tolist()
        self.venue_map: Dict[str, str] = {v.lower().replace(',', ''): v for v in self.all_venue_names}
        self.city_to_venues_map: Dict[str, List[str]] = self.df.dropna(
            subset=['city', 'venue']).groupby('city')['venue'].unique().apply(list).to_dict()
        
        # --- Build Player List (From all sources) ---
        self.all_player_names = self.df['player_of_match'].dropna().unique().tolist()
        
        if not self.deliveries_df.empty:
            bat_col = 'batter' if 'batter' in self.deliveries_df.columns else 'batsman'
            if bat_col in self.deliveries_df.columns:
                self.all_player_names.extend(self.deliveries_df[bat_col].unique().tolist())
            if 'bowler' in self.deliveries_df.columns:
                self.all_player_names.extend(self.deliveries_df['bowler'].unique().tolist())
                
        if self.player_stats_df is not None:
             self.all_player_names.extend(self.player_stats_df['Player'].unique().tolist())
             
        self.all_player_names.extend(self.latest_player_teams.keys())

        # Deduplicate list
        self.all_player_names = list(set([p for p in self.all_player_names if isinstance(p, str)]))
        self.player_map_lower: Dict[str, str] = {p.lower(): p for p in self.all_player_names}

        # --- Build Squads from CSV Data (Accurate History) ---
        self.season_squads = self._build_historical_squads_map()

        self.season_finals_data: Dict[int, dict] = self._compute_season_finals()
        self.season_winners: Dict[int, str] = self._compute_season_winners()
        self.all_time_trophy_counts: pd.Series = pd.Series(list(self.season_winners.values())).value_counts()
        self._build_match_index()

        # --- Load Additional Record Files ---
        self.df_orange_cap = self._load_cap_data("data/orange_cap.csv", ['year', 'player_team', 'matches', 'runs'])
        self.df_purple_cap = self._load_cap_data("data/purple_cap.csv", ['year', 'player_team', 'matches', 'wickets'])
        
        self.df_bat_records = self._load_record_data(
            "data/bat_records.csv", 
            usecols=[1, 2, 3], 
            new_cols=['player', 'team', 'runs']
        )
        self.df_bowl_records = self._load_record_data(
            "data/bowl_records.csv", 
            usecols=[1, 2, 3], 
            new_cols=['player', 'team', 'wickets']
        )
        
        self.most_orange_caps = self._calculate_most_caps(self.df_orange_cap)
        self.most_purple_caps = self._calculate_most_caps(self.df_purple_cap)

    # -------------------------------------------------------------------------
    #  NEW: Squad Building Logic (Accurate History + Latest)
    # -------------------------------------------------------------------------
    def _build_historical_squads_map(self) -> Dict[tuple, List[str]]:
        """
        Builds a map of (Team, Year) -> [List of Players] using deliveries.csv.
        This provides 100% accurate data for past seasons (e.g. 2017).
        """
        if self.deliveries_df.empty or self.df.empty:
            return {}
        
        if 'id' not in self.df.columns or self.col_year not in self.df.columns:
            return {}
            
        squads = {}
        
        # 1. Map Match ID -> Year
        match_id_to_year = self.df.set_index('id')[self.col_year].to_dict()
        
        bat_col = 'batter' if 'batter' in self.deliveries_df.columns else 'batsman'
        
        # Optimize by selecting only needed columns and dropping duplicates
        relevant_cols = ['match_id', 'batting_team', 'bowling_team', bat_col, 'bowler']
        # Filter for columns that actually exist
        available_cols = [c for c in relevant_cols if c in self.deliveries_df.columns]
        
        df_lite = self.deliveries_df[available_cols].drop_duplicates()
        
        for _, row in df_lite.iterrows():
            mid = str(row['match_id'])
            year = match_id_to_year.get(mid)
            
            if not year: continue
            
            # Process Batter
            if bat_col in row and pd.notna(row[bat_col]):
                t1 = self.match_team_from_query(row['batting_team'])
                if t1:
                    key = (t1, int(year))
                    if key not in squads: squads[key] = set()
                    squads[key].add(row[bat_col])
                    
            # Process Bowler
            if 'bowler' in row and pd.notna(row['bowler']):
                t2 = self.match_team_from_query(row['bowling_team'])
                if t2:
                    key = (t2, int(year))
                    if key not in squads: squads[key] = set()
                    squads[key].add(row['bowler'])
                    
        return {k: sorted(list(v)) for k, v in squads.items()}

    def get_team_squad(self, team_name: str, year: Optional[int] = None) -> Dict[str, Any]:
        if self.squads_df.empty: return None
        
        full_team = self.match_team_from_query(team_name)
        if not full_team: return None
        
        # Legacy map
        legacy = {'Delhi Capitals': 'Delhi Daredevils', 'Punjab Kings': 'Kings XI Punjab', 'Sunrisers Hyderabad': 'Deccan Chargers', 'Gujarat Titans': 'Gujarat Lions'}
        
        # Filter by Team
        team_df = self.squads_df[self.squads_df['Team'] == full_team]
        
        # If year provided
        if year:
            y_df = team_df[team_df['Year'] == int(year)]
            # If empty, check legacy name
            if y_df.empty and full_team in legacy:
                team_df = self.squads_df[self.squads_df['Team'] == legacy[full_team]]
                y_df = team_df[team_df['Year'] == int(year)]
                if not y_df.empty: full_team = legacy[full_team] # Update name
            
            final_p = sorted(y_df['Player'].unique().tolist())
            return {"team": full_team, "year": int(year), "players": final_p}
        
        # If NO year, get LATEST year
        else:
            if team_df.empty: return None
            mx_yr = team_df['Year'].max()
            y_df = team_df[team_df['Year'] == mx_yr]
            final_p = sorted(y_df['Player'].unique().tolist())
            return {"team": full_team, "year": int(mx_yr), "players": final_p}

    # --- Helpers ---
    def match_team_from_query(self, q):
        if not q: return None
        q = q.lower().strip()
        codes = {"rcb":"Royal Challengers Bangalore", "csk":"Chennai Super Kings", "mi":"Mumbai Indians", "kkr":"Kolkata Knight Riders", "dc":"Delhi Capitals", "srh":"Sunrisers Hyderabad", "pbks":"Punjab Kings", "rr":"Rajasthan Royals", "gt":"Gujarat Titans", "lsg":"Lucknow Super Giants"}
        if q in codes: return codes[q]
        if q in self.team_map: return self.team_map[q]
        match = process.extractOne(q, self.team_map.keys(), score_cutoff=85)
        return self.team_map[match[0]] if match else None

    def get_player_career_stats(self, p_name):
        if self.player_stats_df is None: return None
        choices = self.player_stats_df['Player'].astype(str).tolist()
        match = process.extractOne(p_name, choices, scorer=fuzz.token_sort_ratio, score_cutoff=70)
        if not match: return None
        row = self.player_stats_df[self.player_stats_df['Player'] == match[0]].iloc[0]
        return {
            "name": match[0], "matches": int(row.get('Matches', 0)),
            "runs": int(row.get('Runs', 0)), "avg": float(row.get('Avg', 0.0)),
            "sr": float(row.get('SR', 0.0)), "wickets": int(row.get('Wickets_taken', 0)),
            "economy": float(row.get('Economy', 0.0))
        }

    def _load_csv(self, f, u, c):
        try: 
            df = pd.read_csv(f, header=None, skiprows=1, usecols=u)
            df.columns = c; return df
        except: return None
    def _load_cap(self, f):
        try: return pd.read_csv(f, header=None, skiprows=1, names=['year','pt','m','val']).set_index('year')
        except: return None
    def _calc_caps(self, df): return df['player'].value_counts() if df is not None else None
    def _calc_winners(self):
        w = {}
        if not self.df.empty and 'season' in self.df.columns and 'winner' in self.df.columns:
            for yr in self.df['season'].unique():
                ydf = self.df[self.df['season'] == yr]
                if not ydf.empty: w[int(yr)] = ydf.iloc[-1]['winner']
        return w

    # -------------------------------------------------------------------------
    #  NEW: Player Career Stats Logic
    # -------------------------------------------------------------------------
    def get_player_career_stats(self, player_name: str) -> Optional[Dict[str, Any]]:
        if self.player_stats_df is None: return None
        
        choices = self.player_stats_df['Player'].tolist()
        match = process.extractOne(player_name, choices, scorer=fuzz.token_sort_ratio, score_cutoff=70)
        
        if not match: return None
        
        row = self.player_stats_df[self.player_stats_df['Player'] == match[0]].iloc[0]
        
        return {
            "name": match[0],
            "matches": int(row.get('Matches', 0)),
            # Batting
            "runs": int(row.get('Runs', 0)),
            "bat_innings": int(row.get('innings_bat', 0)),
            "avg": float(row.get('Avg', 0.0)),
            "sr": float(row.get('SR', 0.0)),
            "100s": int(row.get('Centuries', 0)),
            "50s": int(row.get('Fifties', 0)),
            "4s": int(row.get('Fours', 0)),
            "6s": int(row.get('Sixes', 0)),
            # Bowling
            "wickets": int(row.get('Wickets_taken', 0)),
            "bowl_innings": int(row.get('innings_bowl', 0)),
            "economy": float(row.get('Economy', 0.0)),
            "bowl_avg": float(row.get('Bowling_avg', 0.0)),
            "bowl_sr": float(row.get('Bowling_SR', 0.0))
        }

    # -------------------------------------------------------------------------
    #  Loaders & Helpers (Preserved)
    # -------------------------------------------------------------------------
    def _load_cap_data(self, file_path: str, columns: List[str]) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(file_path, header=None, skiprows=1, names=columns) 
            df['player'] = df['player_team'].astype(str).str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df.dropna(subset=['year', 'player'], inplace=True)
            df['year'] = df['year'].astype(int)
            df.set_index('year', inplace=True)
            return df
        except: return None

    def _load_record_data(self, file_path: str, usecols: List[int], new_cols: List[str]) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(file_path, header=None, skiprows=1, usecols=usecols)
            df.columns = new_cols
            df[new_cols[2]] = pd.to_numeric(df[new_cols[2]], errors='coerce')
            df.dropna(inplace=True)
            return df
        except: return None

    def _calculate_most_caps(self, df_cap: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if df_cap is None: return None
        return df_cap['player'].value_counts()

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        if self.col_year:
            df2["year"] = pd.to_numeric(df2[self.col_year], errors="coerce").astype("Int64")
        elif self.col_date:
            df2[self.col_date] = pd.to_datetime(df2[self.col_date], errors='coerce')
            df2["year"] = df2[self.col_date].dt.year.astype("Int64")
        else:
            df2["year"] = pd.NA

        cols_to_normalize = {
            self.col_team1: "team1", self.col_team2: "team2", self.col_winner: "winner",
            self.col_pom: "player_of_match", self.col_stage: "stage", self.col_ump1: "umpire1",
            self.col_ump2: "umpire2", self.col_toss_win: "toss_winner", self.col_toss_dec: "toss_decision",
            self.col_result: "result", self.col_margin: "result_margin", self.col_date: "date",
            self.col_method: "method", self.col_super_over: "super_over", self.col_venue: "venue",
            self.col_city: "city" 
        }

        for col_name, new_name in cols_to_normalize.items():
            if col_name and col_name in df2.columns:
                if col_name == self.col_date:
                    df2[new_name] = pd.to_datetime(df2[col_name], errors='coerce')
                elif col_name == self.col_margin:
                    df2[new_name] = pd.to_numeric(df2[col_name], errors='coerce').astype("Int64")
                else:
                    df2[new_name] = df2[col_name].fillna("").astype(str).str.strip().str.title()
            else:
                if new_name == "date": df2[new_name] = pd.NaT
                else: df2[new_name] = pd.NA

        if 'city' in df2.columns: df2['city'] = df2['city'].str.lower()
        if 'result' in df2.columns: df2['result'] = df2['result'].replace(['Na', 'N/A'], 'No Result')
        df2['team1'] = df2['team1'].fillna('Unknown Team 1')
        df2['team2'] = df2['team2'].fillna('Unknown Team 2')
        return df2

    def _build_team_map(self) -> Dict[str, str]:
        team_map = {}
        # Standard mappings
        team_map["csk"] = "Chennai Super Kings"; team_map["mi"] = "Mumbai Indians"
        team_map["rcb"] = "Royal Challengers Bangalore"; team_map["kkr"] = "Kolkata Knight Riders"
        team_map["dc"] = "Delhi Capitals"; team_map["dd"] = "Delhi Daredevils"
        team_map["srh"] = "Sunrisers Hyderabad"; team_map["rr"] = "Rajasthan Royals"
        team_map["pbks"] = "Punjab Kings"; team_map["kxip"] = "Kings XI Punjab"
        team_map["gt"] = "Gujarat Titans"; team_map["lsg"] = "Lucknow Super Giants"
        
        for name in self.team_full_names: team_map[name.lower()] = name
        for name in self.team_full_names:
            words = name.split()
            if len(words) > 1:
                abbr = "".join([w[0] for w in words]).lower()
                if abbr not in team_map: team_map[abbr] = name
        return team_map

    def _compute_season_winners(self) -> Dict[int, str]:
        season_winners = {}
        for year, final_data in self.season_finals_data.items():
            if final_data and final_data.get("winner"): 
                season_winners[int(year)] = final_data.get("winner")
        
        # Fallback if finals data missing
        if not season_winners and 'year' in self.df.columns and 'winner' in self.df.columns:
            df_valid = self.df.dropna(subset=["year","winner"])
            for year in df_valid['year'].unique():
                year_df = df_valid[df_valid['year'] == year]
                if 'date' in year_df.columns:
                    last = year_df.loc[year_df['date'].idxmax()]
                    if last['winner']: season_winners[int(year)] = last['winner']
        return season_winners

    def _compute_season_finals(self) -> Dict[int, dict]:
        finals_data = {}
        if 'year' not in self.df.columns: return finals_data
        
        df = self.df.dropna(subset=['year']).copy()
        df['year'] = df['year'].astype(int)
        
        if 'stage' in df.columns:
            final_matches = df[df['stage'].str.contains('Final', case=False, na=False)]
            for _, row in final_matches.iterrows():
                finals_data[int(row['year'])] = row.replace({np.nan: None, pd.NaT: None}).to_dict()
                
        return finals_data

    def _build_match_index(self):
        self.matches_by_key = {}
        for idx, row in self.df.iterrows():
            try: year = int(row["year"]) if pd.notna(row["year"]) else None
            except: year = None
            teams_fs = frozenset([row["team1"], row["team2"]])
            self.matches_by_key.setdefault((year, teams_fs), []).append(row)
            self.matches_by_key.setdefault((None, teams_fs), []).append(row)

    # -------------------------------------------------------------------------
    #  Public Matchers
    # -------------------------------------------------------------------------
    def match_team_from_query(self, query_team: str) -> Optional[str]:
        query_lower = query_team.lower().strip()
        if query_lower in self.team_map: return self.team_map[query_lower]
        match = process.extractOne(query_lower, self.team_map.keys(), score_cutoff=85)
        if match: return self.team_map[match[0]]
        return None

    def match_venue_from_query(self, query_venue: str) -> Optional[str]:
        query_lower = query_venue.lower().strip()
        if not query_lower: return None
        if query_lower in self.city_to_venues_map:
            if len(self.city_to_venues_map[query_lower]) == 1:
                return self.city_to_venues_map[query_lower][0]
        if query_lower in self.venue_map: return self.venue_map[query_lower]
        match = process.extractOne(query_lower, self.venue_map.keys(), scorer=fuzz.partial_token_set_ratio, score_cutoff=90)
        if match: return self.venue_map[match[0]]
        return None

    def match_player_from_query(self, query_player: str) -> Optional[str]:
        query_lower = query_player.lower().strip()
        if not query_lower or not self.all_player_names: return None 
        if query_lower in self.player_map_lower: return self.player_map_lower[query_lower]
        match = process.extractOne(query_lower, self.player_map_lower.keys(), scorer=fuzz.partial_token_set_ratio, score_cutoff=90)
        if match: return self.player_map_lower[match[0]]
        return None

    # -------------------------------------------------------------------------
    #  Public Data Getters
    # -------------------------------------------------------------------------
    def get_hattricks_list(self): return self.hattricks_df.to_dict('records')
    def get_all_venue_names(self): return self.all_venue_names 
    def get_all_player_names(self): return self.all_player_names
    def get_team_map(self): return self.team_map
    def get_team_full_names(self): return self.team_full_names
    def get_all_time_winners_list(self): return self.season_winners 
    
    def get_match_details(self, team_a: str, team_b: str, year: int) -> Optional[Dict[str, Any]]:
        t1 = self.match_team_from_query(team_a); t2 = self.match_team_from_query(team_b)
        if not t1 or not t2: return None
        try: year_int = int(year)
        except: year_int = None
        candidates = self.matches_by_key.get((year_int, frozenset([t1, t2])), [])
        if not candidates: return None
        # Return latest match if multiple
        return candidates[-1].replace({np.nan: None, pd.NaT: None}).to_dict()

    def get_final_match_details(self, year: int) -> Optional[dict]:
        return self.season_finals_data.get(int(year))

    def get_h2h_stats(self, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        t1 = self.match_team_from_query(team_a); t2 = self.match_team_from_query(team_b)
        if not t1 or not t2: return None 
        mask = ((self.df['team1'] == t1) & (self.df['team2'] == t2)) | ((self.df['team1'] == t2) & (self.df['team2'] == t1))
        h2h_df = self.df[mask]
        if h2h_df.empty: return {"team1": t1, "team2": t2, "total_matches": 0, "team1_wins": 0, "team2_wins": 0, "no_result": 0}
        total = len(h2h_df)
        t1_wins = (h2h_df['winner'] == t1).sum()
        t2_wins = (h2h_df['winner'] == t2).sum()
        return {"team1": t1, "team2": t2, "total_matches": total, "team1_wins": int(t1_wins), "team2_wins": int(t2_wins), "no_result": int(total - t1_wins - t2_wins)}

    def get_most_trophies(self) -> Tuple[List[str], int]:
        if self.all_time_trophy_counts.empty: return [], 0
        max_wins = self.all_time_trophy_counts.max()
        return self.all_time_trophy_counts[self.all_time_trophy_counts == max_wins].index.tolist(), int(max_wins)

    def get_most_pom(self, year: Optional[int] = None) -> Tuple[Optional[List[str]], int]:
        df = self.df[self.df['year'] == year] if year else self.df
        if df.empty or 'player_of_match' not in df.columns: return None, 0
        counts = df['player_of_match'].value_counts()
        if counts.empty: return None, 0
        max_awards = counts.max()
        return counts[counts == max_awards].index.tolist(), int(max_awards)

    def get_most_wins_team(self, year: Optional[int] = None) -> Tuple[Optional[List[str]], int]:
        df = self.df[self.df['year'] == year] if year else self.df
        if df.empty: return None, 0
        counts = df['winner'].value_counts()
        if counts.empty: return None, 0
        max_wins = counts.max()
        return counts[counts == max_wins].index.tolist(), int(max_wins)

    def _get_record_match(self, result_type: str, margin_col_sort: str, ascending: bool) -> Optional[Dict[str, Any]]:
        df_f = self.df[self.df['result'] == result_type].dropna(subset=['result_margin'])
        if df_f.empty: return None
        df_f['result_margin'] = pd.to_numeric(df_f['result_margin'], errors='coerce')
        return df_f.sort_values(by=margin_col_sort, ascending=ascending).iloc[0].replace({np.nan: None, pd.NaT: None}).to_dict()

    def get_biggest_win_by_runs(self): return self._get_record_match('Runs', 'result_margin', False)
    def get_biggest_win_by_wickets(self): return self._get_record_match('Wickets', 'result_margin', False)
    def get_narrowest_win_by_runs(self): 
        # For narrowest, we need > 0 and sort ascending
        df_f = self.df[(self.df['result'] == 'Runs') & (self.df['result_margin'].astype(float) > 0)]
        if df_f.empty: return None
        return df_f.sort_values('result_margin', ascending=True).iloc[0].replace({np.nan: None, pd.NaT: None}).to_dict()
    def get_narrowest_win_by_wickets(self):
        df_f = self.df[(self.df['result'] == 'Wickets') & (self.df['result_margin'].astype(float) > 0)]
        if df_f.empty: return None
        return df_f.sort_values('result_margin', ascending=True).iloc[0].replace({np.nan: None, pd.NaT: None}).to_dict()

    def get_most_hosted_stadium(self) -> Tuple[Optional[List[str]], int]:
        counts = self.df['venue'].value_counts()
        if counts.empty: return None, 0
        max_m = counts.max()
        return counts[counts == max_m].index.tolist(), int(max_m)

    def get_dl_method_count(self) -> int:
        return len(self.df[self.df['method'] == 'Dl']) if 'method' in self.df.columns else 0

    def get_super_over_matches(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        if 'super_over' not in self.df.columns: return []
        so_df = self.df[self.df['super_over'] == 'Y']
        if year: so_df = so_df[so_df['year'] == year]
        return so_df.replace({np.nan: None, pd.NaT: None}).to_dict('records')

    def get_toss_winner_win_count(self) -> Tuple[int, int]:
        df = self.df.dropna(subset=['toss_winner', 'winner'])
        df = df[df['winner'] != '']
        matches = len(df)
        wins = len(df[df['toss_winner'] == df['winner']])
        return wins, matches

    def get_team_statistical_summary(self, team_name: str) -> Optional[Dict[str, Any]]:
        full_name = self.match_team_from_query(team_name)
        if not full_name: return None
        matches = self.df[(self.df['team1'] == full_name) | (self.df['team2'] == full_name)]
        total = len(matches)
        if total == 0: return {"team_name": full_name, "total_matches": 0, "total_wins": 0, "win_percentage": 0.0, "trophies_won": 0, "toss_wins": 0}
        wins = (self.df['winner'] == full_name).sum()
        trophies = self.all_time_trophy_counts.get(full_name, 0)
        toss_wins = (self.df['toss_winner'] == full_name).sum()
        return {"team_name": full_name, "total_matches": int(total), "total_wins": int(wins), "win_percentage": (wins/total)*100, "trophies_won": int(trophies), "toss_wins": int(toss_wins)}

    def get_venue_team_record(self, venue_name: str, team_name: str) -> Optional[Dict[str, Any]]:
        v = self.match_venue_from_query(venue_name); t = self.match_team_from_query(team_name)
        if not v or not t: return None
        df = self.df[self.df['venue'] == v]
        df = df[(df['team1'] == t) | (df['team2'] == t)]
        played = len(df)
        wins = (df['winner'] == t).sum()
        return {"venue": v, "team": t, "played": int(played), "wins": int(wins)}

    def get_venue_toss_stats(self, venue_name: str) -> Optional[Dict[str, Any]]:
        v = self.match_venue_from_query(venue_name)
        if not v: return None
        df = self.df[self.df['venue'] == v].dropna(subset=['toss_decision', 'result'])
        total = len(df)
        if total == 0: return {"venue": v, "total": 0, "bat_chosen": 0, "field_chosen": 0, "wins_bat_first": 0, "wins_field_first": 0}
        bat = (df['toss_decision'] == 'Bat').sum()
        field = (df['toss_decision'] == 'Field').sum()
        w_bat = (df['result'] == 'Runs').sum()
        w_field = (df['result'] == 'Wickets').sum()
        return {"venue": v, "total": int(total), "bat_chosen": int(bat), "field_chosen": int(field), "wins_bat_first": int(w_bat), "wins_field_first": int(w_field)}

    def get_player_pom_awards(self, player_name: str) -> Optional[Dict[str, Any]]:
        p = self.match_player_from_query(player_name)
        if not p: return None
        df = self.df[self.df['player_of_match'] == p]
        return {"player": p, "count": int(len(df)), "matches": df.replace({np.nan: None, pd.NaT: None}).to_dict('records')}

    def get_team_season_summary(self, team_name: str, year: int) -> Optional[Dict[str, Any]]:
        t = self.match_team_from_query(team_name)
        if not t: return None
        df = self.df[self.df['year'] == int(year)]
        df = df[(df['team1'] == t) | (df['team2'] == t)]
        return {"team": t, "year": int(year), "matches": df.replace({np.nan: None, pd.NaT: None}).to_dict('records')}

    def get_orange_cap_winner(self, year: int) -> Optional[Dict[str, Any]]:
        if self.df_orange_cap is None: return None
        try: return self.df_orange_cap.loc[int(year)].to_dict()
        except: return None

    def get_purple_cap_winner(self, year: int) -> Optional[Dict[str, Any]]:
        if self.df_purple_cap is None: return None
        try: return self.df_purple_cap.loc[int(year)].to_dict()
        except: return None

    def get_most_orange_caps(self) -> Tuple[List[str], int]:
        if self.most_orange_caps is None: return [], 0
        mx = self.most_orange_caps.max()
        return self.most_orange_caps[self.most_orange_caps == mx].index.tolist(), int(mx)

    def get_most_purple_caps(self) -> Tuple[List[str], int]:
        if self.most_purple_caps is None: return [], 0
        mx = self.most_purple_caps.max()
        return self.most_purple_caps[self.most_purple_caps == mx].index.tolist(), int(mx)

    def get_all_time_top_scorer(self): return self.df_bat_records.iloc[0].to_dict() if self.df_bat_records is not None else None
    def get_all_time_top_wicket_taker(self): return self.df_bowl_records.iloc[0].to_dict() if self.df_bowl_records is not None else None
    def get_top_scorers_list(self, k=5): return self.df_bat_records.head(k).to_dict('records') if self.df_bat_records is not None else []
    def get_top_wicket_takers_list(self, k=5): return self.df_bowl_records.head(k).to_dict('records') if self.df_bowl_records is not None else []