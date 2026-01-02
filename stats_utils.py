# backend/stats_utils.py
import pandas as pd
import os
from rapidfuzz import process, fuzz

ACTIVE_TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

# Mapping for IPL.csv (Old data compatibility)
TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Titans',
    'Pune Warriors': 'Rising Pune Supergiant',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
}

class StatsEngine:
    def __init__(self, ipl_path="data/IPL.csv", player_stats_path="data/player_stats.csv", team_data_path="data/team_data.csv"):
        self.ipl_path = ipl_path
        self.player_stats_path = player_stats_path
        self.team_data_path = team_data_path
        self.df = None
        self.player_stats = None
        self.matches_df = None
        self.team_summary_df = None
        self._load_data()

    def _load_data(self):
        # 1. Load Match Data (For Charts/Trends)
        if os.path.exists(self.ipl_path):
            try:
                self.df = pd.read_csv(self.ipl_path, low_memory=False)
                cols_map = {'innings': 'inning', 'runs_total': 'total_runs', 
                            'player_out': 'player_dismissed', 'match_won_by': 'winner'}
                self.df.rename(columns=cols_map, inplace=True)
                
                self.df['batting_team'] = self.df['batting_team'].replace(TEAM_MAPPING)
                self.df['bowling_team'] = self.df['bowling_team'].replace(TEAM_MAPPING)
                if 'winner' in self.df.columns:
                    self.df['winner'] = self.df['winner'].replace(TEAM_MAPPING)

                if 'date' in self.df.columns:
                    self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
                    self.df['year'] = self.df['date'].dt.year
                else:
                    self.df['year'] = 2024 

                self.matches_df = self.df.drop_duplicates(subset=['match_id']).copy()
                self.matches_df.sort_values('date', inplace=True)
            except Exception as e:
                print(f"Error loading IPL.csv: {e}")

        # 2. Load Player Data
        if os.path.exists(self.player_stats_path):
            try:
                self.player_stats = pd.read_csv(self.player_stats_path)
                self.player_stats['Player'] = self.player_stats['Player'].astype(str)
            except Exception as e:
                print(f"Error loading player_stats: {e}")

        # 3. Load Team Summary Data (The NEW CSV)
        if os.path.exists(self.team_data_path):
            try:
                self.team_summary_df = pd.read_csv(self.team_data_path)
                # Map weird column names to standard ones based on your file structure
                self.team_summary_df.rename(columns={
                    'ds-text-tight-s': 'Team',
                    'ds-min-w-max (2)': 'Matches',
                    'ds-min-w-max (3)': 'Won',
                    'ds-min-w-max (4)': 'Lost',
                    'ds-min-w-max (5)': 'Tied',
                    'ds-min-w-max (8)': 'NR'
                }, inplace=True)
            except Exception as e:
                print(f"Error loading team_data.csv: {e}")

    def get_team_stats(self, team_name):
        # 1. Identify Team
        match = process.extractOne(team_name, ACTIVE_TEAMS, score_cutoff=60)
        full_name = match[0] if match else team_name
        
        # 2. Get Summary from NEW CSV (Priority)
        summary_stats = {"played": 0, "wins": 0, "losses": 0}
        
        if self.team_summary_df is not None:
            # Fuzzy match against the CSV team names
            csv_teams = self.team_summary_df['Team'].tolist()
            csv_match = process.extractOne(full_name, csv_teams, score_cutoff=70)
            
            if csv_match:
                row = self.team_summary_df[self.team_summary_df['Team'] == csv_match[0]].iloc[0]
                summary_stats = {
                    "played": int(row.get('Matches', 0)),
                    "wins": int(row.get('Won', 0)),
                    "losses": int(row.get('Lost', 0))
                }
            else:
                # Fallback to calculating from IPL.csv if team not found in summary
                if self.matches_df is not None:
                    team_matches = self.matches_df[
                        (self.matches_df['batting_team'] == full_name) | 
                        (self.matches_df['bowling_team'] == full_name)
                    ]
                    summary_stats = {
                        "played": len(team_matches),
                        "wins": len(team_matches[team_matches['winner'] == full_name]),
                        "losses": len(team_matches) - len(team_matches[team_matches['winner'] == full_name])
                    }
        
        # 3. Get Charts/Trends from IPL.csv (Matches Data)
        # Note: We still need IPL.csv for the year-wise breakdown as summary csv lacks years
        season_wins = {}
        toss_decisions = {}
        
        if self.matches_df is not None:
            team_matches = self.matches_df[
                (self.matches_df['batting_team'] == full_name) | 
                (self.matches_df['bowling_team'] == full_name)
            ]
            
            if not team_matches.empty:
                # Calculate Season Wins
                sw = team_matches[team_matches['winner'] == full_name].groupby('year').size().to_dict()
                # Force 2008-2025 Timeline
                for y in range(2008, 2026):
                    season_wins[y] = sw.get(y, 0)
                
                # Toss Decisions
                if 'toss_winner' in team_matches.columns:
                    toss_decisions = team_matches[team_matches['toss_winner'] == full_name]['toss_decision'].value_counts().to_dict()

        return {
            "summary": summary_stats,
            "season_wins": season_wins,
            "toss_decisions": toss_decisions
        }

    def get_player_stats(self, player_name):
        stats = {"found": False, "batting": {}, "bowling": {}, "recent_form": []}

        if self.player_stats is not None:
            choices = self.player_stats['Player'].tolist()
            match = process.extractOne(player_name, choices, scorer=fuzz.token_sort_ratio, score_cutoff=70)
            if match:
                row = self.player_stats.iloc[match[2]]
                stats['found'] = True
                stats['name'] = match[0]
                stats['batting'] = {
                    "matches": int(row.get('Matches', 0)),
                    "runs": int(row.get('Runs', 0)),
                    "avg": float(row.get('Avg', 0.0)),
                    "sr": float(row.get('SR', 0.0)),
                    "100s": int(row.get('Centuries', 0)),
                    "50s": int(row.get('Fifties', 0)),
                    "4s": int(row.get('Fours', 0)),
                    "6s": int(row.get('Sixes', 0))
                }
                stats['bowling'] = {
                    "wickets": int(row.get('Wickets_taken', 0)),
                    "economy": float(row.get('Economy', 0.0))
                }
        
        if self.df is not None:
            bat_col = 'batter' if 'batter' in self.df.columns else 'batsman'
            search_name = stats.get('name', player_name)
            
            player_df = self.df[self.df[bat_col] == search_name]
            if not player_df.empty:
                stats['found'] = True
                if 'name' not in stats: stats['name'] = search_name
                
                recent_scores = (
                    player_df.groupby(['match_id', 'date'])['runs_batter']
                    .sum()
                    .sort_index(level=1)
                    .tail(10)
                    .tolist()
                )
                stats['recent_form'] = recent_scores

        return stats if stats['found'] else None