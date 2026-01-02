# backend/nlp_utils.py
import re
import spacy
from rapidfuzz import process, fuzz
from typing import Tuple, List, Dict, TYPE_CHECKING

# Import retriever only for type checking to avoid circular import
if TYPE_CHECKING:
    from retriever import IPLRetriever

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except (IOError, OSError):
    print("spaCy model 'en_core_web_sm' not found. Downloading...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}. Falling back to basic sentencizer.")
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe("sentencizer")

def extract_entities(question: str, retriever: 'IPLRetriever') -> Dict:
    """
    Extracts all known entities from a question using robust fuzzy matching.
    Returns: {'year': int, 'teams': list, 'players': list, 'venues': list}
    """
    q_lower = question.lower()
    entities = {'year': None, 'teams': [], 'players': [], 'venues': []}

    # 1. Extract Year (Regex is best)
    year_match = re.search(r"\b(20(0[8-9]|1[0-9]|2[0-9]))\b", q_lower)
    if year_match:
        entities['year'] = int(year_match.group(1))

    # 2. Extract Teams
    team_choices = list(retriever.get_team_map().keys()) + retriever.get_team_full_names()
    found_teams = process.extract(q_lower, list(set(team_choices)), scorer=fuzz.token_set_ratio, limit=2, score_cutoff=80)
    
    for (team_name_or_abbr, score, index) in found_teams:
        full_name = retriever.match_team_from_query(team_name_or_abbr)
        if full_name and full_name not in entities['teams']:
            entities['teams'].append(full_name)

    # 3. Extract Players
    player_list = retriever.get_all_player_names()
    if player_list: 
        # Increase limit to catch full names better
        found_players = process.extract(q_lower, player_list, scorer=fuzz.partial_token_set_ratio, limit=2, score_cutoff=85)
        
        for (player_name, score, index) in found_players:
            if player_name not in entities['players']:
                entities['players'].append(player_name)
            
    # 4. Extract Venues
    venue_list = retriever.get_all_venue_names()
    if venue_list: 
        found_venues = process.extract(q_lower, venue_list, scorer=fuzz.partial_token_set_ratio, limit=1, score_cutoff=90)
        for (venue_name, score, index) in found_venues:
            verified_venue = retriever.match_venue_from_query(venue_name)
            if verified_venue and verified_venue not in entities['venues']:
                 entities['venues'].append(verified_venue)

    return entities


def extract_intent(q_lower: str, entities: Dict) -> str:
    """
    Determines intent based on keywords and fuzzy matching.
    """
    year, teams, players, venues = entities['year'], entities['teams'], entities['players'], entities['venues']

    if "hattrick" in q_lower or "hat-trick" in q_lower or "hat trick" in q_lower:
        return "list_hattricks"

    # --- NEW: Team Squad Intent (High Priority) ---
    # Catches: "player list of RCB", "CSK squad", "who plays for MI", "list team members"
    if len(teams) > 0 and ("squad" in q_lower or "player list" in q_lower or "team list" in q_lower or "players of" in q_lower or "members" in q_lower or "who plays for" in q_lower):
        return "get_team_squad"

    # --- NEW: General Player Stats Intent (High Priority) ---
    # Catches: "stats of virat", "profile of dhoni", "batting stats", "bowling figures"
    if len(players) > 0 and ("stat" in q_lower or "profile" in q_lower or "career" in q_lower or "batting" in q_lower or "bowling" in q_lower or "record" in q_lower):
        return "get_player_career_stats"

    # --- High Priority Seasonal Award Intents ---
    if "orange cap" in q_lower and year:
        return "get_orange_cap_winner"
    if "purple cap" in q_lower and year:
        return "get_purple_cap_winner"

    # --- High Priority Intents (Order Matters!) ---
    if ("record at" in q_lower or "stats at" in q_lower or (len(teams) >= 1 and len(venues) >= 1 and " at " in q_lower)):
        return "get_venue_team_record"
    if ("bat or field" in q_lower or "preference" in q_lower or "toss decision" in q_lower or "choose at" in q_lower) and len(venues) >= 1:
        return "get_venue_toss_stats"
    if (("matches" in q_lower or "results" in q_lower or "summary" in q_lower) and len(teams) == 1 and year):
        return "get_team_season_summary"
    if ("how many pom" in q_lower or ("pom" in q_lower and len(players) >= 1)):
        return "get_player_pom_awards"
        
    if "biggest win" in q_lower and "wickets" in q_lower:
        return "get_biggest_win_wickets"
    if "narrowest" in q_lower and "runs" in q_lower:
        return "get_narrowest_win_runs"
    if "narrowest" in q_lower and "wickets" in q_lower:
        return "get_narrowest_win_wickets"
        
    if ("detail" in q_lower or "about" in q_lower or "stats for" in q_lower or "know about" in q_lower or "state of" in q_lower) and len(teams) == 1 and not year:
         return "get_team_details"

    # --- "Mosts" & Records (All-Time) ---
    if "most" in q_lower and "orange cap" in q_lower:
        return "get_most_orange_caps"
    if "most" in q_lower and "purple cap" in q_lower:
        return "get_most_purple_caps"
    if "most" in q_lower and ("troph" in q_lower or "title" in q_lower):
        return "get_most_trophies"
    if "most player of the match" in q_lower or "most pom" in q_lower:
        return "get_most_pom"
    if "most" in q_lower and ("win" in q_lower or "victor" in q_lower):
        return "get_most_wins_team"
    if "biggest win" in q_lower and "runs" in q_lower:
        return "get_biggest_win_by_runs" 
    if ("stadium" in q_lower or "venue" in q_lower) and "most" in q_lower:
        return "get_most_hosted_stadium"
    
    # --- All-Time Record Intents ---
    if "highest run" in q_lower or "all time top scorer" in q_lower:
        return "get_all_time_top_scorer"
    if "highest wicket" in q_lower or "all time top wicket" in q_lower:
        return "get_all_time_top_wicket_taker"
        
    # --- Toss & Special Matches ---
    if "d/l method" in q_lower or "duckworth-lewis" in q_lower:
        return "get_dl_method_count"
    if "super over" in q_lower:
        return "get_super_over_matches"
    if "toss winner" in q_lower and "win" in q_lower:
        return "get_toss_winner_win_count"

    # --- Existing Intents (Lower Priority) ---
    if ("list" in q_lower or "all" in q_lower) and "winner" in q_lower:
        return "get_all_winners"
    
    if "winner" in q_lower or "champion" in q_lower or fuzz.token_set_ratio(q_lower, "who won") > 90:
        return "get_winner"
    
    if "head to head" in q_lower or "h2h" in q_lower:
        return "get_h2h_stats"
    
    if (fuzz.token_set_ratio(q_lower, "player of match") > 90 or "pom" in q_lower) and not players:
        if year or len(teams) >= 2:
            return "get_match_details"
            
    if "umpire" in q_lower:
        return "get_umpires"
    if "final" in q_lower and ("details" in q_lower or "umpire" in q_lower or "pom" in q_lower):
        return "get_match_details"

    # --- List Intents ---
    if ("list" in q_lower or "top" in q_lower) and ("batsm" in q_lower or "scorer" in q_lower):
        return "get_top_scorers_list"
    if ("list" in q_lower or "top" in q_lower) and ("bowl" in q_lower or "wicket" in q_lower):
        return "get_top_wicket_takers_list"
        
    if "list teams" in q_lower:
        return "list_teams"
        
    if ("list" in q_lower or "all" in q_lower) and "venu" in q_lower:
        return "list_venues"
        
    if len(teams) >= 2 and year: # Standard match query
         return "get_match_details"
    
    if "orange cap" in q_lower:
        return "get_orange_cap_winner" 
    if "purple cap" in q_lower:
        return "get_purple_cap_winner" 

    # --- Ambiguous Cases (Default Intents) ---
    if len(teams) == 1 and not year and not players and not venues:
        return "get_team_details"
    
    # NEW: Default to career stats if player is mentioned but intent is unclear
    # e.g. just "Virat Kohli"
    if len(players) == 1 and not teams and not year and not venues and not q_lower.startswith("who"): 
        if "pom" in q_lower:
            return "get_player_pom_awards"
        return "get_player_career_stats"

    return "unknown"