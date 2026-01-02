# backend/chatbot_logic.py
from nlp_utils import extract_entities, extract_intent
from retriever import IPLRetriever
import pandas as pd
import random 
from embeddings import rag_search 
from typing import Dict, Any, Tuple

# --- HELP MESSAGE ---
HELP_MESSAGE = "I'm sorry, I didn't understand that. I can answer questions about:\n\n" \
             "• **Team Squads:** 'squad of RCB in 2016', 'CSK player list 2024'\n" \
             "• **Team Stats:** 'details about MI', 'stats for CSK', 'RCB matches in 2016'\n" \
             "• **Player Stats:** 'stats of Virat Kohli', 'batting profile of Dhoni'\n" \
             "• **Seasonal Awards:** 'orange cap in 2020', 'purple cap winner 2023'\n" \
             "• **All-Time Records:** 'most trophies', 'highest run scorer', 'most wickets'\n" \
             "• **Match Details:** 'CSK vs MI 2020', '2019 final details'\n" \
             "• **Head-to-Head:** 'CSK vs MI H2H'\n" \
             "• **Venues:** 'list all venues', 'record at Wankhede'"

# --- Define initial context ---
INITIAL_CONTEXT = {"year": None, "teams": [], "players": [], "venues": [], "intent": None}

# ---- Formatters (Helpers) ----
def format_match_details(details: dict) -> str:
    if not details: return "I found the match, but I'm missing the specific details."
    try:
        t1 = details.get("team1") or "Team 1"; t2 = details.get("team2") or "Team 2"
        year = details.get("year") or ""; venue = details.get("venue") or "N/A"
        toss_winner = details.get("toss_winner") or "N/A"; toss_decision = details.get("toss_decision") or ""
        winner = details.get("winner") or "N/A"; result = details.get("result") or ""
        margin = details.get("result_margin"); pom = details.get("player_of_match") or "N/A"
        margin_str = ""
        if pd.notna(margin) and margin:
            try: margin_str = f"by {int(margin)}"
            except: margin_str = ""
        answer = f"Here are the details for the {t1} vs {t2} match in {year}:\n\n"
        answer += f"• **Venue:** {venue}\n"
        answer += f"• **Toss:** {toss_winner} won the toss and chose to {toss_decision}.\n"
        answer += f"• **Winner:** {winner} won {margin_str} {result}.\n"
        answer += f"• **Player of the Match:** {pom}."
        return answer
    except Exception as e:
        print(f"Error formatting details: {e}"); return "I found the match, but had trouble formatting."

def format_record_match(details: dict, intro: str) -> str:
    if not details: return f"Sorry, I couldn't find the record for {intro}."
    try:
        t1 = details.get("team1"); t2 = details.get("team2"); year = details.get("year")
        venue = details.get("venue"); winner = details.get("winner"); result = details.get("result")
        margin = int(details.get("result_margin", 0))
        answer = f"The **{intro}** was in **{year}** at {venue}:\n\n"
        answer += f"• **Match:** {t1} vs {t2}\n"; answer += f"• **Winner:** {winner}\n"
        answer += f"• **Margin:** {margin} {result}"
        return answer
    except Exception as e:
        print(f"Error formatting record: {e}"); return "I found the record, but had trouble formatting."

def format_team_season_summary(summary: dict) -> str:
    if not summary or not summary.get("matches"):
        return f"Sorry, I couldn't find any matches for **{summary.get('team')}** in **{summary.get('year')}**."
    team = summary['team']; year = summary['year']; matches = summary['matches']
    wins = 0; losses = 0; no_result = 0
    answer = f"Here's a summary for **{team}** in **{year}**:\n\n"
    for match in matches:
        opponent = match['team2'] if match['team1'] == team else match['team1']
        winner = match.get('winner')
        if winner == team: wins += 1; result_str = "**Won**"
        elif not winner or winner == 'N/A' or winner == '': no_result += 1; result_str = "No Result"
        else: losses += 1; result_str = "Lost"
        answer += f"• vs **{opponent}**: {result_str}\n"
    answer += f"\n**Total Record:** {wins} Wins, {losses} Losses, {no_result} No Results."
    return answer

# --- RAG Answer Function ---
def rag_answer(query: str) -> str | None:
    try:
        results = rag_search(query, k=3)
        if not results: return None
        
        valid_results = []
        for r in results:
            if r and isinstance(r, str) and pd.notna(r) and r.strip():
                valid_results.append(r)
            elif r and pd.notna(r) and not isinstance(r, str):
                valid_results.append(str(r))
        
        if not valid_results: return None
            
        answer = "I'm not sure about that, but here's some related information I found:\n\n"
        for r in valid_results:
            answer += f"• {r}\n"
        return answer.strip()
    except Exception as e:
        print(f"Error during RAG search: {e}"); return None


# ---- MAIN LOGIC ----
def process_query_v2(query: str, retriever: IPLRetriever, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # --- 1. Handle Greetings ---
    q_lower = query.lower().strip()
    greetings = ["hello", "hi", "hey", "hola", "yo"]
    if q_lower in greetings:
        return random.choice(["Hi there!", "Hello!", "Hey! Ask me about IPL stats."]), context 
    
    chit_chat_responses = {
        "how are you": "I'm just a bot, but I'm ready to help with your IPL questions!",
        "thanks": "You're welcome!", "thank you": "You're welcome!"
    }
    if q_lower in chit_chat_responses: return chit_chat_responses[q_lower], context

    if q_lower in ["help", "what can you do", "help me"]: return HELP_MESSAGE, context
    if q_lower in ["no", "nope", "stop", "cancel"]: return "Context cleared.", INITIAL_CONTEXT.copy()
    if q_lower in ["yes", "yeah", "ok"]: return "Great! How can I help?", context

    # --- 2. Extract Entities & Intent ---
    entities = extract_entities(query, retriever)
    intent = extract_intent(q_lower, entities)

    # --- 3. Context Merge ---
    year = entities['year'] if entities['year'] is not None else context.get('year')
    teams = entities['teams'] if entities['teams'] else context.get('teams', [])
    players = entities['players'] if entities['players'] else context.get('players', [])
    venues = entities['venues'] if entities['venues'] else context.get('venues', [])
    
    merged_intent = intent if intent != "unknown" else context.get('intent', 'unknown')
    
    updated_context = {'year': year, 'teams': teams, 'players': players, 'venues': venues, "intent": merged_intent}
    
    # --- 4. INTENT HANDLERS (ALL FUNCTIONALITY) ---

    # --- [NEW] 1. Get Team Squad ---
    if merged_intent == "get_team_squad":
        if teams:
            team_name = teams[0]
            # Fetch squad (year defaults to latest if None)
            result = retriever.get_team_squad(team_name, year)
            
            if result and result['players']:
                found_year = result['year']
                found_team = result['team']
                
                answer = f"Here is the squad for **{found_team}** in **{found_year}**:\n\n"
                # Limit to first 15 if list is huge, or show all
                for p in result['players']:
                    answer += f"• {p}\n"
                answer += f"\n**Total Players:** {len(result['players'])}"
                
                updated_context['year'] = found_year
                return answer, updated_context
            else:
                if year:
                    return f"I found **{team_name}**, but I don't have squad data for **{year}**.", updated_context
                return f"I recognized **{team_name}**, but I couldn't find any squad data.", updated_context
        else:
            return "Which team's squad would you like to see? (e.g., 'squad of RCB 2016')", updated_context

    # --- [NEW] 2. Get Player Career Stats ---
    elif merged_intent == "get_player_career_stats":
        target_players = players if players else context.get('players', [])
        if target_players:
            player_name = target_players[0]
            stats = retriever.get_player_career_stats(player_name)
            
            if stats:
                answer = f"Here are the career stats for **{stats['name']}**:\n\n"
                answer += f"**Matches:** {stats['matches']}\n"
                
                if stats['bat_innings'] > 0:
                    answer += "\n🏏 **Batting:**\n"
                    answer += f"• **Runs:** {stats['runs']}\n"
                    answer += f"• **Average:** {stats['avg']}\n"
                    answer += f"• **Strike Rate:** {stats['sr']}\n"
                    answer += f"• **100s/50s:** {stats['100s']} / {stats['50s']}\n"
                    answer += f"• **Boundaries:** {stats['4s']} (4s), {stats['6s']} (6s)\n"
                
                if stats['bowl_innings'] > 0:
                    answer += "\n🥎 **Bowling:**\n"
                    answer += f"• **Wickets:** {stats['wickets']}\n"
                    answer += f"• **Economy:** {stats['economy']}\n"
                    answer += f"• **Average:** {stats['bowl_avg']}\n"
                    answer += f"• **Strike Rate:** {stats['bowl_sr']}"
                    
                return answer, INITIAL_CONTEXT.copy()
            else:
                return f"Sorry, I couldn't find detailed stats for **{player_name}**.", updated_context
        else:
            return "Whose stats do you want to see? (e.g. 'stats of Virat Kohli')", updated_context

    # --- 3. Existing Stats Handlers ---
    elif merged_intent == "list_hattricks":
        hattricks = retriever.get_hattricks_list()
        if not hattricks: return "Sorry, I couldn't find the hat-trick records.", INITIAL_CONTEXT.copy()
        answer = "Here are the IPL Hat-tricks:\n\n"
        for h in hattricks:
            answer += f"• **{h.get('bowler')}** ({h.get('year')}) vs {h.get('match', '').split(' v ')[1] if ' v ' in h.get('match', '') else 'Opponent'}\n"
        return answer, INITIAL_CONTEXT.copy()

    elif merged_intent == "get_team_details":
        if teams:
            stats = retriever.get_team_statistical_summary(teams[0])
            if stats and stats["total_matches"] > 0:
                answer = f"**{stats['team_name']}** Summary:\n"
                answer += f"• Trophies: {stats['trophies_won']}\n"
                answer += f"• Matches: {stats['total_matches']} (Won: {stats['total_wins']})\n"
                answer += f"• Win %: {stats['win_percentage']:.2f}%"
                return answer, INITIAL_CONTEXT.copy()
            else: return f"No data found for {teams[0]}.", updated_context
        return "Which team?", updated_context

    elif merged_intent == "get_team_season_summary":
        if teams and year:
            summary = retriever.get_team_season_summary(teams[0], year)
            return format_team_season_summary(summary), INITIAL_CONTEXT.copy()
        return "Please provide a team and year.", updated_context

    elif merged_intent == "get_venue_team_record":
        if venues and teams:
            stats = retriever.get_venue_team_record(venues[0], teams[0])
            if stats:
                return f"**{stats['team']}** at **{stats['venue']}**:\nPlayed: {stats['played']}, Won: {stats['wins']}", INITIAL_CONTEXT.copy()
            return "Record not found.", updated_context
        return "I need a team and a venue.", updated_context

    elif merged_intent == "get_venue_toss_stats":
        if venues:
            stats = retriever.get_venue_toss_stats(venues[0])
            if stats:
                return f"**{stats['venue']}** Stats:\nBat 1st Wins: {stats['wins_bat_first']}\nField 1st Wins: {stats['wins_field_first']}", INITIAL_CONTEXT.copy()
            return "Venue stats not found.", updated_context
        return "Which venue?", updated_context

    elif merged_intent == "get_player_pom_awards":
        if players:
            stats = retriever.get_player_pom_awards(players[0])
            if stats and stats['count'] > 0:
                return f"**{stats['player']}** has won **{stats['count']}** Player of the Match awards.", INITIAL_CONTEXT.copy()
            return f"{players[0]} hasn't won any POM awards.", updated_context
        return "Which player?", updated_context

    elif merged_intent == "get_match_details":
        if len(teams) >= 2 and year:
            details = retriever.get_match_details(teams[0], teams[1], year)
            return format_match_details(details), INITIAL_CONTEXT.copy()
        elif year and not teams:
            details = retriever.get_final_match_details(year)
            return format_match_details(details), INITIAL_CONTEXT.copy()
        return "Please specify two teams and a year.", updated_context

    # --- 4. Records & Lists ---
    elif merged_intent == "get_most_pom":
        p, c = retriever.get_most_pom(year)
        t_str = f"in {year}" if year else "all-time"
        return f"Most POM awards {t_str}: {', '.join(p)} ({c})", INITIAL_CONTEXT.copy()

    elif merged_intent == "get_most_wins_team":
        t, c = retriever.get_most_wins_team(year)
        t_str = f"in {year}" if year else "all-time"
        return f"Most wins {t_str}: {', '.join(t)} ({c})", INITIAL_CONTEXT.copy()

    elif merged_intent == "get_most_trophies":
        t, c = retriever.get_most_trophies()
        return f"Most Trophies: {', '.join(t)} ({c})", INITIAL_CONTEXT.copy()
        
    elif merged_intent == "get_all_time_top_scorer":
        d = retriever.get_all_time_top_scorer()
        return f"All-time Top Scorer: **{d['player']}** ({d['runs']} runs)", INITIAL_CONTEXT.copy()
        
    elif merged_intent == "get_all_time_top_wicket_taker":
        d = retriever.get_all_time_top_wicket_taker()
        return f"All-time Top Wicket Taker: **{d['player']}** ({d['wickets']} wickets)", INITIAL_CONTEXT.copy()

    elif merged_intent == "get_orange_cap_winner":
        if year:
            d = retriever.get_orange_cap_winner(year)
            return (f"Orange Cap {year}: **{d['player']}** ({d['runs']} runs)", INITIAL_CONTEXT.copy()) if d else ("Data not found.", updated_context)
        return "Which year?", updated_context

    elif merged_intent == "get_purple_cap_winner":
        if year:
            d = retriever.get_purple_cap_winner(year)
            return (f"Purple Cap {year}: **{d['player']}** ({d['wickets']} wickets)", INITIAL_CONTEXT.copy()) if d else ("Data not found.", updated_context)
        return "Which year?", updated_context

    elif merged_intent == "get_most_orange_caps":
        p, c = retriever.get_most_orange_caps()
        return f"Most Orange Caps: {', '.join(p)} ({c})", INITIAL_CONTEXT.copy()

    elif merged_intent == "get_most_purple_caps":
        p, c = retriever.get_most_purple_caps()
        return f"Most Purple Caps: {', '.join(p)} ({c})", INITIAL_CONTEXT.copy()

    elif merged_intent == "get_winner":
        if year:
            w = retriever.season_winners.get(year)
            return (f"Winner of {year}: {w}", INITIAL_CONTEXT.copy()) if w else ("Unknown.", updated_context)
        return "Which year?", updated_context

    elif merged_intent == "get_h2h_stats":
        if len(teams) >= 2:
            s = retriever.get_h2h_stats(teams[0], teams[1])
            if s:
                return f"**{s['team1']}** vs **{s['team2']}**:\nMatches: {s['total_matches']}\n{s['team1']} Wins: {s['team1_wins']}\n{s['team2']} Wins: {s['team2_wins']}", INITIAL_CONTEXT.copy()
        return "Need two teams.", updated_context

    elif merged_intent == "get_umpires":
        if len(teams) >= 2 and year:
            d = retriever.get_match_details(teams[0], teams[1], year)
            if d: return f"Umpires: {d.get('umpire1')}, {d.get('umpire2')}", INITIAL_CONTEXT.copy()
        return "Match not found.", updated_context

    elif merged_intent == "get_top_scorers_list":
        data = retriever.get_top_scorers_list(5)
        ans = "Top 5 Run Scorers:\n"
        for i, d in enumerate(data): ans += f"{i+1}. {d['player']} ({d['runs']})\n"
        return ans, INITIAL_CONTEXT.copy()

    elif merged_intent == "get_top_wicket_takers_list":
        data = retriever.get_top_wicket_takers_list(5)
        ans = "Top 5 Wicket Takers:\n"
        for i, d in enumerate(data): ans += f"{i+1}. {d['player']} ({d['wickets']})\n"
        return ans, INITIAL_CONTEXT.copy()
        
    elif merged_intent == "list_teams":
        return "Teams: " + ", ".join(retriever.get_team_full_names()), INITIAL_CONTEXT.copy()
        
    elif merged_intent == "list_venues":
        return "Venues: " + ", ".join(retriever.get_all_venue_names()[:10]) + "...", INITIAL_CONTEXT.copy()

    # --- 5. Specific Match Records ---
    elif merged_intent == "get_biggest_win_by_runs":
        return format_record_match(retriever.get_biggest_win_by_runs(), "biggest win by runs"), INITIAL_CONTEXT.copy()
    elif merged_intent == "get_biggest_win_wickets":
        return format_record_match(retriever.get_biggest_win_by_wickets(), "biggest win by wickets"), INITIAL_CONTEXT.copy()
    elif merged_intent == "get_narrowest_win_runs":
        return format_record_match(retriever.get_narrowest_win_by_runs(), "narrowest win by runs"), INITIAL_CONTEXT.copy()
    elif merged_intent == "get_narrowest_win_wickets":
        return format_record_match(retriever.get_narrowest_win_by_wickets(), "narrowest win by wickets"), INITIAL_CONTEXT.copy()
    elif merged_intent == "get_most_hosted_stadium":
        s, c = retriever.get_most_hosted_stadium()
        return f"Most hosted: {s[0]} ({c} matches)", INITIAL_CONTEXT.copy()
    elif merged_intent == "get_dl_method_count":
        return f"D/L matches: {retriever.get_dl_method_count()}", INITIAL_CONTEXT.copy()
    elif merged_intent == "get_super_over_matches":
        m = retriever.get_super_over_matches(year)
        return f"Super Overs: {len(m)} found.", INITIAL_CONTEXT.copy()
    elif merged_intent == "get_toss_winner_win_count":
        w, t = retriever.get_toss_winner_win_count()
        return f"Toss winners won match: {w}/{t}", INITIAL_CONTEXT.copy()

    # --- 6. Fallback (RAG) ---
    if merged_intent == "unknown":
        rag_result = rag_answer(query)
        if rag_result: return rag_result, INITIAL_CONTEXT.copy()
        # Intelligent fallback for partial context
        if year or teams or venues or players:
            return f"I see you mentioned {year or ''} {teams or ''} {players or ''}. What specific stat do you want?", updated_context

    return HELP_MESSAGE, INITIAL_CONTEXT.copy()