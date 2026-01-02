# backend/gemini_utils.py
import os
import httpx
from typing import Optional

async def get_gemini_analysis(prompt: str) -> Optional[str]:
    """
    Calls the Gemini API to get a natural language analysis of the match.
    """
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return "AI analysis could not be loaded. The server is missing the GEMINI_API_KEY."

    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

    system_prompt = (
        "You are 'WinBot', an expert cricket analyst. "
        "Your role is to provide a single, concise sentence explaining *why* the winning "
        "team is in a strong position, based on the data. "
        "Focus on the most important factor (e.g., low required run rate, high wickets in hand, etc.). "
        "Do not greet the user. Be direct and insightful."
    )

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 300,
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15.0
            )
            
            response.raise_for_status() # Raise an error for bad HTTP responses
            
            result = response.json()
            
            # --- ROBUST PARSING FIX ---
            # This safely checks for the text, even if the API
            # returns an empty or blocked response.
            try:
                # 1. Safely get the first candidate, or an empty dict
                candidate = result.get("candidates", [{}])[0]
                
                # 2. Check if the API blocked the prompt
                if candidate.get("finishReason") == "SAFETY":
                    print("--- GEMINI ANALYSIS BLOCKED FOR SAFETY ---")
                    return "AI analysis was blocked by safety filters."

                # 3. Safely get the 'content' dictionary
                content = candidate.get("content", {})
                if not content:
                    print(f"--- GEMINI UNEXPECTED RESPONSE (No Content) ---: {result}")
                    return "AI analysis returned no content."

                # 4. Safely get the 'parts' list
                parts_list = content.get("parts", [{}])
                if not parts_list:
                    print(f"--- GEMINI UNEXPECTED RESPONSE (No Parts) ---: {result}")
                    return "AI analysis returned no parts."

                # 5. Safely get the 'text'
                text = parts_list[0].get("text")
                
                if text:
                    return text.strip()
                else:
                    # If 'text' is None or empty, the response was malformed
                    print(f"--- GEMINI UNEXPECTED RESPONSE (No Text) ---: {result}")
                    return "AI analysis returned an empty or unexpected response."

            except Exception as parse_error:
                # This will catch any other parsing error
                print(f"--- GEMINI RESPONSE PARSE ERROR ---: {parse_error}")
                print(f"--- FULL API RESPONSE ---: {result}")
                return f"AI analysis failed while parsing. Error: {parse_error}"
            # --- END OF FIX ---

    except httpx.HTTPStatusError as e:
        # This catches 4xx/5xx errors (like 404, 401, 500)
        print(f"Error calling Gemini API: {e}")
        print(f"API Response Text: {e.response.text}")
        return f"AI analysis failed. HTTP Error: {e.response.text}"
    except Exception as e:
        # This catches other errors like timeouts or connection issues
        print(f"Error calling Gemini API: {e}")
        return f"AI analysis failed. An unexpected error occurred: {e}"
