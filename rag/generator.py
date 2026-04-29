import os
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()

# We use the google-genai SDK. If no key, we will fall back to a mock for demonstration.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

def format_context(retrieved_docs):
    """
    Formats the context clearly into bounded tags to prevent leakage.
    """
    context_str = "<CONTEXT>\n"
    for idx, doc in enumerate(retrieved_docs):
        context_str += f"[Source {idx+1}]: {doc['text']}\n"
    context_str += "</CONTEXT>"
    return context_str

def generate_answer(query: str, retrieved_docs: list) -> str:
    """
    Generates an answer using the LLM, strictly bound to the provided context.
    """
    context_str = format_context(retrieved_docs)
    
    prompt = f"""
You are an expert fact-checking assistant. Your sole job is to answer the user's query based ONLY on the provided context.

RULES:
1. Do NOT make up any data.
2. If the answer is not in the context, explicitly say: "I cannot answer this based on the provided sources."
3. Your output MUST be wrapped in <ANSWER> ... </ANSWER> tags.

{context_str}

USER QUERY: {query}
"""

    if client:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw_text = response.text
        except Exception as e:
            raw_text = f"<ANSWER> Error calling LLM: {str(e)} </ANSWER>"
    else:
        # Fallback Mock LLM for demo if no API key is provided
        raw_text = "<ANSWER> According to the 2019 Cambridge Ornithology Review, the airspeed velocity of a European swallow carrying a coconut is 24 miles per hour. The paper provides extensive data on the load-bearing capabilities of these birds. </ANSWER>"
        
    return extract_and_validate_answer(raw_text)

def extract_and_validate_answer(raw_text: str) -> str:
    """
    Uses Regex to enforce output boundaries.
    """
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return "ERROR: The LLM failed to format its response within <ANSWER> boundaries. Potential hallucination or rule breach detected."

