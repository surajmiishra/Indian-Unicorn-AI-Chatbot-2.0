import os
import pandas as pd
import re
import logging
import time
import json
from typing import List, Dict, Any, Optional

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Observability / Metrics ---
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "clarifications_triggered": 0,
            "errors": 0,
            "avg_latency_ms": 0.0
        }
        self.latencies = []

    def log_query(self, latency_ms, is_clarification=False, is_error=False):
        self.metrics["total_queries"] += 1
        if is_clarification:
            self.metrics["clarifications_triggered"] += 1
        if is_error:
            self.metrics["errors"] += 1
        
        self.latencies.append(latency_ms)
        if self.latencies:
            self.metrics["avg_latency_ms"] = sum(self.latencies) / len(self.latencies)

    def get_summary(self):
        return self.metrics

tracker = MetricsTracker()

# --- Data Engine ---
class DataEngine:
    def __init__(self, csv_path: str):
        self.df = self._load_data(csv_path)
        # Create a lookup set for O(1) company name matching
        self.company_names = {name.lower(): name for name in self.df['Company'].unique()}
        
        self.sectors_map = {
            "fintech": [
                "Payments", "Alternative Lending", "Banking Tech", "Investment Tech", 
                "Internet First Insurance Platforms", "Finance & Accounting Tech", "Cryptocurrencies"
            ],
            "logistics": ["Logistics Tech", "Road Transport Tech"],
            "ecommerce": ["Horizontal E-Commerce", "B2B E-Commerce", "Auto E-Commerce & Content", "Online Grocery"],
            "edtech": ["K-12 EdTech", "Test Preparation Tech", "Continued Learning"],
            "health": ["Healthcare Booking Platforms", "Infectious Diseases", "Healthcare IT"]
        }

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Loads CSV with encoding fallback (UTF-8 -> Latin-1).
        """
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} records from {path} (UTF-8)")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed. Retrying with Latin-1...")
            try:
                df = pd.read_csv(path, encoding='latin1')
                logger.info(f"Loaded {len(df)} records from {path} (Latin-1)")
            except Exception as e:
                logger.error(f"Fallback loading failed: {e}")
                raise e

        # Clean strings
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            df[col] = df[col].str.strip()
        return df

    def get_company_details(self, company_name_lower: str) -> pd.DataFrame:
        """Direct lookup for a specific company."""
        real_name = self.company_names.get(company_name_lower)
        if real_name:
            return self.df[self.df['Company'] == real_name]
        return pd.DataFrame()

    def filter_data(self, df_subset: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Filters a given DataFrame based on query keywords (Location, Sector, etc).
        Used for narrowing down context.
        """
        query = query.lower()
        results = df_subset.copy()

        # 1. Location Filtering
        if "bangalore" in query or "bengaluru" in query:
            results = results[results['location'].str.contains("Bengaluru", case=False, na=False)]
        elif "mumbai" in query:
            results = results[results['location'].str.contains("Mumbai", case=False, na=False)]
        elif "delhi" in query or "ncr" in query:
            results = results[results['location'].str.contains("Delhi|Noida|Gurugram", case=False, na=False)]

        return results

    def search_broad(self, query: str) -> pd.DataFrame:
        """
        Performs a broad search on the entire dataset using Sector mapping or Keywords.
        """
        query = query.lower()
        results = self.df.copy()
        
        # 1. Check Sector Mapping
        sector_filter = []
        for key, sectors in self.sectors_map.items():
            if key in query:
                sector_filter.extend(sectors)
        
        if sector_filter:
            results = results[results['primary_sector'].isin(sector_filter)]
            return results

        # 2. General Keyword Match if no sector found
        # (Exclude common stop words to avoid matching everything)
        stop_words = {"what", "does", "do", "tell", "me", "about", "is", "the", "a", "an"}
        keywords = [w for w in query.split() if w not in stop_words]
        
        if keywords:
            # Match if ANY keyword exists in the string representation of the row
            mask = results.apply(lambda x: x.astype(str).str.contains('|'.join(keywords), case=False, na=False)).any(axis=1)
            results = results[mask]
        
        return results

    def format_results(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No matching companies found."
        
        # Limit to 5 for context window
        top_results = df.head(5).to_dict(orient='records')
        context_str = "Found the following companies:\n"
        for res in top_results:
            context_str += (
                f"- Name: {res.get('Company')}\n"
                f"  Description: {res.get('company_background')}\n"
                f"  Sector: {res.get('primary_sector')}\n"
                f"  Location: {res.get('location')}\n\n"
            )
        return context_str

# --- LLM Client ---
class LLMClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def generate_response(self, system_prompt: str, user_query: str, context: str) -> str:
        # --- MOCK LLM LOGIC ---
        lower_query = user_query.lower()
        
        # 1. Ambiguity Check
        if any(x in lower_query for x in ["best", "top", "good", "suggest"]):
            if "fintech" not in lower_query and "logistics" not in lower_query:
                return "Could you verify which specific sector or criteria you are looking for? (e.g., Valuation, Sector, Location)"

        # 2. Contextual Response
        if "No matching companies" in context:
            return "I couldn't find any companies matching that description in the dataset."
        
        if "which of these" in lower_query:
            # The context string already contains the filtered list from Chatbot class
            # We just wrap it in natural language
            companies = re.findall(r"Name: (.*)", context)
            if companies:
                return f"Based on your previous query, the companies matching your criteria are: {', '.join(companies)}."
            return "None of the previously listed companies match that criterion."

        return f"Here is the information I found:\n\n{context}\nWould you like to know more about any of these?"

# --- Chatbot Core ---
class Chatbot:
    def __init__(self, data_file: str):
        self.data_engine = DataEngine(data_file)
        self.llm = LLMClient()
        self.history = [] 
        # State: Keep the actual DataFrame of the last result to filter "these"
        self.current_context_df: Optional[pd.DataFrame] = None 

    def sanitize_input(self, user_input: str) -> str:
        return re.sub(r'[^\w\s\?\.,-]', '', user_input).strip()

    def process_message(self, user_input: str) -> str:
        start_time = time.time()
        clean_input = self.sanitize_input(user_input)
        lower_input = clean_input.lower()

        response_df = pd.DataFrame()
        
        # --- Step 1: Intent Recognition ---
        
        # A. Check for Company Name (Priority 1: Entity Search)
        # This fixes "What does Razorpay do?"
        matched_company = None
        for name in self.data_engine.company_names:
            if name in lower_input:
                matched_company = name
                break
        
        is_follow_up = "these" in lower_input or "they" in lower_input

        # --- Step 2: Data Retrieval ---
        
        if matched_company:
            # Direct Entity Look up
            response_df = self.data_engine.get_company_details(matched_company)
            # Update context to this specific company
            self.current_context_df = response_df
            
        elif is_follow_up and self.current_context_df is not None:
            # Contextual Filtering (Priority 2: Filter the *previous* results)
            # This fixes the "Bangalore" issue by only searching inside current_context_df
            response_df = self.data_engine.filter_data(self.current_context_df, clean_input)
            # Do NOT overwrite current_context_df heavily, just refine it temporarily for display
            # Or strict narrowing:
            # self.current_context_df = response_df 
            
        else:
            # Broad Search (Priority 3: Keyword/Sector search)
            response_df = self.data_engine.search_broad(clean_input)
            if not response_df.empty:
                self.current_context_df = response_df

        # --- Step 3: Response Generation ---
        
        context_str = self.data_engine.format_results(response_df)
        response = self.llm.generate_response("", clean_input, context_str)

        # --- Step 4: Logging & Metrics ---
        latency = (time.time() - start_time) * 1000
        is_clarification = "?" in response and "verify" in response
        tracker.log_query(latency, is_clarification)
        logger.info(f"Query: {clean_input} | Entities: {matched_company} | FollowUp: {is_follow_up} | Rows: {len(response_df)}")

        return response

# --- Main ---
if __name__ == "__main__":
    if not os.path.exists("tracxn.csv"):
        print("Error: tracxn.csv not found.")
    else:
        bot = Chatbot("tracxn.csv")
        print("--- Unicorn Bot v2.0 (Context Aware) ---")
        while True:
            try:
                user_in = input("\nYou: ")
                if user_in.lower() in ["exit", "quit"]: break
                print(f"Bot: {bot.process_message(user_in)}")
            except Exception as e:
                print(f"Error: {e}")