import logging
import random
import os
import pandas as pd
from datetime import datetime, timedelta
from agents.topic_agent import TopicExtractionAgent
from agents.dedup_agent import DedupAgent
from agents.evolution_agent import EvolutionAgent
from agents.trend_agent import TrendAgent
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DEMO")

class ReviewGenerator:
    def __init__(self):
        self.base_reviews = [
            "Food was great.", "Delivery was fast.", "Tasty food.", 
            "Driver was polite.", "Packaging could be better.",
            "I love this app.", "Best food delivery app.",
            "Rude delivery guy.", "Driver shouted at me.",
            "Too expensive.", "Delivery charges are high."
        ]
        
        self.trends = {
            "Login Bug": {
                "start_day": -5, # 5 days ago
                "end_day": 0,
                "reviews": [
                    "App crashes on login.", "Cannot login to my account.",
                    "Login failed, please fix.", "Error 500 on login screen.",
                    "Why can't I log in anymore?", "Update broke the login."
                ],
                "intensity": 0.6 # 60% of daily reviews will be this trend
            },
            "Cold Food": {
                "start_day": -20,
                "end_day": -15,
                "reviews": [
                    "Food came cold.", "Pizza was cold.", "Cold biryani sent.",
                    "Ice cold food delivered."
                ],
                "intensity": 0.3
            }
        }

    def generate_batch(self, date_offset: int, batch_size=20) -> list[str]:
        """
        Generate a batch of reviews for a specific day offset (0 = today, -30 = 30 days ago).
        """
        daily_reviews = []
        
        # Inject trends if active
        for trend, config in self.trends.items():
            if config['start_day'] <= date_offset <= config['end_day']:
                num_trend = int(batch_size * config['intensity'])
                daily_reviews.extend([random.choice(config['reviews']) for _ in range(num_trend)])
        
        # Fill rest with base noise
        remaining = batch_size - len(daily_reviews)
        if remaining > 0:
            daily_reviews.extend([random.choice(self.base_reviews) for _ in range(remaining)])
            
        random.shuffle(daily_reviews)
        return daily_reviews

def run_demo():
    print("========================================")
    print("      APP REVIEW TREND AI AGENT         ")
    print("========================================")
    
    # 1. Initialize Agents
    print("\n[System] Initializing Agents...")
    topic_agent = TopicExtractionAgent(distance_threshold=0.8) # Higher threshold = fewer, broader topics
    dedup_agent = DedupAgent(memory_path='data/topic_memory.json')
    evolution_agent = EvolutionAgent(memory_path='data/topic_memory.json')
    trend_agent = TrendAgent(storage_path='output/trend_report.csv')
    
    # Clean previous run
    if os.path.exists('data/topic_memory.json'): os.remove('data/topic_memory.json')
    trend_agent.clean_storage()

    generator = ReviewGenerator()
    target_date = datetime.now()
    days_to_simulate = 30
    
    print(f"\n[System] Starting {days_to_simulate}-day simulation...")
    
    # 2. Daily Loop
    for i in tqdm(range(days_to_simulate)):
        offset = i - (days_to_simulate - 1) # -29 to 0
        current_date = target_date + timedelta(days=offset)
        date_str = current_date.strftime('%Y-%m-%d')
        
        # A. Ingest
        reviews = generator.generate_batch(offset, batch_size=15)
        # logger.info(f"Day {offset} ({date_str}): Processed {len(reviews)} reviews.")
        
        # B. Extract
        extracted_topics = topic_agent.extract_topics(reviews)
        
        # C. Dedup
        processed_topics, candidates = dedup_agent.deduplicate(extracted_topics)
        
        # D. Evolve
        new_registered = evolution_agent.evolve(candidates, date_str)
        
        # E. Aggregate
        # Combine processed (existing) + new_registered (just added)
        all_topics = processed_topics + new_registered
        trend_agent.update(date_str, all_topics)

    print("\n[System] Simulation Complete.")
    
    # 3. Report
    print("\n[System] Generated Trend Report (Last 7 Days):")
    matrix = trend_agent.get_trend_matrix()
    
    # Display last 7 days columns
    recent_cols = matrix.columns[-7:]
    print(matrix[recent_cols].fillna(0).astype(int))
    
    print(f"\n[System] Full report saved to {trend_agent.storage_path}")

    # Verify meaningful output
    # Check if "Login Bug" is successfully detected in the last few days
    # We look for a topic label that resembles "Login"
    found_login = False
    for topic in matrix.index:
        if "login" in topic.lower():
            found_login = True
            # print(f"CONFIRMED: Detected '{topic}' spike.")
    
    if not found_login:
        print("WARNING: 'Login Bug' trend might have been clustered under a different name.")

if __name__ == "__main__":
    run_demo()
