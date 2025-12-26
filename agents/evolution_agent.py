import json
import os
import logging
from datetime import datetime

class EvolutionAgent:
    def __init__(self, memory_path='data/topic_memory.json'):
        self.memory_path = memory_path
        self.logger = logging.getLogger(__name__)

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return []

    def _save_memory(self, memory):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, 'w') as f:
            json.dump(memory, f, indent=2)

    def evolve(self, new_candidates: list[dict], current_date: str) -> list[dict]:
        """
        Register new topics into memory.
        
        Args:
            new_candidates: List of topics that failed deduplication.
            current_date: Date string YYYY-MM-DD.
            
        Returns:
            registered_topics: The candidates, now considered registered.
        """
        if not new_candidates:
            return []

        memory = self._load_memory()
        
        # Simple strategy: Accept all strong clusters as new topics
        # In a real system, we might ask for human approval or check frequency thresholds.
        # Here we assume TopicExtractionAgent already filtered for density.
        
        registered = []
        for cand in new_candidates:
            new_entry = {
                'label': cand['topic_label'],
                'embedding': cand['embedding'],
                'first_seen': current_date,
                'exemplar': cand.get('reviews', [''])[0]
            }
            memory.append(new_entry)
            registered.append(cand)
            self.logger.info(f"Registered NEW topic: {cand['topic_label']}")

        self._save_memory(memory)
        return registered
