import json
import os
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

class DedupAgent:
    def __init__(self, memory_path='data/topic_memory.json', similarity_threshold=0.85):
        self.memory_path = memory_path
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return [] # List of {'label': str, 'embedding': list}

    def save_memory(self):
        # Evolution agent usually handles adding, but we might need to save if we update centroids?
        # For now, we just read. Evolution agent will write.
        pass

    def deduplicate(self, day_topics: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Consolidate topics against memory.
        
        Args:
            day_topics: Output from TopicExtractionAgent.
            
        Returns:
            (processed_topics, new_candidates)
            processed_topics: Topics that matched existing memory (label updated).
            new_candidates: Topics that did not match any existing memory.
        """
        processed = []
        candidates = []

        # 1. Always apply normalization first
        for topic in day_topics:
            self._apply_semantic_rules(topic)

        if not self.memory:
            # If memory is empty, everything is a candidate (but now normalized)
            return [], day_topics

        memory_embeddings = np.array([m['embedding'] for m in self.memory])
        memory_labels = [m['label'] for m in self.memory]

        for topic in day_topics:
            topic_vec = np.array(topic['embedding']).reshape(1, -1)
            
            # Compute similarity against all memory
            sims = cosine_similarity(topic_vec, memory_embeddings)[0]
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]

            if best_sim >= self.similarity_threshold:
                # Match found
                canonical_label = memory_labels[best_idx]
                self.logger.info(f"Merged '{topic['topic_label']}' -> '{canonical_label}' (Sim: {best_sim:.2f})")
                
                # Update label to canonical
                topic['topic_label'] = canonical_label
                topic['is_new'] = False
                processed.append(topic)
            else:
                # No match
                self.logger.info(f"New candidate: '{topic['topic_label']}' (Max Sim: {best_sim:.2f})")
                topic['is_new'] = True
                candidates.append(topic)
                
        return processed, candidates

    def _apply_semantic_rules(self, topic: dict):
        """
        Hard-coded semantic rules to force consolidation of known distinct-but-same topics.
        """
        label_lower = topic['topic_label'].lower()
        
        # Rule 1: Login Issues
        if any(x in label_lower for x in ['login', 'log in', 'error 500', 'crashes', 'cannot login']):
            topic['topic_label'] = "Login issues"
            
        # Rule 2: Delivery Cost
        if any(x in label_lower for x in ['too expensive', 'charges are high']):
            topic['topic_label'] = "High delivery cost"
