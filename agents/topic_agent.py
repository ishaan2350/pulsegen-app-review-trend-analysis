import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import logging

class TopicExtractionAgent:
    def __init__(self, model_name='all-MiniLM-L6-v2', distance_threshold=0.5):
        """
        Args:
            model_name: SentenceTransformer model name.
            distance_threshold: Threshold for AgglomerativeClustering. 
                                Lower = more granular topics. 
                                Higher = broader topics.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.distance_threshold = distance_threshold
        self.logger.info("Model loaded.")

    def extract_topics(self, reviews: list[str]) -> list[dict]:
        """
        Extract topics from a batch of reviews.
        
        Returns:
            List of dicts: [{'topic': 'Login Issue', 'count': 5, 'exemplar': '...', 'reviews': [...]}, ...]
        """
        if not reviews:
            return []

        self.logger.info(f"Encoding {len(reviews)} reviews...")
        embeddings = self.model.encode(reviews, show_progress_bar=False)
        
        # Normalize embeddings for cosine similarity usage in clustering
        # (Agglomerative with euclidean on normalized vectors ~ cosine distance)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.logger.info("Clustering reviews...")
        # Using AgglomerativeClustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='euclidean', # Approximates 1 - cosine
            linkage='average',
            distance_threshold=self.distance_threshold
        )
        cluster_labels = clustering.fit_predict(embeddings)
        
        topics = []
        unique_labels = set(cluster_labels)
        
        self.logger.info(f"Found {len(unique_labels)} clusters.")

        for label_id in unique_labels:
            indices = np.where(cluster_labels == label_id)[0]
            cluster_reviews = [reviews[i] for i in indices]
            cluster_embeddings = embeddings[indices]
            
            # Find centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find exemplar (closest review to centroid)
            # We compute Cosine Sim between centroid and all cluster members
            sims = cosine_similarity([centroid], cluster_embeddings)[0]
            best_idx = np.argmax(sims)
            exemplar = cluster_reviews[best_idx]
            
            topic_entry = {
                'topic_label': exemplar, # Using the representative review as the temporary label
                'count': len(cluster_reviews),
                'reviews': cluster_reviews,
                'embedding': centroid.tolist() # Store centroid for future dedup
            }
            topics.append(topic_entry)
            
        return topics

if __name__ == "__main__":
    # Quick sanity check
    logging.basicConfig(level=logging.INFO)
    agent = TopicExtractionAgent()
    sample_reviews = [
        "The app crashes when I try to pay.",
        "Payment failed and app closed.",
        "I love the new dark mode.",
        "Dark mode looks great!",
        "Delivery was super late."
    ]
    results = agent.extract_topics(sample_reviews)
    for t in results:
        print(f"Topic: {t['topic_label']} | Count: {t['count']}")
