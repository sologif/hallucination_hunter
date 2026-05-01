import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class HybridSearchDB:
    def __init__(self, collection_name="hallucination_hunter"):
        # Initialize in-memory Qdrant client for the demo
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        # Use a small but effective model
        self.client.set_model("BAAI/bge-small-en-v1.5")
        
        self.setup_collection()
        self.ingest_sample_data()

    def setup_collection(self):
        # Only setup dense vectors to save memory
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.client.get_fastembed_vector_params()
            )

    def ingest_sample_data(self):
        # Sample Ground Truth knowledge base
        documents = [
            "The 2019 Cambridge Ornithology Review found that the average airspeed velocity of a European swallow is roughly 11 meters per second, or 24 miles per hour.",
            "The 2019 Cambridge Ornithology Review primarily focused on migration patterns of European swallows.",
            "The 2019 Cambridge Ornithology Review did not include any experiments involving load-bearing, such as carrying coconuts.",
            "African swallows are non-migratory and have different average velocities, but exact coconut load-bearing data is inconclusive.",
            "Hallucination in AI occurs when an AI generates information that sounds right but is factually wrong, unsupported, or completely made up."
        ]
        
        self.client.add(
            collection_name=self.collection_name,
            documents=documents,
            metadata=[{"source": f"Doc-{i}", "chunk_id": i} for i in range(len(documents))],
            ids=[i for i in range(len(documents))]
        )

    def search(self, query: str, limit: int = 3):
        # fastembed handles query embedding automatically
        search_result = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=limit
        )
        
        results = []
        for point in search_result:
            results.append({
                "text": point.document,
                "score": point.score,
                "metadata": point.metadata
            })
        return results

# Singleton instance
db = HybridSearchDB()
