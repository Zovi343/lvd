import chromadb
import pandas as pd
from chromadb.config import Settings

# Load the CSV file
csv_file_path = '../data/random_embeddings.csv'
data = pd.read_csv(csv_file_path)

# Initialize the Chroma client
client = chromadb.Client()

# Create a new Chroma collection
collection_name = "new_collection"
collection = client.create_collection(name=collection_name)

# Assuming 'embeddings' are the first three columns
# 'status' is the fourth column, 'document' is the fifth column, and 'id' is the sixth column
collection.add(
    embeddings=data[['embedding1', 'embedding2', 'embedding3']].values.tolist(),
    metadatas=[{"status": status} for status in data['status']],
    documents=data['document'].values.tolist(),
    ids=data['id'].values.tolist(),
)

collection.build_index()

results = collection.query(
    query_embeddings=[0.5488135039273248,0.7151893663724195,0.6027633760716439],
    include=["metadatas", 'embeddings', 'distances'],
    where={"status": "read"},
    n_results=10,
    n_buckets=1,
    bruteforce_threshold=0.0, # TODO: this causes problems, hotfix applied for now
)

print(results['ids'])
print(results['distances'])
print(results['metadatas'])