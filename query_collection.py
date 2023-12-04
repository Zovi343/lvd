import chromadb
from chromadb.config import Settings
#%%
client = chromadb.Client()
#%%
# Create a new chroma collection
collection_name = "new_colleciton"
collection = client.create_collection(name=collection_name)

collection.add(
    embeddings=[
        [1.5, 4.3, 7.2],
        [6.5, 5.9, 3.4],
    ],
    metadatas=[
        {"status": "read"},
        {"status": "unread"},
    ],
    documents=["A document that discusses domestic policy", "A document that discusses international affairs"],
    ids=[ "id9", "id10"],
)

results = collection.query(
    query_embeddings=[1.1, 2.3, 3.2],
    include=["documents", 'embeddings'],
    n_results=2
)

print(results)