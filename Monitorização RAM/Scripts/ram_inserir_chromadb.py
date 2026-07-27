import argparse
import numpy as np
import chromadb
import time

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

DATA_FILE = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS = 1000000
BATCH_SIZE = 5000

if args.metric == "l2":
    COLLECTION_NAME = "disk_test_l2"
    SPACE = "l2"
    NORMALIZE = False
else:
    COLLECTION_NAME = "disk_test_cosine"
    SPACE = "cosine"
    NORMALIZE = True

client = chromadb.HttpClient(host="localhost", port=8000)

print(f"A preparar coleção '{COLLECTION_NAME}' no ChromaDB local...")
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": SPACE, "hnsw:M": 16, "hnsw:construction_ef": 200}
)

print("A carregar vetores do disco...")
vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)

if NORMALIZE:
    print("A normalizar vetores (Cosseno)...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

vectors = vectors.tolist()

print("A iniciar a inserção em lotes...")
for i in range(0, TOTAL_VECTORS, BATCH_SIZE):
    batch_vectors = vectors[i : i + BATCH_SIZE]
    batch_ids = [str(j) for j in range(i, i + len(batch_vectors))]
    
    collection.add(embeddings=batch_vectors, ids=batch_ids)
    

print("\nFeito! A coleção ChromaDB foi criada com sucesso.")
