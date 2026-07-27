import argparse
import numpy as np
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

DATA_FILE = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS = 1000000

if args.metric == "l2":
    COLLECTION_NAME = "disk_test_l2"
    DISTANCE = Distance.EUCLID
    NORMALIZE = False
else:
    COLLECTION_NAME = "disk_test_cosine"
    DISTANCE = Distance.COSINE
    NORMALIZE = True

client = QdrantClient(host="localhost", port=6333)

print(f"A preparar colecao '{COLLECTION_NAME}' no Qdrant...")


if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)


client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=128, distance=DISTANCE),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=200)
)

print("A carregar vetores do disco...")
vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)

if NORMALIZE:
    print("A normalizar vetores (Cosseno)...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

print("A começar a  inserção (aguarde)...")

client.upload_collection(
    collection_name=COLLECTION_NAME,
    vectors=vectors,
    batch_size=5000
)

print("\nConcluído! O índice HNSW foi construído com sucesso.")
