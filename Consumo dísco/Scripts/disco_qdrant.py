import argparse
import numpy as np
import time
import csv
import subprocess
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

CONTAINER_NAME  = "qdrant"
DATA_FILE       = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS   = 1_000_000
EF_CONSTRUCTION = 200
M_VALUE         = 16

if args.metric == "l2":
    COLLECTION_NAME = "disk_test_l2"
    DISTANCE = Distance.EUCLID
    NORMALIZE = False
    CSV_FILE = "disk_qdrant_l2.csv"
else:
    COLLECTION_NAME = "disk_test_cosine"
    DISTANCE = Distance.COSINE
    NORMALIZE = True
    CSV_FILE = "disk_qdrant_cosine.csv"

def get_size_mb(container_name, path):
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "du", "-sb", path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0]) / (1024 * 1024)
    except:
        pass
    return 0.0

client = QdrantClient(host="localhost", port=6333, timeout=300)
vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)

if NORMALIZE:
    print("A normalizar vetores (Cosseno)...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

ids = list(range(TOTAL_VECTORS))

print(f"Removendo a antiga colecao '{COLLECTION_NAME}'...")
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

print("Criando colecao...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=128, distance=DISTANCE),
    hnsw_config=HnswConfigDiff(m=M_VALUE, ef_construct=EF_CONSTRUCTION)
)

print("Inserindo vetores...")
client.upload_collection(
    collection_name=COLLECTION_NAME,
    vectors=vectors,
    ids=ids,
    batch_size=10000
)
# Processo do Qdrant em background, garantido assim que o mesmo guarde tudo
print("Aguardando persistencia (30s)...")
time.sleep(30)

base_path = f"/qdrant/storage/collections/{COLLECTION_NAME}/0"

# Medir a pasta inteira da coleção
collection_total_mb = get_size_mb(CONTAINER_NAME, f"/qdrant/storage/collections/{COLLECTION_NAME}")
segments_mb = get_size_mb(CONTAINER_NAME, f"{base_path}/segments")
wal_mb = get_size_mb(CONTAINER_NAME, f"{base_path}/wal")

# O que sobrar da pasta, assume-se que são configurações e metadados
metadata_mb = max(0.0, collection_total_mb - segments_mb - wal_mb)

total_mb = segments_mb + wal_mb + metadata_mb

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DB", "Metric", "Segments_MB", "WAL_MB", "Metadata_MB", "Total_MB"])
    writer.writerow([
        "Qdrant", args.metric.upper(),
        round(segments_mb, 2), round(wal_mb, 2), 
        round(metadata_mb, 2), round(total_mb, 2)
    ])

print(f"\nFeito! -> {CSV_FILE}")
print(f"Segments_MB : {segments_mb:.2f}")
print(f"WAL_MB      : {wal_mb:.2f}")
print(f"Metadata_MB : {metadata_mb:.2f}")
print(f"Total_MB    : {total_mb:.2f}")
