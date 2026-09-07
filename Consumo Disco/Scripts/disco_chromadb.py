import argparse
import numpy as np
import chromadb
import time
import csv
import os
import sqlite3
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

DATA_FILE = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS = 1_000_000
BATCH_SIZE = 5000

if args.metric == "l2":
    COLLECTION_NAME = "disk_test_l2"
    SPACE = "l2"
    NORMALIZE = False
    CSV_FILE = "disk_chromadb_l2.csv"
    CHROMA_PATH = "./chroma_disk_test_l2"
else:
    COLLECTION_NAME = "disk_test_cosine"
    SPACE = "cosine"
    NORMALIZE = True
    CSV_FILE = "disk_chromadb_cosine.csv"
    CHROMA_PATH = "./chroma_disk_test_cosine"

def get_file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0

def get_dir_size_mb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            fp = os.path.join(root, file)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)

if NORMALIZE:
    print("A normalizar vetores (Cosseno)...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

# Limpar pasta anterior se existir
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": SPACE, "hnsw:M": 16, "hnsw:construction_ef": 200}
)

print("Inserindo vetores...")
for i in range(0, TOTAL_VECTORS, BATCH_SIZE):
    batch = vectors[i:i + BATCH_SIZE]
    collection.add(
        ids=[str(x) for x in range(i, i + len(batch))],
        embeddings=batch.tolist()
    )
    if (i + BATCH_SIZE) % 50000 == 0:
        print(f"  {i + BATCH_SIZE} / {TOTAL_VECTORS} guardados...")

# Obriga a um checkpoint do SQLite para garantir que o ficheiro WAL é descarregado
print("A forcar checkpoint do SQLite...")
try:
    conn = sqlite3.connect(f"{CHROMA_PATH}/chroma.sqlite3")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
except Exception as e:
    print(f"Aviso ao forcar checkpoint: {e}")

time.sleep(5) # Esperar para garantir que os ficheiros foram atualizados no disco

# Extração de métricas
wal_mb = get_file_size_mb(f"{CHROMA_PATH}/chroma.sqlite3-wal")

hnsw_mb = 0.0
for root, dirs, files in os.walk(CHROMA_PATH):
    for file in files:
        if file.endswith(".bin") or file.endswith(".hnsw") or "index" in file.lower():
            hnsw_mb += get_file_size_mb(os.path.join(root, file))

total_mb = get_dir_size_mb(CHROMA_PATH)
metadata_mb = max(0, total_mb - hnsw_mb - wal_mb)

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DB", "Metric", "Index_MB", "Metadata_MB", "WAL_MB", "Total_MB"])
    writer.writerow([
        "ChromaDB", args.metric.upper(),
        round(hnsw_mb, 2), round(metadata_mb, 2), 
        round(wal_mb, 2), round(total_mb, 2)
    ])

print(f"\nFeito! -> {CSV_FILE}")
print(f"Index_MB (vec+grafo): {hnsw_mb:.2f}")
print(f"Metadata_MB (inclui SQLite): {metadata_mb:.2f}")
print(f"WAL_MB      : {wal_mb:.2f}")
print(f"Total_MB    : {total_mb:.2f}")
