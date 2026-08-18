"""
Benchmark de latência detalhada para Qdrant.
"""
import argparse
import os
import numpy as np
import time
import csv
from scipy.stats import skew, kurtosis
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

# Parse de argumentos - permite reutilizar o script para L2 e Cosseno
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

QUERIES_FILE = "sift/sift_query.npy"
NUM_QUERIES  = 10000 
EF_SEARCH    = 64     
CSV_FILE     = f"latencias_qdrant_{args.metric}.csv"

# Nome da coleção depende da métrica a escolher.
if args.metric == "l2":
    COLLECTION_NAME = "disk_test_l2"
else:
    COLLECTION_NAME = "disk_test_cosine"

#Load Queries
print(f"Carregando {NUM_QUERIES} queries...")
queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)

# Normalização obrigatória para Cosseno - sem isto, os resultados não estariam certos com o Ground Truth
if args.metric == "cosine":
    print("Normalizando os vetores (Cosine)...")
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Evitar divisão por zero em vetores nulos
    queries = queries / norms

queries = queries.tolist()

#Database Connection
client = QdrantClient(host="localhost", port=6333)

#Warmup
# O Qdrant usa mmap para mapear os ficheiros do disco para a RAM, 
# as primeiras queries são lentas por causa de page faults.
# Fazemos 500 queries de aquecimento para estabilizar a page cache do SO.
# Sem isto, o p50 fica com valores artificialmente elevados nas primeiras medições.
print("A executar warmup (500 queries para estabilizar page cache)...")
for q in queries[:500]:
    client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q,
        limit=10,
        search_params=SearchParams(hnsw_ef=EF_SEARCH)
    )

#Run Benchmark
print("Executando queries sequenciais (medindo latência pura)...")
latencies = []
for i, q in enumerate(queries):
    t0 = time.perf_counter()  # perf_counter tem resolução superior a time.time()
    client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q,
        limit=10,
        search_params=SearchParams(hnsw_ef=EF_SEARCH)
    )
    lat = (time.perf_counter() - t0) * 1000  # Converter para ms
    latencies.append(lat)
    
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1} / {NUM_QUERIES} processadas...")

#Stats Calculation
lat_np = np.array(latencies)
p = np.percentile(lat_np, [10, 25, 50, 75, 90, 95, 99, 99.9])
sk = skew(lat_np)
ku = kurtosis(lat_np)

#CSV Export

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Query_ID", "Latency_ms"])
    for i, lat in enumerate(latencies):
        writer.writerow([i + 1, round(lat, 4)])

RESUMO_FILE = "latencias_qdrant_resumo.csv"
resumo_existe = os.path.isfile(RESUMO_FILE)
with open(RESUMO_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not resumo_existe:
        writer.writerow([
            "Motor", "Metrica", "p10", "p25", "p50", "p75", "p90",
            "p95", "p99", "p99_9", "Skewness", "Kurtosis"
        ])
    writer.writerow([
        "Qdrant", args.metric,
        f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{p[3]:.4f}",
        f"{p[4]:.4f}", f"{p[5]:.4f}", f"{p[6]:.4f}", f"{p[7]:.4f}",
        f"{sk:.4f}", f"{ku:.4f}"
    ])
