import argparse
import numpy as np
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

# Config
QUERIES_FILE    = "sift/sift_query.npy"
COLLECTION_NAME = "sift1m_qdrant"
NUM_QUERIES     = 1000
THREADS_LIST    = [1, 2, 4, 8, 16, 24, 32]
EF_SEARCH       = 64
CSV_FILE        = f"resultados_concorrencia_qdrant_{args.metric}.csv"

print("A carregar queries...")
queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)

if args.metric == "cosine":
    norms   = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1
    queries = queries / norms

queries = queries.tolist()

# O QdrantClient é thread-safe por natureza, sendo não é necessário instanciar pools manualmente
client = QdrantClient(host="localhost", port=6333)

def worker_qdrant(query_vector):
    t0 = time.perf_counter()
    client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=10,
        search_params=SearchParams(hnsw_ef=EF_SEARCH)
    )
    return time.perf_counter() - t0

#Benchmark 
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Threads", "QPS", "p95_Latency_ms"])

for num_threads in THREADS_LIST:
    print(f"\n[Qdrant {args.metric.upper()}] A testar com {num_threads} Threads...")

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(worker_qdrant, queries)
        latencies = list(results)

    total_time = time.perf_counter() - start_time

    qps    = NUM_QUERIES / total_time
    lat_ms = np.array(latencies) * 1000
    p95    = np.percentile(lat_ms, 95)

    print(f"  Resultado: {qps:.0f} QPS | Latência p95: {p95:.2f} ms")

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([num_threads, round(qps, 2), round(p95, 2)])

print(f"\nConcluído! Guardado em {CSV_FILE}")
