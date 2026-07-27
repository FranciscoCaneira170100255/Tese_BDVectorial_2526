import argparse
import numpy as np
import time
import csv
import chromadb
from concurrent.futures import ThreadPoolExecutor

#Args
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

#Config
QUERIES_FILE    = "sift/sift_query.npy"
COLLECTION_NAME = "sift1m_chroma"
NUM_QUERIES     = 1000
THREADS_LIST    = [1, 2, 4, 8, 16, 24, 32]
CSV_FILE        = f"resultados_concorrencia_chromadb_{args.metric}.csv"

print("A carregar queries...")
queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)

if args.metric == "cosine":
    norms   = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1
    queries = queries / norms

queries = queries.tolist()

# O HttpClient faz a gestão da multiplexagem e as ligações HTTP internamente (via requests)
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection(COLLECTION_NAME)

def worker_chromadb(query_vector):
    t0 = time.perf_counter()
    #Ao comparar com o Qdrant ou pgvector, o ChromaDB não mostra a alteração do 'ef_search'
    #dinamicamente na query da API, herdando assim a configuração que demos no momento da indexação
    collection.query(
        query_embeddings=[query_vector],
        n_results=10
    )
    return time.perf_counter() - t0

#Benchmark 
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Threads", "QPS", "p95_Latency_ms"])

for num_threads in THREADS_LIST:
    print(f"\n[ChromaDB {args.metric.upper()}] A testar com {num_threads} Threads...")

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(worker_chromadb, queries)
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
