"""
Benchmark de latência detalhada para ChromaDB.
"""
import argparse
import numpy as np
import time
import csv
import chromadb
from scipy.stats import skew, kurtosis

#Args
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

QUERIES_FILE = "sift/sift_query.npy"
NUM_QUERIES  = 10000
EF_SEARCH    = 64
CSV_FILE     = f"latencias_chromadb_{args.metric}.csv"

if args.metric == "l2":
    COLLECTION_NAME = "disk_test_l2"
else:
    COLLECTION_NAME = "disk_test_cosine"

#Load Queries
print(f"Carregando {NUM_QUERIES} queries...")
queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)

if args.metric == "cosine":
    print("Normalizando vetores (Cosine)...")
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1
    queries = queries / norms

#Aqui o API do ChromaDB precisa de listas normais de Python e não arrays que sejam Numpy
queries = queries.tolist()

#Database connection
client = chromadb.HttpClient(host="localhost", port=8000)

#Definir o ef_search=64, por via de Metadata
print(f"Definindo ef_search={EF_SEARCH} via metadata da coleção...")
try:
    collection = client.get_collection(COLLECTION_NAME)
    collection.modify(metadata={"hnsw:search_ef": EF_SEARCH})
    print(f"Coleção '{COLLECTION_NAME}' atualizada com ef_search={EF_SEARCH}")
    # O ChromaDB ignora, por vezes, as alterações dinâmicas do ef_search sem um restart ao serviço
except Exception as e:
    print(f"ERRO: {e}")
    raise e

#Warmup necessário, para inicalizar a ligação HTTP
#e força a biblioteca hnswlib a alocar estas estruturas em memória
print("A executar warmup (200 queries para estabilizar HTTP/hnswlib)...")
for q in queries[:200]:
    collection.query(
        query_embeddings=[q],
        n_results=10
    )

#Run benchmark
print("Executando queries sequenciais (medindo latência pura)...")
latencies = []
for i, q in enumerate(queries):
    #Importante para medir a precisão na casa dos microssegundos
    t0 = time.perf_counter()
    collection.query(
        query_embeddings=[q],
        n_results=10
    )
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1} / {NUM_QUERIES} processadas...")

# Stats Calculation
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

#Print Results
print("\nFeito! ->", CSV_FILE)
print("\n=== RESULTADOS ===")
print(f"p10   : {p[0]:.2f} ms")
print(f"p25   : {p[1]:.2f} ms")
print(f"p50   : {p[2]:.2f} ms (Mediana)")
print(f"p75   : {p[3]:.2f} ms")
print(f"p90   : {p[4]:.2f} ms")
print(f"p95   : {p[5]:.2f} ms")
print(f"p99   : {p[6]:.2f} ms")
print(f"p99.9 : {p[7]:.2f} ms")
print("-" * 30)
print(f"Assimetria (Skewness) : {sk:.2f}")
print(f"Curtose (Kurtosis)    : {ku:.2f}")
