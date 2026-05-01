import numpy as np
import time
import csv
import docker
import os
import gc
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

QUERIES_FILE = "sift/sift_query.npy"
GT_FILE = "sift/gt/gt_1000000.npy"
CSV_FILE = "resultados_cold_warm_qdrant_10reps.csv"

METRIC_TEST = "L2"
QDRANT_COLLECTION = "cpu_test" 
NUM_QUERIES = 1000
REPLICATES = 10

queries = np.load(QUERIES_FILE)[:NUM_QUERIES]
gt = np.load(GT_FILE)[:NUM_QUERIES]

def calculate_recall(results, ground_truth):
    total_hits = sum(len(set(res) & set(gt_row[:10])) for res, gt_row in zip(results, ground_truth))
    return (total_hits / (len(results) * 10)) * 100

def wait_for_qdrant():
    start = time.time()
    client = QdrantClient(host="localhost", port=6333)
    while True:
        try:
            client.get_collections()
            return time.time() - start
        except:
            time.sleep(0.1)

headers = ["Replicate", "DB", "Metric", "First_Resp_Cold_ms", "First_Resp_Warm_ms", 
           "p50_Cold_ms", "p50_Warm_ms", "p95_Cold_ms", "p95_Warm_ms", 
           "Recall_Cold", "Recall_Warm", "Stabilization_Time_s", "Index_Load_Time_s"]

with open(CSV_FILE, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=headers).writeheader()

docker_client = docker.from_env()

for r in range(1, REPLICATES + 1):
    print(f"\n[QDRANT] A INICIAR a replica {r}/{REPLICATES}")
    os.system("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    time.sleep(1)
    
    container = docker_client.containers.get("qdrant")
    container.restart()
    load_time = wait_for_qdrant()
    
    client = QdrantClient(host="localhost", port=6333)

    cold_latencies, cold_results = [], []
    t_start_cold = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        res_raw = client.search(collection_name=QDRANT_COLLECTION, query_vector=q, limit=10, search_params=SearchParams(hnsw_ef=64))
        cold_results.append([int(hit.id) for hit in res_raw])
        cold_latencies.append(time.perf_counter() - t0)
    cold_total_time = time.perf_counter() - t_start_cold

    warm_latencies, warm_results = [], []
    for q in queries:
        t0 = time.perf_counter()
        res_raw = client.search(collection_name=QDRANT_COLLECTION, query_vector=q, limit=10, search_params=SearchParams(hnsw_ef=64))
        warm_results.append([int(hit.id) for hit in res_raw])
        warm_latencies.append(time.perf_counter() - t0)

    cold_lats_ms, warm_lats_ms = np.array(cold_latencies) * 1000, np.array(warm_latencies) * 1000

    metrics = {
        "Replicate": r, "DB": "Qdrant", "Metric": METRIC_TEST,
        "First_Resp_Cold_ms": cold_lats_ms[0], "First_Resp_Warm_ms": warm_lats_ms[0],
        "p50_Cold_ms": np.percentile(cold_lats_ms, 50), "p50_Warm_ms": np.percentile(warm_lats_ms, 50),
        "p95_Cold_ms": np.percentile(cold_lats_ms, 95), "p95_Warm_ms": np.percentile(warm_lats_ms, 95),
        "Recall_Cold": calculate_recall(cold_results, gt), "Recall_Warm": calculate_recall(warm_results, gt),
        "Stabilization_Time_s": max(0, cold_total_time - sum(warm_latencies)),
        "Index_Load_Time_s": load_time
    }

    with open(CSV_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writerow(metrics)

    del cold_latencies, cold_results, warm_latencies, warm_results, cold_lats_ms, warm_lats_ms, client
    gc.collect()