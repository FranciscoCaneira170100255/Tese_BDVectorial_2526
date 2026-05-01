import numpy as np
import time
import csv
import docker
import os
import gc
import psycopg2

QUERIES_FILE = "sift/sift_query.npy"
GT_FILE = "sift/gt/gt_1000000.npy"
CSV_FILE = "resultados_cold_warm_pgvector_10reps_cosine.csv"

METRIC_TEST = "cosine"
PG_TABLE = "items"
NUM_QUERIES = 1000
REPLICATES = 10

queries = np.load(QUERIES_FILE)[:NUM_QUERIES]
gt = np.load(GT_FILE)[:NUM_QUERIES]

def calculate_recall(results, ground_truth):
    total_hits = sum(len(set(res) & set(gt_row[:10])) for res, gt_row in zip(results, ground_truth))
    return (total_hits / (len(results) * 10)) * 100

def wait_for_pg():
    start = time.time()
    while True:
        try:
            conn = psycopg2.connect(dbname="vectordb", user="postgres", password="postgres", host="localhost")
            conn.close()
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
    print(f"\n[PGVECTOR] A INICIAR replica {r}/{REPLICATES}")
    os.system("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    time.sleep(1)
    
    container = docker_client.containers.get("pgvector")
    container.restart()
    load_time = wait_for_pg()
    
    conn = psycopg2.connect(dbname="vectordb", user="postgres", password="postgres", host="localhost")
    cur = conn.cursor()
    query_sql = f"SELECT id FROM {PG_TABLE} ORDER BY embedding {'<->' if METRIC_TEST == 'L2' else '<=>'} %s::vector LIMIT 10;"

    cold_latencies, cold_results = [], []
    t_start_cold = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        cur.execute(query_sql, (q.tolist(),))
        cold_results.append([row[0] - 1 for row in cur.fetchall()])
        cold_latencies.append(time.perf_counter() - t0)
    cold_total_time = time.perf_counter() - t_start_cold

    warm_latencies, warm_results = [], []
    for q in queries:
        t0 = time.perf_counter()
        cur.execute(query_sql, (q.tolist(),))
        warm_results.append([row[0] - 1 for row in cur.fetchall()])
        warm_latencies.append(time.perf_counter() - t0)

    cur.close()
    conn.close()

    cold_lats_ms, warm_lats_ms = np.array(cold_latencies) * 1000, np.array(warm_latencies) * 1000
    
    metrics = {
        "Replicate": r, "DB": "pgvector", "Metric": METRIC_TEST,
        "First_Resp_Cold_ms": cold_lats_ms[0], "First_Resp_Warm_ms": warm_lats_ms[0],
        "p50_Cold_ms": np.percentile(cold_lats_ms, 50), "p50_Warm_ms": np.percentile(warm_lats_ms, 50),
        "p95_Cold_ms": np.percentile(cold_lats_ms, 95), "p95_Warm_ms": np.percentile(warm_lats_ms, 95),
        "Recall_Cold": calculate_recall(cold_results, gt), "Recall_Warm": calculate_recall(warm_results, gt),
        "Stabilization_Time_s": max(0, cold_total_time - sum(warm_latencies)),
        "Index_Load_Time_s": load_time
    }

    with open(CSV_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writerow(metrics)

    del cold_latencies, cold_results, warm_latencies, warm_results, cold_lats_ms, warm_lats_ms
    gc.collect()