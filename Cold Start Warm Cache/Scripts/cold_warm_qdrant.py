"""
Este script avalia o desempenho do Qdrant em cenarios de Cold Start e
Warm Start utilizando o dataset SIFT.
"""
import argparse
import numpy as np
import time
import csv
import docker
import subprocess
import gc
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

#Config
QUERIES_FILE      = "sift/sift_query.npy"
NUM_QUERIES       = 1000
REPLICATES        = 10
QDRANT_COLLECTION = "cpu_test"
CONTAINER_NAME    = "qdrant"


METRIC_CONFIG = {
    "l2": {
        "label":      "L2",
        "gt_file":    "sift/gt/gt_1000000.npy",
        "normalize":  False,
        "csv_file":   "resultados_cold_warm_qdrant_L2.csv",
    },
    "cosine": {
        "label":      "Cosine",
        "gt_file":    "sift/gt_cosine/gt_1000000.npy",
        "normalize":  True,
        "csv_file":   "resultados_cold_warm_qdrant_Cosine.csv",
    },
}

HEADERS = [
    "Replicate", "DB", "Metric",
    "First_Resp_Cold_ms", "First_Resp_Warm_ms",
    "Cold_First10_Mean_ms",
    "p50_Cold_ms", "p50_Warm_ms",
    "p95_Cold_ms", "p95_Warm_ms",
    "Recall_Cold", "Recall_Warm",
    "Stabilization_Time_s", "Index_Load_Time_s",
]

#Utils
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def calculate_recall(results, ground_truth, k=10) -> float:
    hits = sum(len(set(res) & set(gt_row[:k])) for res, gt_row in zip(results, ground_truth))
    return (hits / (len(results) * k)) * 100

def wait_for_qdrant() -> float:
    start  = time.time()
    client = QdrantClient(host="localhost", port=6333)
    while True:
        try:
            client.get_collections()
            return time.time() - start
        except Exception:
            time.sleep(0.1)

def drop_os_page_cache() -> None:
    #O Qdrant usa mmap, delegando assim a gestão ao SO,
    #assim limpar a page cache é a única forma que haja um cold start em condições.
    sync_result = subprocess.run(["sync"], capture_output=True, text=True)
    if sync_result.returncode != 0:
        raise RuntimeError(
            f"Falha ao executar 'sync' antes da limpeza da cache: "
            f"{sync_result.stderr.strip()}"
        )

    drop_result = subprocess.run(
        ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        capture_output=True, text=True
    )
    if drop_result.returncode != 0:
        raise RuntimeError(
            "Falha ao limpar a page cache do SO (drop_caches). "
            "A medição de Cold Start ficara assim inválida. Verifique as "
            "permissoes de 'sudo' (NOPASSWD) para este comando. "
            f"Detalhe do erro: {drop_result.stderr.strip()}"
        )

    time.sleep(1)

#Benchmark
def run_cold_warm(metric: str):
    cfg        = METRIC_CONFIG[metric]
    docker_cli = docker.from_env()

    queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)
    if cfg["normalize"]:
        queries = normalize(queries)
    gt = np.load(cfg["gt_file"])[:NUM_QUERIES]

    with open(cfg["csv_file"], "w", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writeheader()

    for r in range(1, REPLICATES + 1):
        print(f"\n[QDRANT {cfg['label']}] Replica {r}/{REPLICATES}")

        drop_os_page_cache()

        # Reiniciar container para simular arranque a frio real
        container = docker_cli.containers.get(CONTAINER_NAME)
        container.restart()
        load_time = wait_for_qdrant()

        client = QdrantClient(host="localhost", port=6333)
        # Não se utiliza o warmup inicial nesta parte 
        # porque assim havia uma inflação artificial
        # na primeira cold query do mmap, já que partilham os mesmos entrypoints do HNSW
        
        #Fase Cold
        cold_latencies, cold_results = [], []
        t_start_cold = time.perf_counter()
        for q in queries:
            t0      = time.perf_counter()
            res_raw = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=q,
                limit=10,
                search_params=SearchParams(hnsw_ef=64)
            )
            cold_results.append([int(hit.id) for hit in res_raw])
            cold_latencies.append(time.perf_counter() - t0)
        cold_total_time = time.perf_counter() - t_start_cold

        #Fase Warm
        warm_latencies, warm_results = [], []
        for q in queries:
            t0      = time.perf_counter()
            res_raw = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=q,
                limit=10,
                search_params=SearchParams(hnsw_ef=64)
            )
            warm_results.append([int(hit.id) for hit in res_raw])
            warm_latencies.append(time.perf_counter() - t0)

        cold_ms = np.array(cold_latencies) * 1000
        warm_ms = np.array(warm_latencies) * 1000

        # Realiza a média das primeiras 10 queries,
        #permitindo suavizar outliers de rede ou o I/O que necessitam de mais esforços
        cold_first10_mean = round(float(np.mean(cold_ms[:10])), 4)

        row = {
            "Replicate":            r,
            "DB":                   "Qdrant",
            "Metric":               cfg["label"],
            "First_Resp_Cold_ms":   round(cold_ms[0], 4),
            "First_Resp_Warm_ms":   round(warm_ms[0], 4),
            "Cold_First10_Mean_ms": cold_first10_mean,
            "p50_Cold_ms":          round(np.percentile(cold_ms, 50), 4),
            "p50_Warm_ms":          round(np.percentile(warm_ms, 50), 4),
            "p95_Cold_ms":          round(np.percentile(cold_ms, 95), 4),
            "p95_Warm_ms":          round(np.percentile(warm_ms, 95), 4),
            "Recall_Cold":          round(calculate_recall(cold_results, gt), 2),
            "Recall_Warm":          round(calculate_recall(warm_results, gt), 2),
            "Stabilization_Time_s": round(max(0, cold_total_time - sum(warm_latencies)), 4),
            "Index_Load_Time_s":    round(load_time, 4),
        }

        with open(cfg["csv_file"], "a", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writerow(row)

        print(f"  Cold p50={row['p50_Cold_ms']:.2f}ms | "
              f"Warm p50={row['p50_Warm_ms']:.2f}ms | "
              f"Recall Cold={row['Recall_Cold']:.1f}%")

        del cold_latencies, cold_results, warm_latencies, warm_results, cold_ms, warm_ms, client
        gc.collect()

    print(f"\nFeito! Ficheiro: {cfg['csv_file']}")

#Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold/Warm Qdrant")
    parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
    args = parser.parse_args()
    run_cold_warm(args.metric)
