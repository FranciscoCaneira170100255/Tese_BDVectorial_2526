"""
BENCHMARK: Qdrant | Métrica configurável: Euclidiana (L2) ou Cosseno
Uso:
python bench_qdrant.py --metric l2
python bench_qdrant.py --metric cosine
"""
import argparse
import numpy as np
import time
import csv
import docker
import threading
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff, SearchParams
)

# --- Configuração -------------------------------------------------------------
CONTAINER_NAME      = "qdrant"
SCALES              = [100, 1_000, 10_000, 100_000, 1_000_000]
REPLICATES          = 10
QUERY_FILE          = "sift/sift_query.npy"
WARMUP_QUERIES      = 50
INDEX_WAIT_TIMEOUT  = 600
INDEX_POLL_INTERVAL = 2

METRIC_CONFIG = {
    "l2": {
        "distance":     Distance.EUCLID,
        "gt_dir":       "sift/gt",
        "normalize":    False,
        "results_file": "results_qdrant_l2.csv",
    },
    "cosine": {
        "distance":     Distance.COSINE,
        "gt_dir":       "sift/gt_cosine",
        "normalize":    True,
        "results_file": "results_qdrant_cosine.csv",
    },
}

# --- Docker Monitor -----------------------------------------------------------
class DockerMonitor(threading.Thread):
    def __init__(self, container_name):
        super().__init__()
        self.container      = docker.from_env().containers.get(container_name)
        self.memory_samples = []
        self.stop_flag      = False

    def run(self):
        while not self.stop_flag:
            try:
                stats = self.container.stats(stream=False)
                mem   = stats['memory_stats']['usage'] / (1024 ** 3)
                self.memory_samples.append(mem)
                time.sleep(0.5)
            except Exception:
                break

    def stop(self):
        self.stop_flag = True

    def get_results(self):
        if not self.memory_samples:
            return 0.0, 0.0
        return np.mean(self.memory_samples), np.max(self.memory_samples)


def get_baseline_ram(container_name: str) -> float:
    container = docker.from_env().containers.get(container_name)
    return container.stats(stream=False)['memory_stats']['usage'] / (1024 ** 3)


# --- Espera pela indexação assíncrona -----------------------------------------
def wait_for_index(client: QdrantClient, collection_name: str) -> None:
    """
    O Qdrant constrói o grafo HNSW de forma assíncrona após o upload.
    Esta função aguarda até o status da coleção virar para 'green',
    garantindo a medição do tempo de indexação de forma igual.
    """
    deadline = time.time() + INDEX_WAIT_TIMEOUT
    while time.time() < deadline:
        info   = client.get_collection(collection_name)
        status = info.status.value if hasattr(info.status, 'value') else str(info.status)
        if status == "green":
            return
        time.sleep(INDEX_POLL_INTERVAL)
    print(f" Timeout a aguardar indexação após {INDEX_WAIT_TIMEOUT}s.")


# --- Utils --------------------------------------------------------------------
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def calculate_recall(results, ground_truth, k=10) -> float:
    hits = sum(
        len(set(results[i]) & set(ground_truth[i][:k]))
        for i in range(len(results))
    )
    return hits / (len(results) * k)


# --- Benchmark ----------------------------------------------------------------
def run_bench(metric: str):
    cfg    = METRIC_CONFIG[metric]
    client = QdrantClient(host="localhost", port=6333)
    query_vecs = np.load(QUERY_FILE).astype(np.float32)
    if cfg["normalize"]:
        query_vecs = normalize(query_vecs)

    print(f"\nA iniciar benchmark para métrica: {metric.upper()}")

    with open(cfg["results_file"], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Replicate", "Scale", "Index_Time_s",
            "RAM_Avg_GB", "RAM_Peak_GB",
            "p50_ms", "p95_ms", "p99_ms",
            "QPS", "Recall@10"
        ])

        for r in range(1, REPLICATES + 1):
            print(f"\n=== RÉPLICA {r} ===")
            for n in SCALES:
                print(f"> QDRANT [{metric.upper()}] | Escala: {n}")

                vecs    = np.load(f"sift/subsets/sift_base_{n}.npy").astype(np.float32)
                gt_vecs = np.load(f"{cfg['gt_dir']}/gt_{n}.npy")
                if cfg["normalize"]:
                    vecs = normalize(vecs)

                baseline_ram = get_baseline_ram(CONTAINER_NAME)

                #Limpar coleção anterior com delay
                if client.collection_exists("sift_bench"):
                    client.delete_collection("sift_bench")
                    time.sleep(2)  # Garante que o Qdrant liberta os ficheiros do disco

                client.create_collection(
                    collection_name="sift_bench",
                    vectors_config=VectorParams(size=128, distance=cfg["distance"]),
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200)
                )

                monitor = DockerMonitor(CONTAINER_NAME)
                monitor.start()

                # --- Ingestão e Indexação ---
                t0 = time.perf_counter()
                client.upload_collection(
                    collection_name="sift_bench",
                    vectors=vecs,
                    ids=list(range(len(vecs))),
                    batch_size=1000
                )
                wait_for_index(client, "sift_bench")
                idx_time = time.perf_counter() - t0

                #B1: Warmup para aquecer o mmap e eliminar page faults
                print(f"A aquecer o mmap com {WARMUP_QUERIES} queries...")
                for q in query_vecs[:WARMUP_QUERIES]:
                    client.search(
                        collection_name="sift_bench",
                        query_vector=q,
                        limit=10,
                        search_params=SearchParams(hnsw_ef=64)
                    )

                # --- Queries de medição ---
                latencias = []
                results   = []
                for q in query_vecs:
                    t1  = time.perf_counter()
                    res = client.search(
                        collection_name="sift_bench",
                        query_vector=q,
                        limit=10,
                        search_params=SearchParams(hnsw_ef=64)
                    )
                    latencias.append(time.perf_counter() - t1)
                    results.append([int(hit.id) for hit in res])

                monitor.stop()
                monitor.join()

                # --- Métricas ---
                avg_abs, peak_abs = monitor.get_results()
                avg_ram  = max(0.0, avg_abs  - baseline_ram)
                peak_ram = max(0.0, peak_abs - baseline_ram)

                lat_ms = np.array(latencias) * 1000
                p50    = np.percentile(lat_ms, 50)
                p95    = np.percentile(lat_ms, 95)
                p99    = np.percentile(lat_ms, 99)
                qps    = len(query_vecs) / np.sum(latencias)
                recall = calculate_recall(results, gt_vecs)

                writer.writerow([
                    r, n, f"{idx_time:.4f}",
                    f"{avg_ram:.4f}", f"{peak_ram:.4f}",
                    f"{p50:.4f}", f"{p95:.4f}", f"{p99:.4f}",
                    f"{qps:.2f}", f"{recall:.4f}"
                ])
                print(f"Escala {n}: Index={idx_time:.2f}s | "
                      f"p50={p50:.2f}ms | QPS={qps:.0f} | "
                      f"RAM_Peak={peak_ram:.2f}GB | "
                      f"Recall@10={recall:.4f}")

    print("\n Benchmark realizado com sucesso.")


# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Qdrant")
    parser.add_argument(
        "--metric", choices=["l2", "cosine"], required=True,
        help="Métrica de distância: 'l2' (Euclidiana) ou 'cosine'"
    )
    args = parser.parse_args()
    run_bench(args.metric)
