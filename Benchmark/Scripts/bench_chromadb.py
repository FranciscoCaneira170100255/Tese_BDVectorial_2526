"""
BENCHMARK: ChromaDB | Métrica configurável: Euclidiana (L2) ou Cosseno
Uso:
    python bench_chromadb.py --metric l2
    python bench_chromadb.py --metric cosine
"""

import argparse
import numpy as np
import time
import csv
import gc
import docker
import threading
import chromadb

#Config

CONTAINER_NAME = "chromadb"
SCALES         = [100, 1_000, 10_000, 100_000, 1_000_000]
REPLICATES     = 10
QUERY_FILE     = "sift/sift_query.npy"
BATCH_SIZE     = 5000
WARMUP_QUERIES = 50

METRIC_CONFIG = {
    "l2": {
        "hnsw_space": "l2",
        "gt_dir":     "sift/gt",
        "normalize":  False,
        "results_file": "results_chromadb_l2.csv",
    },
    "cosine": {
        "hnsw_space": "cosine",
        "gt_dir":     "sift/gt_cosine",
        "normalize":  True,
        "results_file": "results_chromadb_cosine.csv",
    },
}


#Docker Monitor

#Corremos a leitura da RAM numa thread separada, de forma a não atrasar o loop principal do benchmark
class DockerMonitor(threading.Thread):
    def __init__(self, container_name):
        super().__init__()
        self.client    = docker.from_env()
        self.container = self.client.containers.get(container_name)
        self.memory_samples = []
        self.stop_flag = False

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


#Utils

def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def calculate_recall(results, ground_truth, k=10) -> float:
    hits = sum(
        len(set(results[i]) & set(ground_truth[i][:k]))
        for i in range(len(results))
    )
    return hits / (len(results) * k)


#Benchmark

def run_bench(metric: str):
    cfg    = METRIC_CONFIG[metric]
    client = chromadb.HttpClient(host="localhost", port=8000)

    query_vecs = np.load(QUERY_FILE).astype(np.float32)
    if cfg["normalize"]:
        query_vecs = normalize(query_vecs)

    with open(cfg["results_file"], mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Replicate", "Scale", "Index_Time_s",
            "RAM_Avg_GB", "RAM_Peak_GB",
            "p50_ms", "p95_ms", "p99_ms",
            "QPS", "Recall@10"
        ])

        for r in range(1, REPLICATES + 1):
            print(f"\n========== Replica {r}/{REPLICATES} ==========")

            for n in SCALES:
                print(f">>> CHROMADB [{metric.upper()}] | Escala {n}")

                vecs    = np.load(f"sift/subsets/sift_base_{n}.npy").astype(np.float32)
                gt_vecs = np.load(f"{cfg['gt_dir']}/gt_{n}.npy")

                if cfg["normalize"]:
                    vecs = normalize(vecs)

                gc.collect()
                baseline_ram = get_baseline_ram(CONTAINER_NAME)

                # Garante que o run começa do zero
                try:
                    client.delete_collection("sift_bench")
                    time.sleep(1)
                except Exception:
                    pass

                collection = client.create_collection(
                    name="sift_bench",
                    metadata={
                        "hnsw:space":           cfg["hnsw_space"],
                        "hnsw:M":               16,
                        "hnsw:construction_ef": 200,
                        "hnsw:search_ef":       64,
                    }
                )

                monitor = DockerMonitor(CONTAINER_NAME)
                monitor.start()

                #Indexação
                t0 = time.perf_counter()
                for i in range(0, len(vecs), BATCH_SIZE):
                    batch = vecs[i:i + BATCH_SIZE]
                    ids   = [str(x) for x in range(i, i + len(batch))]
                    collection.add(ids=ids, embeddings=batch.tolist())
                idx_time = time.perf_counter() - t0

                #Warmup
                for q in query_vecs[:WARMUP_QUERIES]:
                    collection.query(query_embeddings=[q.tolist()], n_results=10)

                #Queries Sequenciais
                latencias      = []
                search_results = []

                for q in query_vecs:
                    t1  = time.perf_counter()
                    res = collection.query(query_embeddings=[q.tolist()], n_results=10)
                    latencias.append(time.perf_counter() - t1)
                    search_results.append([int(x) for x in res["ids"][0]])

                monitor.stop()
                monitor.join()

                #Métricas
                avg_abs, peak_abs = monitor.get_results()
                avg_ram  = max(0.0, avg_abs  - baseline_ram)
                peak_ram = max(0.0, peak_abs - baseline_ram)

                lat_ms = np.array(latencias) * 1000
                p50    = np.percentile(lat_ms, 50)
                p95    = np.percentile(lat_ms, 95)
                p99    = np.percentile(lat_ms, 99)
                qps    = len(query_vecs) / np.sum(latencias)
                recall = calculate_recall(search_results, gt_vecs)

                writer.writerow([
                    r, n, f"{idx_time:.4f}",
                    f"{avg_ram:.4f}", f"{peak_ram:.4f}",
                    f"{p50:.4f}", f"{p95:.4f}", f"{p99:.4f}",
                    f"{qps:.2f}", f"{recall:.4f}"
                ])

                print(f"  Escala {n}: Index={idx_time:.2f}s | "
                      f"p95={p95:.2f}ms | QPS={qps:.0f} | "
                      f"RAM_Peak={peak_ram:.2f}GB | Recall@10={recall:.4f}")


#Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ChromaDB")
    parser.add_argument(
        "--metric", choices=["l2", "cosine"], required=True,
        help="Métrica de distância: 'l2' (Euclidiana) ou 'cosine'"
    )
    args = parser.parse_args()
    run_bench(args.metric)
