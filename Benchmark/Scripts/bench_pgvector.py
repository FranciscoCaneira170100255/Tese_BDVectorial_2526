"""
BENCHMARK: pgvector | Métrica configurável: Euclidiana (L2) ou Cosseno
Versão corrigida (revisão da tese) — A1, A4, B2 resolvidos
"""
import argparse
import numpy as np
import time
import csv
import os
import docker
import threading
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

# --- Configuração -------------------------------------------------------------
CONTAINER_NAME = "pgvector"
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SCALES         = [100, 1_000, 10_000, 100_000, 1_000_000]
REPLICATES     = 10
QUERY_FILE     = os.path.join(BASE_DIR, "sift/sift_query.npy")
WARMUP_QUERIES = 50
INSERT_PAGE_SIZE = 1000

METRIC_CONFIG = {
    "l2": {                                                    # >>> sem espaços
        "hnsw_ops":   "vector_l2_ops",
        "query_op":   " <-> ",
        "gt_dir":     os.path.join(BASE_DIR, "sift/gt"),
        "normalize":  False,
        "results_file": os.path.join(BASE_DIR, "results_pgvector_l2.csv"),
    },
    "cosine": {
        "hnsw_ops":   "vector_cosine_ops",
        "query_op":   " <=> ",
        "gt_dir":     os.path.join(BASE_DIR, "sift/gt_cosine"),
        "normalize":  True,
        "results_file": os.path.join(BASE_DIR, "results_pgvector_cosine.csv"),
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


# --- Utils --------------------------------------------------------------------
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def calculate_recall(results, ground_truth, k=10) -> float:
    hits = sum(
        len(set(results[i]) & set(ground_truth[i][:k]))
        for i in range(len(results))
    )
    return hits / (len(results) * k)


# --- Benchmark ----------------------------------------------------------------
def run_bench(metric: str):
    cfg = METRIC_CONFIG[metric]
    print(f"\n[INICIO] A iniciar benchmark para métrica: {metric.upper()}")

    conn = psycopg2.connect(
        host="127.0.0.1", port=5432,
        dbname="vectordb", user="postgres", password="postgres"
    )
    cur = conn.cursor()
    register_vector(conn)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    print("[Setup] A limpar conexões bloqueadas de execuções anteriores...")
    cur.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = 'vectordb'
          AND pid <> pg_backend_pid();
    """)
    conn.commit()

    # >>> ALTERAÇÃO B2/A1: Forçar uso do índice HNSW em TODAS as queries
    # Isto resolve o problema A1 (sequential scan em Cosseno a escalas grandes)
    cur.execute("SET enable_seqscan = off;")
    conn.commit()

    query_vecs = np.load(QUERY_FILE).astype(np.float32)
    if cfg["normalize"]:
        query_vecs = normalize(query_vecs)

    with open(cfg["results_file"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Replicate", "Scale", "Ingest_Time_s", "Index_Build_s",
            "RAM_Avg_GB", "RAM_Peak_GB",
            "p50_ms", "p95_ms", "p99_ms",
            "QPS", "Recall@10"
        ])

        for r in range(1, REPLICATES + 1):
            print(f"\n=== REPLICA {r} ===")
            for n in SCALES:
                print(f"> PGVECTOR [{metric.upper()}] | Escala: {n}")

                vecs = np.load(os.path.join(BASE_DIR, f"sift/subsets/sift_base_{n}.npy")).astype(np.float32)
                gt_vecs = np.load(os.path.join(cfg["gt_dir"], f"gt_{n}.npy"))
                if cfg["normalize"]:
                    vecs = normalize(vecs)

                baseline_ram = get_baseline_ram(CONTAINER_NAME)

                cur.execute("DROP TABLE IF EXISTS sift_bench;")
                cur.execute("CREATE TABLE sift_bench (id SERIAL PRIMARY KEY, embedding vector(128));")
                conn.commit()

                monitor = DockerMonitor(CONTAINER_NAME)
                monitor.start()

                # --- Ingestão ---
                t0 = time.perf_counter()
                execute_values(
                    cur,
                    "INSERT INTO sift_bench (embedding) VALUES %s",
                    [(v.tolist(),) for v in vecs],
                    page_size=INSERT_PAGE_SIZE
                )
                conn.commit()
                ingest_time = time.perf_counter() - t0

                # --- Construção do índice HNSW ---
                cur.execute("SET maintenance_work_mem = '1GB';")
                t1 = time.perf_counter()
                cur.execute(f"""
                    CREATE INDEX ON sift_bench
                    USING hnsw (embedding {cfg['hnsw_ops']})
                    WITH (m = 16, ef_construction = 200);
                """)
                conn.commit()
                idx_time = time.perf_counter() - t1

                # >>> ALTERAÇÃO A4: Coluna Idx = APENAS Index_Build_s
                # A legenda da Tabela 4.1/4.2 diz "excluindo ingestão",
                # e o CSV já escreve as duas colunas separadas.
                # Garantimos que a tese usa Index_Build_s.

                cur.execute("ANALYZE sift_bench;")
                conn.commit()

                cur.execute("SET hnsw.ef_search = 64;")
               
                cur.execute("SET enable_seqscan = off;")
                conn.commit()

                # --- Warmup ---
                for q in query_vecs[:WARMUP_QUERIES]:
                    cur.execute(
                        f"SELECT id FROM sift_bench ORDER BY embedding {cfg['query_op']} %s::vector LIMIT 10;",
                        (q.tolist(),)
                    )
                    cur.fetchall()

                # --- Queries de medição ---
                latencias = []
                results = []
                for q in query_vecs:
                    t2 = time.perf_counter()
                    cur.execute(
                        f"SELECT id FROM sift_bench ORDER BY embedding {cfg['query_op']} %s::vector LIMIT 10;",
                        (q.tolist(),)
                    )
                    res = cur.fetchall()
                    latencias.append(time.perf_counter() - t2)
                    results.append([row[0] for row in res])

                monitor.stop()
                monitor.join()

                # Ajuste de IDs (PG começa em 1, GT em 0)
                adjusted = [[x - 1 for x in rlist] for rlist in results]

                avg_abs, peak_abs = monitor.get_results()
                avg_ram  = max(0.0, avg_abs  - baseline_ram)
                peak_ram = max(0.0, peak_abs - baseline_ram)

                lat_ms = np.array(latencias) * 1000
                p50    = np.percentile(lat_ms, 50)
                p95    = np.percentile(lat_ms, 95)
                p99    = np.percentile(lat_ms, 99)
                qps    = len(query_vecs) / np.sum(latencias)
                recall = calculate_recall(adjusted, gt_vecs)

                writer.writerow([
                    r, n, f"{ingest_time:.4f}", f"{idx_time:.4f}",
                    f"{avg_ram:.4f}", f"{peak_ram:.4f}",
                    f"{p50:.4f}", f"{p95:.4f}", f"{p99:.4f}",
                    f"{qps:.2f}", f"{recall:.4f}"
                ])
                print(f"  Escala {n}: Ingest={ingest_time:.2f}s | Index={idx_time:.2f}s | p50={p50:.2f}ms | QPS={qps:.0f} | RAM_Peak={peak_ram:.2f}GB")

    cur.close()
    conn.close()
    print("\nBenchmark realizado com sucesso.")


# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pgvector")
    parser.add_argument(
        "--metric", choices=["l2", "cosine"], required=True,
        help="Métrica de distância: 'l2' (Euclidiana) ou 'cosine'"
    )
    args = parser.parse_args()
    run_bench(args.metric)
