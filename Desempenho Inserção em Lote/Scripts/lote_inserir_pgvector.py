"""
Inserção em lote: pgvector (Bulk Loading) | Métricas configuráveis: L2 ou Cosine
"""
import argparse
import numpy as np
import time
import csv
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import gc

# --- Configuração -------------------------------------------------------------
DATA_FILE       = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS   = 100_000
BATCH_SIZES     = [100, 1_000, 10_000, 50_000]
M               = 16
EF_CONSTRUCTION = 200

METRIC_CONFIG = {
    "l2": {                                                     
        "hnsw_ops":   "vector_l2_ops",
        "normalize":  False,
        
        "csv_file":   "resultados_batch_pgvector_L2.csv",
        "label":      "L2",
    },
    "cosine": {
        "hnsw_ops":   "vector_cosine_ops",
        "normalize":  True,

        "csv_file":   "resultados_batch_pgvector_Cosine.csv",
        "label":      "Cosine",
    },
}

# --- Utils --------------------------------------------------------------------
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

# --- Benchmark ----------------------------------------------------------------
def run_batch_insertion(metric: str):
    cfg = METRIC_CONFIG[metric]

    print("A carregar dados...")
    vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)
    if cfg["normalize"]:
        vectors = normalize(vectors)

    conn = psycopg2.connect(
        dbname="vectordb", user="postgres",
        password="postgres", host="localhost"
    )
    conn.autocommit = True
    cur = conn.cursor()
    register_vector(conn)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Importante: permite dar mais memória ao PostgreSQL, para que a criação
    # do índice não tenha bottleneck
    cur.execute("SET maintenance_work_mem = '2GB';")

    with open(cfg["csv_file"], "w", newline="") as f:
        csv.writer(f).writerow(["DB", "Batch_Size", "Time_s", "Throughput_vec_s"])

    for b_size in BATCH_SIZES:
        print(f"\n[pgvector OTIMIZADO {cfg['label']}] Lote: {b_size}")

        cur.execute("DROP TABLE IF EXISTS items;")
        cur.execute(
            "CREATE TABLE items (id SERIAL PRIMARY KEY, embedding vector(128));"
        )

        t0 = time.perf_counter()

        # Utilização de Bulk Loading: inserção seguida de criação do índice
        print(f"A inserir {TOTAL_VECTORS} vetores...")
        execute_values(
            cur,
            "INSERT INTO items (embedding) VALUES %s",
            [(v,) for v in vectors],
            page_size=b_size
        )

        print("A construir o índice HNSW...")
        cur.execute(f"""
            CREATE INDEX ON items
            USING hnsw (embedding {cfg['hnsw_ops']})
            WITH (m = {M}, ef_construction = {EF_CONSTRUCTION});
        """)

        time_taken = time.perf_counter() - t0
        throughput = TOTAL_VECTORS / time_taken
        print(f"Tempo Total: {time_taken:.2f}s | Throughput Final: {throughput:.0f} vec/s")

        with open(cfg["csv_file"], "a", newline="") as f:
            csv.writer(f).writerow([
                "pgvector_bulk", cfg["label"], b_size,
                round(time_taken, 2), round(throughput, 0)
            ])

        gc.collect()

    cur.close()
    conn.close()
    print(f"\nFeito! Ficheiro: {cfg['csv_file']}")

# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":                                        
    parser = argparse.ArgumentParser(description="Batch insertion pgvector Optimizada")
    parser.add_argument(
        "--metric", choices=["l2", "cosine"], required=True,
        help="Métrica de distância: 'l2' (Euclidiana) ou 'cosine'"
    )
    args = parser.parse_args()
    run_batch_insertion(args.metric)
