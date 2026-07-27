"""
BENCHMARK: pgvector (L2 e Cosine)
Mede RAM real via docker stats (cgroups v2)
"""
import argparse
import numpy as np
import time
import csv
import psycopg2
import docker
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

# --- Configuração Global ------------------------------------------------------
DATA_FILE        = "sift/subsets/sift_base_1000000.npy"
QUERIES_FILE     = "sift/sift_query.npy"
PG_TABLE         = "items_optimize"
TOTAL_VECTORS    = 1_000_000
NUM_QUERIES      = 1_000
EF_CONSTRUCTION  = 200
WARMUP_QUERIES   = 50
CONTAINER_NAME   = "pgvector"

M_VALUES         = [16, 24, 32]
EF_SEARCH_VALUES = [64, 128, 256]

METRIC_CONFIG = {
    "l2": {
        "hnsw_ops":    "vector_l2_ops",
        "query_op":    "<->",
        "gt_file":     "sift/gt/gt_1000000.npy",
        "normalize":   False,
        "csv_file":    "resultados_otimizacao_pgvector_L2.csv",
        "label":       "L2",
    },
    "cosine": {
        "hnsw_ops":    "vector_cosine_ops",
        "query_op":    "<=>",
        "gt_file":     "sift/gt_cosine/gt_1000000.npy",
        "normalize":   True,
        "csv_file":    "resultados_otimizacao_pgvector_Cosine.csv",
        "label":       "Cosine",
    },
}

# --- Helpers ------------------------------------------------------------------
def normalize(vectors: np.ndarray) -> np.ndarray:
    """Normaliza vetores para norma unitária."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def calc_recall(results, ground_truth, k=10) -> float:
    hits = sum(len(set(res) & set(gt_row[:k])) for res, gt_row in zip(results, ground_truth))
    return (hits / (len(results) * k)) * 100

def get_pgvector_ram_gb_docker(container_name=CONTAINER_NAME, duration_s=5, interval_s=0.5):
    """
    Mede a RAM  do contentor pgvector via docker stats (cgroups v2).
    Retorna o pico de RAM em GB durante o intervalo de amostragem.
    """
    client = docker.from_env()
    container = client.containers.get(container_name)
    
    samples = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        stats = container.stats(stream=False)
        # memory usage em bytes (usage - cache, para medir RAM ativa)
        mem_usage = stats["memory_stats"]["usage"] - stats["memory_stats"].get("stats", {}).get("inactive_file", 0)
        samples.append(mem_usage / (1024**3))
        time.sleep(interval_s)
    
    return max(samples)  # retorna o pico, consistente com os outros motores

# --- Main Loop ----------------------------------------------------------------
def run_optimization(metric: str):
    cfg = METRIC_CONFIG[metric]
    conn = psycopg2.connect(dbname="vectordb", user="postgres", password="postgres", host="localhost")
    conn.autocommit = True
    cur = conn.cursor()
    register_vector(conn)
    
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("SET maintenance_work_mem = '2GB';")
    
    print("A carregar dados para a RAM...")
    base_vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)
    queries      = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)
    gt           = np.load(cfg["gt_file"])[:NUM_QUERIES]
    
    if cfg["normalize"]:
        base_vectors = normalize(base_vectors)
        queries      = normalize(queries)
    
    # Criar tabela
    print(f"\n[FASE 1] A criar tabela '{PG_TABLE}'...")
    cur.execute(f"DROP TABLE IF EXISTS {PG_TABLE};")
    cur.execute(f"CREATE TABLE {PG_TABLE} (id INTEGER PRIMARY KEY, embedding vector(128));")
    
    print(f"A inserir {TOTAL_VECTORS:,} vetores (Bulk insert, pode demorar)...")
    data_to_insert = [(i, v.tolist()) for i, v in enumerate(base_vectors)]
    execute_values(cur, f"INSERT INTO {PG_TABLE} (id, embedding) VALUES %s", data_to_insert, page_size=10_000)
    
    # Criar o header do CSV
    with open(cfg["csv_file"], "w", newline="") as f:
        csv.writer(f).writerow(["DB", "Metric", "m", "ef_search", "Recall_10", "p50_ms", "RAM_GB"])
    
    query_sql = f"SELECT id FROM {PG_TABLE} ORDER BY embedding {cfg['query_op']} %s::vector LIMIT 10;"
    
    for m in M_VALUES:
        print(f"\n=== m={m} | A construir índice HNSW ===")
        cur.execute("DROP INDEX IF EXISTS idx_optimize;")
        cur.execute(f"""
            CREATE INDEX idx_optimize ON {PG_TABLE} 
            USING hnsw (embedding {cfg['hnsw_ops']}) 
            WITH (m = {m}, ef_construction = {EF_CONSTRUCTION});
        """)
        print(f"  Índice pronto.")
        
        cur.execute("SET enable_seqscan = off;")  # Forçar uso do índice
        
        for ef_s in EF_SEARCH_VALUES:
            print(f"  -> ef_search={ef_s}")
            cur.execute(f"SET hnsw.ef_search = {ef_s};")
            
            # Warmup para estabilizar a buffer pool
            for q in queries[:WARMUP_QUERIES]:
                cur.execute(query_sql, (q.tolist(),))
                cur.fetchall()
            
            latencies, results = [], []
            for q in queries:
                t0 = time.perf_counter()
                cur.execute(query_sql, (q.tolist(),))
                results.append([row[0] for row in cur.fetchall()])
                latencies.append((time.perf_counter() - t0) * 1000)
            
            recall = calc_recall(results, gt)
            p50    = np.percentile(latencies, 50)
            
            # Medir RAM via docker stats
            ram_gb = get_pgvector_ram_gb_docker()
            
            print(f"     Recall={recall:.2f}% | p50={p50:.2f}ms | RAM={ram_gb:.2f}GB")
            
            with open(cfg["csv_file"], "a", newline="") as f:
                csv.writer(f).writerow([
                    "pgvector", cfg["label"], m, ef_s, 
                    round(recall, 2), round(p50, 2), round(ram_gb, 2)
                ])
    
    cur.close()
    conn.close()
    print(f"\nFeito! Ficheiro: {cfg['csv_file']}")

# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otimização pgvector")
    parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
    args = parser.parse_args()
    run_optimization(args.metric)
