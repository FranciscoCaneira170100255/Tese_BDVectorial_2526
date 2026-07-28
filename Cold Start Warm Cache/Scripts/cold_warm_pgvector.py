"""
Este script executa testes de Cold Start e Warm Start ao pgvector sobre
PostgreSQL utilizando o dataset SIFT.
"""
import argparse
import numpy as np
import time
import csv
import docker
import os
import gc
import psycopg2

#Config
QUERIES_FILE = "sift/sift_query.npy"
NUM_QUERIES  = 1000
REPLICATES   = 10
PG_TABLE     = "items"

METRIC_CONFIG = {
    "l2": {
        "label":      "L2",
        "gt_file":    "sift/gt/gt_1000000.npy",
        "normalize":  False,
        "query_op":   "<->",   
        "hnsw_ops":   "vector_l2_ops",
        "csv_file":   "resultados_cold_warm_pgvector_L2.csv",
    },
    "cosine": {
        "label":      "Cosine",
        "gt_file":    "sift/gt_cosine/gt_1000000.npy",
        "normalize":  True,
        "query_op":   "<=>",   
        "hnsw_ops":   "vector_cosine_ops",
        "csv_file":   "resultados_cold_warm_pgvector_Cosine.csv",
    },
}

HEADERS = [
    "Replicate", "DB", "Metric",
    "First_Resp_Cold_ms", "First_Resp_Warm_ms",
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

def wait_for_pg() -> float:
    start = time.time()
    while True:
        try:
            conn = psycopg2.connect(
                dbname="vectordb", user="postgres",
                password="postgres", host="localhost"
            )
            conn.close()
            return time.time() - start
        except Exception:
            time.sleep(0.1)

#Benchmark
def run_cold_warm(metric: str):
    cfg = METRIC_CONFIG[metric]
    queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)
    if cfg["normalize"]:
        queries = normalize(queries)
    gt = np.load(cfg["gt_file"])[:NUM_QUERIES]

    query_sql = (f"SELECT id FROM {PG_TABLE} "
                 f"ORDER BY embedding {cfg['query_op']} %s::vector LIMIT 10;")
    docker_cli = docker.from_env()

    #Garantir que o indice HNSW existe e esteja otimizado
    print(f"[SETUP] A verificar indice HNSW ({cfg['hnsw_ops']}) na tabela '{PG_TABLE}'...")
    conn_setup = psycopg2.connect(
        dbname="vectordb", user="postgres",
        password="postgres", host="localhost"
    )
    cur_setup = conn_setup.cursor()
    try:
        # Apagar indice antigo (se existir) para evitar conflito de opclass
        cur_setup.execute("DROP INDEX IF EXISTS items_hnsw_idx;")
        cur_setup.execute(f"""
            CREATE INDEX items_hnsw_idx
            ON {PG_TABLE}
            USING hnsw (embedding {cfg['hnsw_ops']})
            WITH (m = 16, ef_construction = 200);
        """)
        conn_setup.commit()
        # Importante para o Query Planner assumir o índice
        cur_setup.execute(f"ANALYZE {PG_TABLE};")
        conn_setup.commit()
        print("Indice recriado e estatisticas atualizadas com sucesso.")
    except Exception as e:
        print(f"Erro ao preparar indice: {e}")
    finally:
        cur_setup.close()
        conn_setup.close()

    with open(cfg["csv_file"], "w", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writeheader()

    for r in range(1, REPLICATES + 1):
        print(f"\n[PGVECTOR {cfg['label']}] Replica {r}/{REPLICATES}")

        # Limpar page cache do SO para simular cold start real
        os.system("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
        time.sleep(1)

        # Reiniciar container para simular arranque a frio real
        container = docker_cli.containers.get("pgvector")
        container.restart()
        load_time = wait_for_pg()

        conn = psycopg2.connect(
            dbname="vectordb", user="postgres",
            password="postgres", host="localhost"
        )
        cur = conn.cursor()

    
        cur.execute("SET hnsw.ef_search = 64;")

        # Warmup minimo (10 queries) para estabilizar a ligacao
        # antes de medir a primeira resposta cold 
        # evita assim que o cold start inclua o overhead do handshake da rede
        for q in queries[:10]:
            cur.execute(query_sql, (q.tolist(),))
            cur.fetchall()

        #Fase Cold 
        cold_latencies, cold_results = [], []
        t_start_cold = time.perf_counter()
        for q in queries:
            t0 = time.perf_counter()
            cur.execute(query_sql, (q.tolist(),))
            cold_results.append([row[0] - 1 for row in cur.fetchall()])
            cold_latencies.append(time.perf_counter() - t0)
        cold_total_time = time.perf_counter() - t_start_cold

        # Fase Warm
        #Nesta fase o índice já tem de estar quase alojado na shared_buffers
        warm_latencies, warm_results = [], []
        for q in queries:
            t0 = time.perf_counter()
            cur.execute(query_sql, (q.tolist(),))
            warm_results.append([row[0] - 1 for row in cur.fetchall()])
            warm_latencies.append(time.perf_counter() - t0)

        cur.close()
        conn.close()

        cold_ms = np.array(cold_latencies) * 1000
        warm_ms = np.array(warm_latencies) * 1000

        row = {
            "Replicate":            r,
            "DB":                   "pgvector",
            "Metric":               cfg["label"],
            "First_Resp_Cold_ms":   round(cold_ms[0], 4),
            "First_Resp_Warm_ms":   round(warm_ms[0], 4),
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

        print(f"  Cold p50={row['p50_Cold_ms']:.2f}ms | Warm p50={row['p50_Warm_ms']:.2f}ms | "
              f"Recall Cold={row['Recall_Cold']:.1f}%")

        del cold_latencies, cold_results, warm_latencies, warm_results, cold_ms, warm_ms
        gc.collect()

    print(f"\nFeito! Ficheiro: {cfg['csv_file']}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold/Warm pgvector")
    parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
    args = parser.parse_args()
    run_cold_warm(args.metric)