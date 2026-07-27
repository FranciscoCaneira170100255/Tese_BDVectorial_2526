"""
Este script avalia a escalabilidade concorrente do pgvector sobre
PostgreSQL utilizando o dataset SIFT.
"""
import argparse
import numpy as np
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from psycopg2.pool import ThreadedConnectionPool

#Args
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

#Config 
QUERIES_FILE = "sift/sift_query.npy"
TABLE_NAME   = "items"  
NUM_QUERIES  = 1000
THREADS_LIST = [1, 2, 4, 8, 16, 24, 32]
EF_SEARCH    = 64
CSV_FILE     = f"resultados_concorrencia_pgvector_{args.metric}.csv"

QUERY_OP = "<=>" if args.metric == "cosine" else "<->"
HNSW_OPS = "vector_cosine_ops" if args.metric == "cosine" else "vector_l2_ops"

print("A carregar queries...")
queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)

#Normaliza para o cosseno
if args.metric == "cosine":
    norms   = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1
    queries = queries / norms
queries = queries.tolist()

# Este pool usa ligações para que não termine o Postgres com o overhead de abrir e fechar conexões novas por thread
db_pool = ThreadedConnectionPool(
    minconn=1, maxconn=35,
    dbname="vectordb", user="postgres", password="postgres", host="localhost"
)

# Preparar o índice
print(f"A preparar o indice HNSW para a metrica {args.metric.upper()}...")
conn_setup = db_pool.getconn()
cur_setup = conn_setup.cursor()
try:
    # 1. Para garantir que usamos a classe de operadores corretos para a métrica que vamos escolher
    cur_setup.execute("DROP INDEX IF EXISTS items_hnsw_idx;")
    
    # 2. O ANALYZE é importante, porque sem ele o Query Planner do Postgres ignora o índice e faz um Scan Sequencial
    cur_setup.execute(f"""
        CREATE INDEX items_hnsw_idx
        ON {TABLE_NAME}
        USING hnsw (embedding {HNSW_OPS})
        WITH (m = 16, ef_construction = 200);
    """)
    conn_setup.commit()
    
   
    cur_setup.execute(f"ANALYZE {TABLE_NAME};")
    conn_setup.commit()
    print(f"Indice recriado com sucesso para a metrica {args.metric.upper()}")
except Exception as e:
    print(f"ERRO ao preparar indice: {e}")
finally:
    cur_setup.close()
    db_pool.putconn(conn_setup)


def worker_pgvector(query_vector):
    conn = db_pool.getconn()
    cur  = conn.cursor()
    cur.execute(f"SET hnsw.ef_search = {EF_SEARCH};")
    t0 = time.perf_counter()
    cur.execute(
        f"SELECT id FROM {TABLE_NAME} ORDER BY embedding {QUERY_OP} %s::vector LIMIT 10;",
        (query_vector,)
    )
    cur.fetchall()
    latency = time.perf_counter() - t0
    cur.close()
    db_pool.putconn(conn)
    return latency

# BENCHMARK
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Threads", "QPS", "p95_Latency_ms"])

    for num_threads in THREADS_LIST:
        print(f"\n[pgvector {args.metric.upper()}] A testar com {num_threads} Threads...")
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(worker_pgvector, queries)
            latencies = list(results)

        total_time = time.perf_counter() - start_time

        qps    = NUM_QUERIES / total_time
        lat_ms = np.array(latencies) * 1000
        p95    = np.percentile(lat_ms, 95)

        print(f"  Resultado: {qps:.0f} QPS | Latencia p95: {p95:.2f} ms")

      
        writer.writerow([num_threads, round(qps, 2), round(p95, 2)])
        f.flush() # Obriga à escrita imediata no disco para não haver perda de dados se o script for interrompido

db_pool.closeall()
print(f"\nFeito! Guardado em {CSV_FILE}")