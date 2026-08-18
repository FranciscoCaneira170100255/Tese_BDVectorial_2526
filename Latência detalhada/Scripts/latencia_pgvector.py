"""
Benchmark de latência detalhada para pgvector.
"""
import argparse
import os
import numpy as np
import psycopg2
import time
import csv
from scipy.stats import skew, kurtosis

#Args
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

QUERIES_FILE = "sift/sift_query.npy"
NUM_QUERIES  = 10000
TABLE_NAME   = "items"
EF_SEARCH    = 64     
CSV_FILE     = f"latencias_pgvector_{args.metric}.csv"


if args.metric == "l2":
    QUERY_OP = "<->"
else:
    QUERY_OP = "<=>"

#Load Queries
print(f"Carregando {NUM_QUERIES} queries...")
queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)

# Normalização para Cosseno - pré-requisito matemático para equivalência L2/Cosine
if args.metric == "cosine":
    print("Normalizando vetores (Cosine)...")
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1
    queries = queries / norms

queries = queries.tolist()

#Database connection
conn = psycopg2.connect(
    host="localhost", port=5432, user="postgres", password="postgres", dbname="vectordb"
)
conn.autocommit = True  # Necessário para o conjunto de comandos SET sem iniciar blocos de transação explícitos
cur = conn.cursor()

#Verifica se o índice existe
print(f"Verificando se o indice HNSW existe na tabela '{TABLE_NAME}'...")
cur.execute(f"""
    SELECT indexname FROM pg_indexes
    WHERE tablename = '{TABLE_NAME}'
    AND indexdef LIKE '%hnsw%';
""")
existing_indexes = cur.fetchall()

if not existing_indexes:
    print("Nenhum indice HNSW encontrado! A criar indice...")
    cur.execute("SET maintenance_work_mem = '2GB';")
    # A classe de operadores TEM de bater certo com a métrica - senão o planner ignora o índice
    hnsw_ops = "vector_l2_ops" if args.metric == "l2" else "vector_cosine_ops"
    cur.execute(f"""
        CREATE INDEX items_hnsw_idx
        ON {TABLE_NAME}
        USING hnsw (embedding {hnsw_ops})
        WITH (m = 16, ef_construction = 200);
    """)
    cur.execute(f"ANALYZE {TABLE_NAME};")
    print("Indice criado com sucesso.")
else:
    print(f"Indice HNSW ja existe: {existing_indexes[0][0]}")

#Config e Warmup
print(f"Configurando ef_search = {EF_SEARCH}...")
cur.execute(f"SET hnsw.ef_search = {EF_SEARCH};")

# Força o uso do índice. Sem isto, o planner pode fazer seqscan se achar que é mais rápido
# para tabelas pequenas. Isto invalidaria completamente o benchmark 
# porque pode achar que é mais eficaz fazer um Sequencial Scan em vez de usar HNSW
cur.execute("SET enable_seqscan = off;")

# Warmup do buffer pool do PostgreSQL. As primeiras queries carregam páginas do disco para a RAM.
# Sem isto, o p50 fica enviesado com os cold misses iniciais
# O warmup percorre a mesma carga de trabalho completa que vai ser medida a seguir
# (as 10000 queries), não só uma amostra parcial, caso contrário, o buffer pool só fica
# aquecido para a região do grafo HNSW mexida pela amostra e o resto das queries sofrem
# cache misses reais já dentro do ciclo cronometrado (confirmado empiricamente pelos outliers
# concentrados nos primeiros ~200 IDs a seguir à fronteira de uma amostra de warmup pequena)
print(f"A executar warmup ({NUM_QUERIES} queries para aquecer buffer pool com a carga completa)...")
for q in queries:
    cur.execute(f"SELECT id FROM {TABLE_NAME} ORDER BY embedding {QUERY_OP} %s::vector LIMIT 10;", (q,))
    cur.fetchall()

#Run Benchmark
print("Executando queries sequenciais (medindo latência pura)...")
latencies = []
for i, q in enumerate(queries):
    t0 = time.perf_counter()
    cur.execute(f"SELECT id FROM {TABLE_NAME} ORDER BY embedding {QUERY_OP} %s::vector LIMIT 10;", (q,))
    cur.fetchall()
    lat = (time.perf_counter() - t0) * 1000  # ms
    latencies.append(lat)
    
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1} / {NUM_QUERIES} processadas...")

cur.close()
conn.close()

#Stats calculation
lat_np = np.array(latencies)
p = np.percentile(lat_np, [10, 25, 50, 75, 90, 95, 99, 99.9])
sk = skew(lat_np)
ku = kurtosis(lat_np)

#CSV export
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Query_ID", "Latency_ms"])
    for i, lat in enumerate(latencies):
        writer.writerow([i + 1, round(lat, 4)])

RESUMO_FILE = "latencias_pgvector_resumo.csv"
resumo_existe = os.path.isfile(RESUMO_FILE)
with open(RESUMO_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not resumo_existe:
        writer.writerow([
            "Motor", "Metrica", "p10", "p25", "p50", "p75", "p90",
            "p95", "p99", "p99_9", "Skewness", "Kurtosis"
        ])
    writer.writerow([
        "pgvector", args.metric,
        f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{p[3]:.4f}",
        f"{p[4]:.4f}", f"{p[5]:.4f}", f"{p[6]:.4f}", f"{p[7]:.4f}",
        f"{sk:.4f}", f"{ku:.4f}"
    ])
