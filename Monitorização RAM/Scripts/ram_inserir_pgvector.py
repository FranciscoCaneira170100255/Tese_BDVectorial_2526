import argparse
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

DATA_FILE = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS = 1000000
BATCH_SIZE = 5000

if args.metric == "l2":
    TABLE_NAME = "disk_test_l2"
    INDEX_OPS = "vector_l2_ops"
    NORMALIZE = False
else:
    TABLE_NAME = "disk_test_cosine"
    INDEX_OPS = "vector_cosine_ops"
    NORMALIZE = True


conn = psycopg2.connect(dbname="postgres", user="postgres", password="postgres", host="localhost", port=5432)
conn.autocommit = True
cur = conn.cursor()

print(f"A recriar tabela '{TABLE_NAME}' no pgvector...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
cur.execute(f"CREATE TABLE {TABLE_NAME} (id bigserial PRIMARY KEY, embedding vector(128));")
print("A carregar vetores...")
vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)

if NORMALIZE:
    print("A normalizar vetores (Cosseno)...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

print("A iniciar inserção em lotes...")
for i in range(0, TOTAL_VECTORS, BATCH_SIZE):
    batch = vectors[i : i + BATCH_SIZE].tolist()
    # Converte para string de array que o pgvector entende
    values = [(f"[{','.join(map(str, v))}]",) for v in batch]
    execute_values(cur, f"INSERT INTO {TABLE_NAME} (embedding) VALUES %s", values)
    if (i + BATCH_SIZE) % 100000 == 0:
        print(f"  {i + BATCH_SIZE} vetores inseridos...")

print("A construir índice HNSW...")
cur.execute(f"""
    CREATE INDEX ON {TABLE_NAME} USING hnsw (embedding {INDEX_OPS}) 
    WITH (m = 16, ef_construction = 200);
""")

print("\nConcluído! Índice construído no PostgreSQL.")
cur.close()
conn.close()
