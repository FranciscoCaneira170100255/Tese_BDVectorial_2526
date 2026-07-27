"""
Inserção em lote: Qdrant | Métricas configuráveis: L2 ou Cosine
Exemplo de como podemos utilizar:
python lote_inserir_qdrant.py --metric l2
python lote_inserir_qdrant.py --metric cosine
"""
import argparse
import numpy as np
import time
import csv
import gc
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

# --- Configuração -------------------------------------------------------------
DATA_FILE         = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS     = 100_000
BATCH_SIZES       = [100, 1_000, 10_000, 50_000]
COLLECTION_NAME   = "batch_test"
M                 = 16
EF_CONSTRUCTION   = 200

# Limite máximo por pedido HTTP ao Qdrant. Lotes maiores que este valor
# são divididos internamente em pedaços de 10k, mas o tempo total é medido na totalidade
# o throughput reflecte o tamanho lógico testado, não o sub-lote HTTP.
MAX_QDRANT_BATCH  = 10_000

METRIC_CONFIG = {
    "l2": {                                                     
        "distance":  Distance.EUCLID,
        "normalize": False,
        "csv_file":  "resultados_batch_qdrant_L2.csv",
        "label":     "L2",
    },
    "cosine": {
        "distance":  Distance.COSINE,
        "normalize": True,
        "csv_file":  "resultados_batch_qdrant_Cosine.csv",
        "label":     "Cosine",
    },
}

# --- Utils --------------------------------------------------------------------
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

# --- Benchmark ----------------------------------------------------------------
def run_batch_test(metric: str):
    cfg    = METRIC_CONFIG[metric]
    client = QdrantClient(host="localhost", port=6333, timeout=300)

    print(f"A carregar {TOTAL_VECTORS} vetores para a RAM...")
    vecs = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)
    if cfg["normalize"]:
        vecs = normalize(vecs)
    ids = list(range(TOTAL_VECTORS))

    with open(cfg["csv_file"], "w", newline="") as f:
        csv.writer(f).writerow(["DB", "Metric", "Batch_Size",
                                "Time_s", "Throughput_vec_s"])

    for b_size in BATCH_SIZES:
        print(f"\n[QDRANT {cfg['label']}] Lote: {b_size}")

        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=cfg["distance"]),
            hnsw_config=HnswConfigDiff(m=M, ef_construct=EF_CONSTRUCTION)
        )

        t0 = time.perf_counter()

        # Qdrant rejeita payloads HTTP > ~10K vetores, para não ultrapassar o limite.
        # São divididos em pedaços de MAX_QDRANT_BATCH mas medimos o tempo total
        # o throughput reflecte o tamanho lógico de b_size.
        http_batch = min(b_size, MAX_QDRANT_BATCH)
        client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=vecs,
            ids=ids,
            batch_size=http_batch,
            parallel=1
        )

        time_taken = time.perf_counter() - t0
        throughput = TOTAL_VECTORS / time_taken
        print(f"  Tempo: {time_taken:.2f}s | Throughput: {throughput:.0f} vec/s")

        with open(cfg["csv_file"], "a", newline="") as f:
            csv.writer(f).writerow([
                "Qdrant", cfg["label"], b_size,
                round(time_taken, 2), round(throughput, 0)
            ])

        gc.collect()

    print(f"\nFeito! Ficheiro: {cfg['csv_file']}")

# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":                                        
    parser = argparse.ArgumentParser(description="Batch insertion Qdrant")
    parser.add_argument(
        "--metric", choices=["l2", "cosine"], required=True,
        help="Métrica de distância: 'l2' (Euclidiana) ou 'cosine'"
    )
    args = parser.parse_args()
    run_batch_test(args.metric)
