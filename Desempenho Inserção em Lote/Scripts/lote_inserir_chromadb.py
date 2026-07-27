"""
INSERÇÃO EM LOTE: ChromaDB | Métrica configurável: L2 ou Cosine

Exemplo de uso:
    python lote_inserir_chromadb.py --metric l2
    python lote_inserir_chromadb.py --metric cosine
"""

import argparse
import chromadb
import numpy as np
import time
import csv
import gc

#Configuração

DATA_FILE        = "sift/subsets/sift_base_1000000.npy"
TOTAL_VECTORS    = 100_000
BATCH_SIZES      = [100, 1_000, 10_000, 50_000]
COLLECTION_NAME  = "batch_test"
CHROMA_PATH      = "./chroma_batch_test"
MAX_CHROMA_BATCH = 5_000   # limite físico da arquitectura SQLite do ChromaDB, não aceita transações maiores

METRIC_CONFIG = {
    "l2": {
        "hnsw_space": "l2",
        "normalize":  False,
        "csv_file":   "resultados_batch_chromadb_L2.csv",
        "label":      "L2",
    },
    "cosine": {
        "hnsw_space": "cosine",
        "normalize":  True,
        "csv_file":   "resultados_batch_chromadb_Cosine.csv",
        "label":      "Cosine",
    },
}


#Utils
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


#Benchmark
def run_batch_test(metric: str):
    cfg = METRIC_CONFIG[metric]

    print(f"A carregar {TOTAL_VECTORS} vetores para a memória...")
    vecs = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)

    if cfg["normalize"]:
        vecs = normalize(vecs)

    # No ChromaDB ele consume listas em Python, em de numpy arrays diretamente
    vectors = vecs.tolist()
    ids     = [str(i) for i in range(TOTAL_VECTORS)]

    with open(cfg["csv_file"], "w", newline="") as f:
        csv.writer(f).writerow(["DB", "Metric", "Batch_Size",
                                 "Time_s", "Throughput_vec_s"])

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for b_size in BATCH_SIZES:
        print(f"\n[CHROMADB {cfg['label']}] Lote: {b_size}")

        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space":           cfg["hnsw_space"],
                "hnsw:M":               16,
                "hnsw:construction_ef": 200,
            }
        )

        t0 = time.perf_counter()

        for i in range(0, TOTAL_VECTORS, b_size):
            end_idx      = min(i + b_size, TOTAL_VECTORS)
            batch_vectors = vectors[i:end_idx]
            batch_ids     = ids[i:end_idx]

            # Sub-loteamento para respeitar o limite do SQLite
            for j in range(0, len(batch_vectors), MAX_CHROMA_BATCH):
                sub_end = min(j + MAX_CHROMA_BATCH, len(batch_vectors))
                collection.add(
                    embeddings=batch_vectors[j:sub_end],
                    ids=batch_ids[j:sub_end]
                )

        time_taken = time.perf_counter() - t0
        throughput = TOTAL_VECTORS / time_taken

        print(f"  Tempo: {time_taken:.2f}s | Throughput: {throughput:.0f} vec/s")

        with open(cfg["csv_file"], "a", newline="") as f:
            csv.writer(f).writerow([
                "ChromaDB", cfg["label"], b_size,
                round(time_taken, 2), round(throughput, 0)
            ])

        gc.collect()

    print(f"\nFeito! Ficheiro: {cfg['csv_file']}")


#Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch insertion ChromaDB")
    parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
    args = parser.parse_args()
    run_batch_test(args.metric)
