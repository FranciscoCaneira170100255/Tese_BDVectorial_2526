"""
BENCHMARK: Qdrant (L2 vs Cosine)
"""
import argparse
import numpy as np
import time
import csv
import docker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, SearchParams

#Config
CONTAINER_NAME  = "qdrant"
DATA_FILE       = "sift/subsets/sift_base_1000000.npy"
QUERIES_FILE    = "sift/sift_query.npy"
TOTAL_VECTORS   = 1_000_000
NUM_QUERIES     = 1_000
EF_CONSTRUCTION = 200
COLLECTION_NAME = "optimize_test"

M_VALUES         = [16, 24, 32]
EF_SEARCH_VALUES = [64, 128, 256]

METRIC_CONFIG = {
    "l2": {
        "distance":  Distance.EUCLID,
        "gt_file":   "sift/gt/gt_1000000.npy",
        "normalize": False,
        "csv_file":  "resultados_otimizacao_qdrant_L2.csv",
        "label":     "L2",
    },
    "cosine": {
        "distance":  Distance.COSINE,
        "gt_file":   "sift/gt_cosine/gt_1000000.npy",
        "normalize": True,
        "csv_file":  "resultados_otimizacao_qdrant_Cosine.csv",
        "label":     "Cosine",
    },
}

#Utils
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def calc_recall(results, ground_truth, k=10) -> float:
    hits = sum(len(set(res) & set(gt_row[:k])) for res, gt_row in zip(results, ground_truth))
    return (hits / (len(results) * k)) * 100

def get_container_ram_gb(container_name: str) -> float:
   #O Qdrant está dependente de ficheiros mmap.
   # O docker em si reporta o uso de mmap no campo de "usage" porque são ficheiros retidos na Page Cache
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        stats = container.stats(stream=False)
        # Uso total sem subtrair cache para alinhar com a metodologia da tese
        mem_usage = stats["memory_stats"]["usage"]
        return mem_usage / (1024 ** 3)
    except Exception as e:
        print(f"Aviso: Erro ao obter RAM: {e}")
        return 0.0

#Bench
def run_optimization(metric: str):
    cfg = METRIC_CONFIG[metric]
    client = QdrantClient(host="localhost", port=6333, timeout=300)

    print("A carregar dados...")
    base_vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)
    queries      = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)
    gt           = np.load(cfg["gt_file"])[:NUM_QUERIES]

    if cfg["normalize"]:
        base_vectors = normalize(base_vectors)
        queries      = normalize(queries)

    ids = list(range(TOTAL_VECTORS))

    with open(cfg["csv_file"], "w", newline="") as f:
        csv.writer(f).writerow(["DB", "Metric", "m", "ef_search", "Recall_10", "p50_ms", "RAM_GB"])

    for m in M_VALUES:
        print(f"\n=== m={m} ===")
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        #Força a flag on_disk = True, de forma a evitar que o Qdrant aloque toda a sua estrutura,
        # diretamente no heap do Rust, obriga assim a partilhar recursos via mmap
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=cfg["distance"], on_disk=True),
            hnsw_config=HnswConfigDiff(m=m, ef_construct=EF_CONSTRUCTION, on_disk=True)
        )

        print(f"A fazer upload de {TOTAL_VECTORS} vetores...")
        client.upload_collection(collection_name=COLLECTION_NAME, vectors=base_vectors, ids=ids, batch_size=10_000)

        print("A aguardar que o indice fique pronto (status: green)...")
        while True:
            info = client.get_collection(COLLECTION_NAME)
            if str(info.status).lower() == "green": break
            time.sleep(5)

        # Ao fazer a ingestão, acumula-se muito "lixo" temporário na RAM
        # Restart para limpar a RAM de ingestão e ficar só com o índice mapeado
        print("A reiniciar o contentor Qdrant para limpar RAM de ingestao...")
        client.close() 
        docker_client = docker.from_env()
        docker_client.containers.get(CONTAINER_NAME).restart()
        time.sleep(10)
        
        client = QdrantClient(host="localhost", port=6333, timeout=300)
        
        # Warmup para forçar o SO a carregar os ficheiros mmap para a Page Cache
        print("A executar warm-up (a carregar mmap para RAM)...")
        for q in queries[:100]:
            try:
                client.search(collection_name=COLLECTION_NAME, query_vector=q, limit=10, search_params=SearchParams(hnsw_ef=64))
            except Exception:
                time.sleep(2)

        ram_gb = get_container_ram_gb(CONTAINER_NAME)
        print(f"Pronto | RAM Ocupada (Total): {ram_gb:.3f} GB")

        for ef_s in EF_SEARCH_VALUES:
            print(f"  -> ef_search={ef_s}")
            latencies, results = [], []
            for q in queries:
                t0 = time.perf_counter()
                res = client.search(collection_name=COLLECTION_NAME, query_vector=q, limit=10, search_params=SearchParams(hnsw_ef=ef_s))
                latencies.append((time.perf_counter() - t0) * 1000)
                results.append([int(hit.id) for hit in res])

            recall = calc_recall(results, gt)
            p50    = np.percentile(latencies, 50)
            print(f"     Recall={recall:.2f}% | p50={p50:.2f}ms")

            with open(cfg["csv_file"], "a", newline="") as f:
                csv.writer(f).writerow([
                    "Qdrant", cfg["label"], m, ef_s, 
                    round(recall, 2), round(p50, 2), round(ram_gb, 3)
                ])

    print(f"\nBenchmark feito! Resultados em: {cfg['csv_file']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
    args = parser.parse_args()
    run_optimization(args.metric)
