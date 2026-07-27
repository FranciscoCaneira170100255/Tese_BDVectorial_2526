"""
BENCHMARK: ChromaDB (L2 vs Cosine)
"""
import httpx
import argparse
import chromadb
import numpy as np
import time
import csv
import gc
import docker

#Config
CONTAINER_NAME   = "chromadb"
DATA_FILE        = "sift/subsets/sift_base_1000000.npy"
QUERIES_FILE     = "sift/sift_query.npy"
TOTAL_VECTORS    = 1_000_000
NUM_QUERIES      = 1_000
MAX_BATCH        = 5_000 
EF_CONSTRUCTION  = 200
COLLECTION_NAME  = "optimize_test"
WARMUP_QUERIES   = 50

M_VALUES         = [16, 24, 32]
EF_SEARCH_VALUES = [64, 128, 256]

METRIC_CONFIG = {
    "l2": {
        "hnsw_space":  "l2",
        "gt_file":     "sift/gt/gt_1000000.npy",
        "normalize":   False,
        "csv_file":    "resultados_otimizacao_chromadb_L2.csv",
        "label":       "L2",
    },
    "cosine": {
        "hnsw_space":  "cosine",
        "gt_file":     "sift/gt_cosine/gt_1000000.npy",
        "normalize":   True,
        "csv_file":    "resultados_otimizacao_chromadb_Cosine.csv",
        "label":       "Cosine",
    },
}

# Utils
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def calc_recall(results, ground_truth, k=10) -> float:
    hits = sum(len(set(res) & set(gt_row[:k])) for res, gt_row in zip(results, ground_truth))
    return (hits / (len(results) * k)) * 100

def get_docker_ram_gb(docker_client, container_name: str) -> float:
    try:
        stats = docker_client.containers.get(container_name).stats(stream=False)
        mem_usage = stats["memory_stats"]["usage"]
        # O ChromaDB carrega o índice para a heap, não usa tanto mmap como o Qdrant.
        return mem_usage / (1024 ** 3)
    except Exception:
        return 0.0

def restart_container_and_connect(docker_client, container_name: str, host="localhost", port=8000):
    print("  A reiniciar container Docker...")
    container = docker_client.containers.get(container_name)
    container.restart()
    time.sleep(10) # Dar tempo ao SQLite e ao hnswlib para iniciarem as ligações
    
    client = chromadb.HttpClient(host=host, port=port)
    deadline = time.time() + 120 
    while time.time() < deadline:
        try:
            client.list_collections() 
            return client
        except (httpx.ReadError, Exception):
            time.sleep(2)
    raise Exception("O ChromaDB demorou mais de 2 minutos a reiniciar.")

# Main
def run_optimization(metric: str):
    cfg          = METRIC_CONFIG[metric]
    docker_client = docker.from_env()

    print("A carregar dados para a RAM...")
    base_vectors = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)
    queries_arr  = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)
    gt           = np.load(cfg["gt_file"])[:NUM_QUERIES]

    if cfg["normalize"]:
        base_vectors = normalize(base_vectors)
        queries_arr  = normalize(queries_arr)

    #O ChromaDB faz o processo da API via Python, convertendo o Numpy array para uma lista pura
    base_list    = base_vectors.tolist()
    queries_list = queries_arr.tolist()
    ids          = [str(i) for i in range(TOTAL_VECTORS)]

    with open(cfg["csv_file"], "w", newline="") as f:
        csv.writer(f).writerow(["DB", "Metric", "m", "ef_search", "Recall_10", "p50_ms", "RAM_GB"])

    client = None
    for m in M_VALUES:
        print(f"\n=== m={m} | A construir indice HNSW ===")
        client = restart_container_and_connect(docker_client, CONTAINER_NAME)
        
        try: client.delete_collection(COLLECTION_NAME)
        except Exception: pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space":           cfg["hnsw_space"],
                "hnsw:M":               m,
                "hnsw:construction_ef": EF_CONSTRUCTION,
            }
        )

        print(f"  A inserir {TOTAL_VECTORS:,} vetores...")
        for i in range(0, TOTAL_VECTORS, MAX_BATCH):
            end = min(i + MAX_BATCH, TOTAL_VECTORS)
            collection.add(embeddings=base_list[i:end], ids=ids[i:end])

        print("A aguardar 15s para as threads de background do ChromaDB estabilizarem...")
        time.sleep(15)
        
        ram_gb = get_docker_ram_gb(docker_client, CONTAINER_NAME)
        print(f"  Indice pronto. RAM={ram_gb:.2f} GB")

        for ef_s in EF_SEARCH_VALUES:
            print(f"\n  -> ef_search={ef_s}")
            
            #Alterar metadados na coleção
            collection.modify(metadata={"hnsw:search_ef": ef_s})
            #Limitações arquiteturais: O ChromaDB não realiza atualizações logo logo do ef_search
            #Sendo preciso forçar o processo C++ (hnswlib) a carregar os novos metadados via um restart
            print("A reiniciar o servidor para forçar o HNSW a assumir o novo ef_search...")
            client = restart_container_and_connect(docker_client, CONTAINER_NAME)
            collection = client.get_collection(COLLECTION_NAME)
            
            # Como a cache do disco e RAM cairam durante o restart, é preciso realizar um warmup
            print("A reconstruir a cache da RAM...")
            for q in queries_list[:500]:
                collection.query(query_embeddings=[q], n_results=10)

            #Medição real
            latencies, results = [], []
            for q in queries_list:
                t0      = time.perf_counter()
                res_raw = collection.query(query_embeddings=[q], n_results=10)
                latencies.append((time.perf_counter() - t0) * 1000)
                results.append([int(x) for x in res_raw["ids"][0]])

            recall = calc_recall(results, gt)
            p50    = np.percentile(latencies, 50)
            print(f"     Recall={recall:.2f}% | p50={p50:.2f}ms")

            with open(cfg["csv_file"], "a", newline="") as f:
                csv.writer(f).writerow([
                    "ChromaDB", cfg["label"], m, ef_s, 
                    round(recall, 2), round(p50, 2), round(ram_gb, 2)
                ])
        
        gc.collect()

    print(f"\nFeito! Ficheiro: {cfg['csv_file']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimizar ChromaDB")
    parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
    args = parser.parse_args()
    run_optimization(args.metric)
