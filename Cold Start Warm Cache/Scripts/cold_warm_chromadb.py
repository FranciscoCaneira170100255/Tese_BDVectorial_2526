"""
Este script avalia o comportamento do ChromaDB em cenários de Cold Start
e Warm Start utilizando índices HNSW sobre o dataset SIFT.
"""
import argparse
import numpy as np
import time
import csv
import subprocess
import gc
import docker
import chromadb

# --- Argumentos ---------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["l2", "cosine"], required=True)
args = parser.parse_args()

# --- Configuração -------------------------------------------------------------
DATA_FILE        = "sift/subsets/sift_base_1000000.npy"
QUERIES_FILE     = "sift/sift_query.npy"
CONTAINER_NAME   = "chromadb"
COLLECTION_NAME  = "coldwarm_1m"
TOTAL_VECTORS    = 1_000_000
BATCH_SIZE       = 5_000
NUM_QUERIES      = 1_000
REPLICATES       = 10

if args.metric == "l2":
    METRIC_TEST  = "L2"
    HNSW_SPACE   = "l2"
    GT_FILE      = "sift/gt/gt_1000000.npy"
    CSV_FILE     = "resultados_cold_warm_chromadb_L2.csv"
else:
    METRIC_TEST  = "Cosine"
    HNSW_SPACE   = "cosine"
    GT_FILE      = "sift/gt_cosine/gt_1000000.npy"
    CSV_FILE     = "resultados_cold_warm_chromadb_Cosine.csv"

# --- Utils --------------------------------------------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    """Normaliza vetores para norma unitária."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms

def calculate_recall(results, ground_truth):
    """Calcula o Recall@10 comparando resultados com ground truth."""
    hits = sum(len(set(res) & set(gt_row[:10])) for res, gt_row in zip(results, ground_truth))
    return (hits / (len(results) * 10)) * 100

def wait_for_chromadb(host="localhost", port=8000) -> float:
    """Espera até o ChromaDB estar pronto a responder."""
    start = time.time()
    while True:
        try:
            chromadb.HttpClient(host=host, port=port).heartbeat()
            return time.time() - start
        except Exception:
            time.sleep(0.5)

def drop_os_page_cache() -> None:
    """
    Limpa a page cache do SO para garantir que o cold start é realmente cold.
    Deve ser chamado IMEDIATAMENTE antes da primeira query, não no início da réplica.
    """
    sync_result = subprocess.run(["sync"], capture_output=True, text=True)
    if sync_result.returncode != 0:
        raise RuntimeError(
            f"Falha ao executar 'sync' antes da limpeza da cache: "
            f"{sync_result.stderr.strip()}"
        )
    
    drop_result = subprocess.run(
        ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        capture_output=True, text=True
    )
    if drop_result.returncode != 0:
        raise RuntimeError(
            "Falha ao limpar a page cache do SO (drop_caches). "
            "A medição de Cold Start ficaria inválida. Verifica "
            "permissões de 'sudo' (NOPASSWD) para este comando. "
            f"Detalhe do erro: {drop_result.stderr.strip()}"
        )
    time.sleep(1)

# --- Carregar dados -----------------------------------------------------------
print("A carregar queries e ground truth...")
queries = np.load(QUERIES_FILE)[:NUM_QUERIES].astype(np.float32)
gt      = np.load(GT_FILE)[:NUM_QUERIES]

if args.metric == "cosine":
    queries = normalize(queries)

print("\n[FASE 0] A verificar se a coleção existe no container...")
wait_for_chromadb()
client = chromadb.HttpClient(host="localhost", port=8000)

# Bloco que verifica se a coleção não tiver exatamente 1M de vetores, recria-se
try:
    col   = client.get_collection(COLLECTION_NAME)
    count = col.count()
    if count == TOTAL_VECTORS:
        print(f"  Coleção '{COLLECTION_NAME}' existe com {count:,} vetores.")
    else:
        print(f"  Coleção existe mas tem {count:,} vetores (esperado {TOTAL_VECTORS:,}). A recriar...")
        client.delete_collection(COLLECTION_NAME)
        raise Exception("recriar")
except Exception:
    print(f"  A criar coleção '{COLLECTION_NAME}' com {TOTAL_VECTORS:,} vetores...")
    print("  A carregar vetores para a RAM (pode demorar)...")
    vecs = np.load(DATA_FILE)[:TOTAL_VECTORS].astype(np.float32)
    if args.metric == "cosine":
        vecs = normalize(vecs)
    ids = [str(i) for i in range(TOTAL_VECTORS)]
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space":           HNSW_SPACE,
            "hnsw:M":               16,
            "hnsw:construction_ef": 200,
            "hnsw:search_ef":       64,
        }
    )
    
    for i in range(0, TOTAL_VECTORS, BATCH_SIZE):
        end = min(i + BATCH_SIZE, TOTAL_VECTORS)
        collection.add(
            embeddings=vecs[i:end].tolist(),
            ids=ids[i:end]
        )
        if (i // BATCH_SIZE) % 40 == 0:
            print(f"    {end:,}/{TOTAL_VECTORS:,}")
    
    print(f"  Coleção criada com {collection.count():,} vetores")
    del vecs, ids

# --- Benchmark Cold/Warm ------------------------------------------------------
headers = [
    "Replicate", "DB", "Metric",
    "First_Resp_Cold_ms", "First_Resp_Warm_ms",
    "Cold_First10_Mean_ms",
    "p50_Cold_ms", "p50_Warm_ms",
    "p95_Cold_ms", "p95_Warm_ms",
    "Recall_Cold", "Recall_Warm",
    "Stabilization_Time_s", "Index_Load_Time_s"
]

with open(CSV_FILE, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=headers).writeheader()

docker_client = docker.from_env()

for r in range(1, REPLICATES + 1):
    print(f"\n[CHROMADB {METRIC_TEST}] Replica {r}/{REPLICATES}")
    
    # 1. Parar o container completamente (com proteção)
    print("  [Cold] A parar container...")
    try:
        container = docker_client.containers.get(CONTAINER_NAME)
        # Verificar se o container está realmente a correr antes de parar
        if container.status == "running":
            container.stop()
            print(f"    Container parado com sucesso.")
        else:
            print(f"    Container já estava parado (status={container.status}). A ignorar .stop().")
        time.sleep(2)  # Aguardar paragem completa
    except Exception as e:
        print(f"    [Aviso] Exceção ao parar container: {e}")
        print(f"    A continuar mesmo assim...")
        time.sleep(2)

    # 2. Reiniciar o container
    print("  [Cold] A reiniciar container...")
    t_load_start = time.time()
    docker_client.containers.get(CONTAINER_NAME).start()
    load_time = wait_for_chromadb()
    
    # 3. Limpar page cache DEPOIS do restart (garante cold start real)
    print("  [Cold] A limpar page cache do SO...")
    drop_os_page_cache()
    
    # 4. Conectar e obter coleção
    client     = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_collection(COLLECTION_NAME)
    
    # --- Fase Cold ---
    cold_latencies, cold_results = [], []
    t_start_cold = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        res_raw = collection.query(query_embeddings=[q.tolist()], n_results=10)
        cold_results.append([int(x) for x in res_raw["ids"][0]]) 
        cold_latencies.append(time.perf_counter() - t0)
    cold_total_time = time.perf_counter() - t_start_cold
    
    # --- Fase Warm ---
    warm_latencies, warm_results = [], []
    for q in queries:
        t0 = time.perf_counter()
        res_raw = collection.query(query_embeddings=[q.tolist()], n_results=10)
        warm_results.append([int(x) for x in res_raw["ids"][0]]) 
        warm_latencies.append(time.perf_counter() - t0)
    
    cold_ms = np.array(cold_latencies) * 1000
    warm_ms = np.array(warm_latencies) * 1000
    
    cold_first10_mean = float(np.mean(cold_ms[:10]))
    
    metrics = {
        "Replicate":            r,
        "DB":                   "ChromaDB",
        "Metric":               METRIC_TEST,
        "First_Resp_Cold_ms":   cold_ms[0],
        "First_Resp_Warm_ms":   warm_ms[0],
        "Cold_First10_Mean_ms": cold_first10_mean,
        "p50_Cold_ms":          np.percentile(cold_ms, 50),
        "p50_Warm_ms":          np.percentile(warm_ms, 50),
        "p95_Cold_ms":          np.percentile(cold_ms, 95),
        "p95_Warm_ms":          np.percentile(warm_ms, 95),
        "Recall_Cold":          calculate_recall(cold_results, gt),
        "Recall_Warm":          calculate_recall(warm_results, gt),
        "Stabilization_Time_s": max(0, cold_total_time - sum(warm_latencies)),
        "Index_Load_Time_s":    load_time
    }
    
    with open(CSV_FILE, "a", newline="") as f:  
        csv.DictWriter(f, fieldnames=headers).writerow(metrics)
    
    print(f"  Cold p50={metrics['p50_Cold_ms']:.2f}ms | "  
          f"Warm p50={metrics['p50_Warm_ms']:.2f}ms | "
          f"Recall={metrics['Recall_Cold']:.1f}%")
    
    del cold_latencies, cold_results, warm_latencies, warm_results, cold_ms, warm_ms
    gc.collect()

print(f"\nFeito! Ficheiro: {CSV_FILE}")
