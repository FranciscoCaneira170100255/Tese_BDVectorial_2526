import numpy as np
import time, threading, csv
import docker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
from qdrant_client.http.models import CollectionStatus

# CONFIG
DATA_FILE = "sift/subsets/sift_base_1000000.npy"
CSV_FILE = "cpu_qdrant_timeseries.csv"
CONTAINER_NAME = "qdrant"

M = 16
EF_CONSTRUCTION = 200

vectors = np.load(DATA_FILE)

client = QdrantClient(host="localhost", port=6333)
docker_client = docker.from_env()

# RESET
if client.collection_exists("cpu_test"):
    client.delete_collection("cpu_test")

client.create_collection(
    collection_name="cpu_test",
    vectors_config=VectorParams(size=128, distance=Distance.EUCLID),
    hnsw_config=HnswConfigDiff(m=M, ef_construct=EF_CONSTRUCTION)
)

 #--- ESTRUTURA PARA SeRIE TEMPORAL ---
cpu_data = [] 
running = True
index_started = False
start_time = 0

def monitor():
    container = docker_client.containers.get(CONTAINER_NAME)
    import psutil

    while running:
        if not index_started:
            time.sleep(0.1)
            continue

        stats = container.stats(stream=False)

        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - \
                    precpu_stats.get("cpu_usage", {}).get("total_usage", 0)

        system_delta = cpu_stats.get("system_cpu_usage", 0) - \
                       precpu_stats.get("system_cpu_usage", 0)

        num_cpus = len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", []))
        if num_cpus == 0:
            num_cpus = psutil.cpu_count()

        if system_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * num_cpus * 100
            elapsed = time.time() - start_time
            cpu_data.append((elapsed, cpu_percent))

        time.sleep(0.5)

t = threading.Thread(target=monitor)
t.start()

# INDEX
start_time = time.time()
index_started = True

client.upload_collection(
    collection_name="cpu_test",
    vectors=vectors,
    ids=list(range(len(vectors))),
    batch_size=1000
)

# Aguardar otimizacao do indice em background
while True:
    info = client.get_collection("cpu_test")
    if info.status == CollectionStatus.GREEN and info.optimizer_status == "ok":
        break
    time.sleep(1)

running = False
t.join()

# SAVE CSV (Formato Serie Temporal para a Figura)
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time_s", "CPU_Percent"])
    for elapsed_time, cpu in cpu_data:
        writer.writerow([round(elapsed_time, 2), round(cpu, 2)])

print("Saved Time-Series:", CSV_FILE)