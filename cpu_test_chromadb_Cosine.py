import chromadb
import numpy as np
import time, threading, psutil, csv

# CONFIG
DATA_FILE = "sift/subsets/sift_base_1000000.npy"
CSV_FILE = "cpu_chroma_timeseries_cosine.csv"

M = 16
EF_CONSTRUCTION = 200

vectors = np.load(DATA_FILE).tolist()

client = chromadb.PersistentClient(path="./chroma_cpu_test")

try:
    client.delete_collection("cpu_test")
except Exception:
    pass

collection = client.create_collection(
    name="cpu_test",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": M,
        "hnsw:construction_ef": EF_CONSTRUCTION
    }
)

# --- ESTRUTURA PARA TEMPORAL ---
cpu_data = []
running = True
index_started = False
start_time = 0
process = psutil.Process()

# Inicializar o contador do psutil
process.cpu_percent(interval=None)

def monitor():
    while running:
        if not index_started:
            time.sleep(0.1)
            continue
            
        cpu = process.cpu_percent(interval=0.5)
        elapsed = time.time() - start_time
        cpu_data.append((elapsed, cpu))

t = threading.Thread(target=monitor)
t.start()

batch_size = 5000

start_time = time.time()
index_started = True

for i in range(0, len(vectors), batch_size):
    collection.add(
        embeddings=vectors[i:i+batch_size],
        ids=[str(j) for j in range(i, min(i+batch_size, len(vectors)))]
    )

running = False
t.join()

# SAVE CSV (Formato para a Figura)
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time_s", "CPU_Percent"])
    for elapsed_time, cpu in cpu_data:
        writer.writerow([round(elapsed_time, 2), round(cpu, 2)])

print("Saved Time-Series:", CSV_FILE)