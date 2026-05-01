import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import time, threading, psutil, csv

# CONFIG
DATA_FILE = "sift/subsets/sift_base_1000000.npy"
CSV_FILE = "cpu_pgvector_timeseries_cosine.csv"

M = 16
EF_CONSTRUCTION = 200

vectors = np.load(DATA_FILE).tolist()

conn = psycopg2.connect(
    dbname="vectordb", user="postgres", password="postgres", host="localhost"
)
cur = conn.cursor()

# RESET
cur.execute("DROP TABLE IF EXISTS items;")
cur.execute("CREATE TABLE items (id SERIAL PRIMARY KEY, embedding vector(128));")

# INSERT
execute_values(
    cur,
    "INSERT INTO items (embedding) VALUES %s",
    [(v,) for v in vectors],
    page_size=5000
)
conn.commit()

cur.execute("SET maintenance_work_mem = '2GB';")
conn.commit()

# --- ESTRUTURA PARA SeRIE TEMPORAL ---
cpu_data = [] # Vai guardar tuplos (Tempo, CPU)
running = True
index_started = False
start_time = 0

def monitor():
    while running:
        if not index_started:
            time.sleep(0.1)
            continue
            
        total_cpu = 0
        for p in psutil.process_iter(['name', 'cpu_percent']):
            try:
                if 'postgres' in p.info['name']:
                    # interval=0.5 funciona como o nosso "time.sleep" 
                    total_cpu += p.cpu_percent(interval=0.5)
            except:
                pass
                
        elapsed = time.time() - start_time
        cpu_data.append((elapsed, total_cpu))

t = threading.Thread(target=monitor)
t.start()

# INDEX
start_time = time.time()
index_started = True 

cur.execute(f"""
CREATE INDEX idx_hnsw ON items
USING hnsw (embedding vector_cosine_ops)
WITH (m = {M}, ef_construction = {EF_CONSTRUCTION});
""")
conn.commit()

running = False
t.join()

# SAVE CSV (Formato Serie Temporal para a Figura)
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time_s", "CPU_Percent"])
    for elapsed_time, cpu in cpu_data:
        writer.writerow([round(elapsed_time, 2), round(cpu, 2)])

print("Saved Time-Series:", CSV_FILE)