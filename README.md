# Análise Comparativa de Bases de Dados Vetoriais: Qdrant vs. pgvector vs. ChromaDB

Repositório de código-fonte associado à dissertação de Mestrado **"Análise Comparativa de Sistemas de Gestão de Bases de Dados Vetoriais"**, desenvolvida no âmbito do Mestrado em Informática Aplicada da Universidade Politécnica de Santarém (2025/2026).

Autor: Francisco Caneira

---

## Sobre o projeto

Esta dissertação avalia empiricamente três paradigmas distintos de bases de dados vetoriais: o motor nativo **Qdrant** (Rust), a extensão relacional **pgvector** (PostgreSQL) e o sistema *wrapper* **ChromaDB** (Python/C++). Estes testes utilizam o conjunto de dados [SIFT1M](http://corpus-texmex.irisa.fr/), avaliando o desempenho operacional, a eficiência no uso de recursos e o comportamento arquitetural sob as métricas de Distância Euclidiana (L2) e Similaridade de Cosseno.

Os três sistemas executam, de forma isolada, em contentores Docker, com atribuição equivalente de CPU e memória RAM para assegurar a comparabilidade dos resultados.

## Estrutura do repositório

O código está organizado por cenário experimental, onde cada diretoria corresponde a uma secção de resultados da dissertação:

| Pasta | O que mede | Secção da tese |
|---|---|---|
| `Benchmark/` | Desempenho de base: latência (p50/p95/p99), *throughput*, tempo de indexação e RAM, para escalas de 100 a 1M vetores | Desempenho Base |
| `Otimização/` | Varrimento paramétrico do HNSW ($m \in \{16,24,32\}$, $ef\_search \in \{64,128,256\}$) | Otimização Paramétrica |
| `Desempenho Inserção em Lote/` | *Throughput* de inserção para diferentes tamanhos de lote (100 a 50.000 vetores) | Inserção em Lote |
| `Cold Start Warm Cache/` | Comparação entre a primeira consulta após reinício e o regime de cache quente | Arranque a Frio vs. Cache Quente |
| `Consumo disco/` | Medição do espaço em disco ocupado por dados, índice e metadados de cada motor | Consumo de Disco |
| `Escalabilidade/` | *Throughput* e latência sob carga concorrente crescente (1 a 32 *threads*) | Escalabilidade sob Concorrência |
| `Latência detalhada/` | Distribuição completa de latência (percentis, assimetria, curtose) sobre 10.000 consultas sequenciais | Distribuição Detalhada de Latência |
| `Monitorização RAM/` | Evolução temporal do consumo de RAM durante a construção do índice | Análise Temporal de RAM |
| `Análise Estatística/` | Testes de significância (Welch, Mann-Whitney U, Shapiro-Wilk) sobre os resultados de latência | Testes de Significância Estatística |

Cada pasta contém os *scripts* Python usados para recolha das métricas (um por motor: `*_qdrant.py`, `*_pgvector.py`, `*_chromadb.py`) e os respetivos ficheiros de dados em formato CSV.

## Motores avaliados

| Motor | Tipo | Linguagem do núcleo | Algoritmo de indexação |
|---|---|---|---|
| [Qdrant](https://qdrant.tech/) | Motor vetorial nativo | Rust | HNSW |
| [pgvector](https://github.com/pgvector/pgvector) | Extensão para PostgreSQL | C | HNSW |
| [ChromaDB](https://www.trychroma.com/) | Base de dados vetorial *wrapper* | Python + C++ (`hnswlib`) | HNSW |

## Requisitos

- Ubuntu Server (testado em 24.04 LTS)
- Docker CE + Docker Compose
- Python 3.12
- Bibliotecas: `qdrant-client`, `psycopg2-binary`, `pgvector`, `chromadb`, `numpy`, `pandas`, `scipy`, `tqdm`, `docker` (SDK Python)

## Como reproduzir os ensaios

### 1. Instalar o Docker

```bash
sudo apt update
sudo apt install ca-certificates curl gnupg -y

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  [https://download.docker.com/linux/ubuntu](https://download.docker.com/linux/ubuntu) \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin -y

sudo usermod -aG docker $USER
newgrp docker
```
### 2. Criar um ambiente virtual Python
```bash
sudo apt install python3-venv python3-full -y
cd ~/tese_benchmarking
python3 -m venv ambiente-virtual
source ambiente-virtual/bin/activate

pip install --upgrade pip
pip install qdrant-client psycopg2-binary pgvector chromadb numpy pandas scipy tqdm docker
```
### 3. Descarregar e preparar o dataset do SIFT1M
```bash
mkdir ~/sift1M && cd ~/sift1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xvzf sift.tar.gz
```

### 4. Realizar a conversão de .fvecs /ivecs para npy
```bash
import numpy as np

def read_fvecs(fname):
    with open(fname, 'rb') as f:
        data = f.read()
    dim = np.frombuffer(data, dtype=np.int32)[0]
    vecs = np.frombuffer(data, dtype=np.float32).reshape(-1, dim + 1)
    return vecs[:, 1:]

def read_ivecs(fname):
    with open(fname, 'rb') as f:
        data = f.read()
    dim = np.frombuffer(data, dtype=np.int32)[0]
    vecs = np.frombuffer(data, dtype=np.int32).reshape(-1, dim + 1)
    return vecs[:, 1:]

base = read_fvecs("sift/sift_base.fvecs")
query = read_fvecs("sift/sift_query.fvecs")
gt = read_ivecs("sift/sift_groundtruth.ivecs")

np.save("sift_base.npy", base)
np.save("sift_query.npy", query)
np.save("sift_gt.npy", gt)

print("Conversão concluída:")
print("Base:", base.shape)
print("Query:", query.shape)
print("GT:", gt.shape)

