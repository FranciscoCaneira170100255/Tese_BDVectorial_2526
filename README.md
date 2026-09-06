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
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
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

### Realizar a conversão de .fvecs /ivecs para npy
```python
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
```
### Gerar subconjuntos por escala (100 / 1.000 / 10.000 / 100.000 / 1.000.000 vetores)
```python
import numpy as np
from pathlib import Path

# ---- CONFIGURAÇÃO ----
INPUT_BASE = Path("data/sift/sift_base.npy")
OUTPUT_DIR = Path("data/sift/subsets")

SUBSET_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]

# ---------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] A carregar dataset base: {INPUT_BASE}")
    base_vectors = np.load(INPUT_BASE)

    total = base_vectors.shape[0]
    print(f"[INFO] Total de vectores disponíveis: {total}")

    for size in SUBSET_SIZES:
        if size > total:
            print(f"[WARN] A ignorar subset {size} (não existe)")
            continue

        subset = base_vectors[:size]  # determinístico
        output_file = OUTPUT_DIR / f"sift_base_{size}.npy"

        np.save(output_file, subset)
        print(f"[OK] Criado {output_file} com shape {subset.shape}")

    print("[DONE] Subsets gerados com sucesso")

if __name__ == "__main__":
    main()
```
### 4. Iniciar os contentores
Cada motor tem o seu próprio docker-compose.yml, com CPU e RAM limitadas, via cgroups, 
para garantir a paridade experimental (estes três contentores estão limitados a 4 núcleos de CPU e 8 GB de RAM):

**`docker-compose-qdrant.yml`**
```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.9.4
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    mem_limit: 8G
    memswap_limit: 8G
    cpus: "4"
    restart: unless-stopped

volumes:
  qdrant_data:
```

**`docker-compose-pgvector.yml`**
```yaml
services:
  postgres:
    image: pgvector/pgvector:0.8.1-pg16-bookworm
    container_name: pgvector
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vectordb
    shm_size: '4gb'
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    mem_limit: 8G
    memswap_limit: 8G
    cpus: "4"
    restart: unless-stopped

volumes:
  pgvector_data:
```

**`docker-compose-chromadb.yml`**
```yaml
services:
  chromadb:
    image: chromadb/chroma:1.5.5
    container_name: chromadb
    ports:
      - "8000:8000"
    volumes:
      - chromadb_data:/chroma/.chroma/index
    mem_limit: 8g
    memswap_limit: 8g
    cpus: 4
    restart: unless-stopped

volumes:
  chromadb_data:
```
### Inicializar cada contentor:

```bash
docker compose -f docker-compose-qdrant.yml up -d
docker compose -f docker-compose-pgvector.yml up -d
docker compose -f docker-compose-chromadb.yml up -d
```

### Analisar se os contentores executam os serviços e os limites dos recursos estão a ser aplicados corretamente:
```bash
docker ps
docker inspect qdrant | grep -i memory
docker inspect pgvector | grep -i memory
docker inspect chromadb | grep -i memory
```

### 5. Povoar os motores (preparando para os ensaios do disco e RAM)
Os ensaios de consumo de disco, monitorização de RAM e latência detalhada requerem que cada motor contenha, previamente, uma coleção com 1M de vetores. Os scripts de preparação criam as coleções disk_test_l2 e disk_test_cosine em cada sistema:

```bash
python ram_inserir_qdrant.py   --metric l2
python ram_inserir_pgvector.py --metric l2
python ram_inserir_chromadb.py --metric l2
```
(procedimento idêntico com a opção --metric cosine para a Similaridade de Cosseno)

### 6. Executar os ensaios
A maioria dos scripts aceita como argumentos o `--metric l2` ou o `--metric cosine`:

```bash
python bench_qdrant.py     --metric l2
python otimizacao_qdrant.py --metric cosine
python latencia_pgvector.py --metric l2
python escalabilidade_chromadb.py --metric cosine
```

### Pré-requisito específico: ensaios de arranque a frio (`cold_warm_*.py`)
Estes scripts limpam a page cache do sistema operativo antes de cada réplica para garantir que o arranque ocorre efetivamente a frio. Este passo exige executar echo 3 > /proc/sys/vm/drop_caches com privilégios de administração, abortando a execução caso as permissões não sejam concedidas. É necessário configurar o comando no ficheiro sudoers sem pedido de palavra-passe:

```bash
sudo visudo
```
Adicionar a seguinte linha (substituindo `<utilizador>` pelo nome do utilizador do sistema operativo associado):

```
<utilizador> ALL=(ALL) NOPASSWD: /bin/sh -c echo 3 > /proc/sys/vm/drop_caches
```

### Pré-requisitos estruturais por script
Nem todos os ensaios recorrem à mesma estrutura de dados. Antes de executar, convém validar a presença de tabelas e de coleções importantes:

| Script | Estrutura esperada | Origem da criação |
|---|---|---|
| `latencia_qdrant.py`, `latencia_chromadb.py` | Coleções `disk_test_l2` / `disk_test_cosine` | `ram_inserir_qdrant.py` / `ram_inserir_chromadb.py` (passo 5) |
| `latencia_pgvector.py` | Tabela `items` | Não é criada pelos scripts de preparação: o script cria o índice HNSW se não existir, mas a tabela e os vetores têm de ser carregados previamente |
| `escalabilidade_pgvector.py` | Tabela `items` | Não é criada pelos scripts de preparação: o script recria o índice HNSW no arranque |
| `escalabilidade_qdrant.py`, `escalabilidade_chromadb.py` | Coleções `sift1m_qdrant` / `sift1m_chroma` | Não são criadas por nenhum script deste repositório (ver nota abaixo) |
| `cold_warm_chromadb.py` | Coleção `coldwarm_1m` | criada automaticamente pelo próprio script, se não existir |
| `cold_warm_qdrant.py` | Coleção `cpu_test` | Não é criada pelo script, devendo existir previamente |
| `bench_*.py`, `lote_inserir_*.py`, `disco_*.py`, `otimizacao_*.py` | Criam e removem as próprias estruturas | Gestão automática pelo script |

Nota sobre os ensaios de otimização (`otimizacao_*.py`): correspondem à fase de maior exigência computacional de toda a campanha. Cada execução reconstrói o índice HNSW na íntegra sobre 1M de vetores em três ocasiões distintas (uma para cada valor de $m$), o que pode representar várias horas de processamento. O ficheiro `otimizacao_chromadb.py` reinicia o contentor a cada novo valor de `ef_search`, procedimento intencional e indispensável para forçar a biblioteca `hnswlib` a recarregar os metadados da coleção. Ainda assim, o valor de `ef_search` no ChromaDB mostrou-se invariante quanto ao `Recall@10`, consistindo numa particularidade arquitetural detalhada no documento da dissertação.

Limitação de reprodutibilidade conhecida: As coleções `sift1m_qdrant`, `sift1m_chroma` e `cpu_test` foram povoadas manualmente na campanha experimental e não integram um script de preparação próprio neste repositório. Dado que os ensaios associados avaliam apenas throughput e latência (e não `Recall@10`), a validade das medições mantém-se inalterada. Contudo, a reprodução exata requer a criação prévia destas coleções com os mesmos parâmetros HNSW ($m=16$, $ef\_construction=200$) e 1M de vetores. Esta restrição metodológica encontra-se documentada na dissertação como trabalho futuro.

### 7. Monitorização temporal de RAM
O script de análise do consumo de memória RAM atua como invólucro do comando em teste, aceitando o identificador do contentor, o ficheiro de destino e a instrução a monitorizar:

```bash
python3 monitorizar_ram.py --container chromadb \
                           --output ram_chromadb_l2.csv \
                           --cmd "python3 ram_inserir_chromadb.py --metric l2"

python3 monitorizar_ram.py --container qdrant \
                           --output ram_qdrant_cosine.csv \
                           --cmd "python3 ram_inserir_qdrant.py --metric cosine"
```

A opção --clean elimina o volume Docker persistente antes do início do teste, garantindo que o registo parte de um estado limpo.

***Nomenclatura dos volumes Docker:*** O ficheiro ``monitorizar_ram.py`` inclui nomes estáticos para os volumes (``docker_qdrant_data``, ``docker_chroma_data``, ``docker_pgvector_data``). O prefixo ``docker_`` decorre do padrão do Docker Compose (``<pasta>_<volume>``), assumindo que a inicialização ocorreu a partir de uma pasta denominada ``docker/``. Caso os comandos sejam executados a partir de outro diretório, os volumes terão prefixos distintos e a opção ``--clean`` não localizará o volume correto. Os nomes em uso devem ser confirmados através de ``docker volume ls``, ajustando-se a estrutura ``CONFIG`` no código fonte quando aplicável.

### 8. Análise estatística
Após gerar todos os ficheiros ``results_*.csv`` referentes ao ensaio base:

```bash
python analise_estatistica.py
```
**Importante:** O script procura os ficheiros ```results_*.csv``` na mesma pasta onde se encontra guardado. Deve verificar-se que os seis ficheiros de resultados (`results_qdrant_l2.csv`, `results_pgvector_l2.csv`, `results_chromadb_l2.csv` e os respetivos equivalentes para Cosseno) estão na mesma diretoria antes do arranque.

São produzidos os ficheiros ```analise_estatistica_l2.csv``` e ```analise_estatistica_cosine.csv```, contendo os p-valores dos testes de Welch, Mann-Whitney U e Shapiro-Wilk para cada volume vetorial.
 
A pasta `Análise Estatística/jamovi/` contém a verificação independente destes testes, realizada na aplicação [jamovi](https://www.jamovi.org). Inclui os dados em formato longo (`jamovi_l2.csv` e `jamovi_cosine.csv`) e os relatórios exportados para as quatro comparações à escala de 1M de vetores. Para replicar:

- Abrir o csv no jamovi;
- Criar um filtro (`Scale == 1000000 and Motor != "ChromaDB"`, por exemplo);
- Executar indo a `Análises` -> `Testes t` -> `Teste T para amostras independentes`, marcando Welch, Mann-Whitney U e os testes de pressupostos.

**Nota de reprodutibilidade:** A campanha de ensaios não recorre a um orquestrador centralizado (como um encadeamento global via Docker Compose). Cada rotina deve ser executada individualmente na ordem definida no protocolo de testes, diferenciando motor e métrica. Esta decisão metodológica encontra-se devidamente justificada no capítulo de limitações da dissertação.
