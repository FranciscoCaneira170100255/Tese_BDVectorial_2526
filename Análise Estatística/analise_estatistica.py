"""
SCRIPT DE ANÁLISE ESTATÍSTICA 
Gera os p-valores (Teste t de Welch) para as Tabelas 4.1 e 4.2,
comparando pgvector e ChromaDB contra o baseline (Qdrant) na latência p95.
Inclui também Shapiro-Wilk e Mann-Whitney U para robustez metodológica.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings

# Ignorar avisos de precisão do Shapiro-Wilk para n=10 (é esperado e válido)
warnings.filterwarnings("ignore", module="scipy")

# --- Configuração ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALES = [100, 1_000, 10_000, 100_000, 1_000_000]
METRICS = ["l2", "cosine"]

# Nomes dos ficheiros CSV
CSV_FILES = {
    "l2": {
        "qdrant": os.path.join(BASE_DIR, "results_qdrant_l2.csv"),
        "pgvector": os.path.join(BASE_DIR, "results_pgvector_l2.csv"), 
        "chromadb": os.path.join(BASE_DIR, "results_chromadb_l2.csv") 
    },
    "cosine": {
        "qdrant": os.path.join(BASE_DIR, "results_qdrant_cosine.csv"),
        "pgvector": os.path.join(BASE_DIR, "results_pgvector_cosine.csv"),
        "chromadb": os.path.join(BASE_DIR, "results_chromadb_cosine.csv")
    }
}

def format_pvalue(p):
    """Formata o p-valor conforme a convenção da tese."""
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"

def run_statistical_analysis():
    print("="*70)
    print(" ANÁLISE ESTATÍSTICA: Teste t de Welch (p95) vs Qdrant (Baseline)")
    print("="*70)
    
    for metric in METRICS:
        print(f"\n--- MÉTRICA: {metric.upper()} ---")
        
        # 1. Carregar dados
        try:
            df_q = pd.read_csv(CSV_FILES[metric]["qdrant"])
            df_p = pd.read_csv(CSV_FILES[metric]["pgvector"])
            df_c = pd.read_csv(CSV_FILES[metric]["chromadb"])
        except FileNotFoundError as e:
            print(f"Erro: Ficheiro não encontrado. Verifica os caminhos no script:\n   {e}")
            continue
            
        results = []
        
        for scale in SCALES:
            # 2. Extrair os valores de p95_ms para as 10 réplicas nesta escala
            p95_q = df_q[df_q["Scale"] == scale]["p95_ms"].dropna().values
            p95_p = df_p[df_p["Scale"] == scale]["p95_ms"].dropna().values
            p95_c = df_c[df_c["Scale"] == scale]["p95_ms"].dropna().values
            
            if len(p95_q) < 2 or len(p95_p) < 2 or len(p95_c) < 2:
                print(f"  ⚠️ Escala {scale}: Dados insuficientes (necessárias pelo menos 2 réplicas).")
                continue
                
            # 3. Teste de Shapiro-Wilk (Normalidade)
            shapiro_q = stats.shapiro(p95_q)
            shapiro_p = stats.shapiro(p95_p)
            shapiro_c = stats.shapiro(p95_c)
            
            # 4. Teste t de Welch (comparando com Qdrant, equal_var=False)
            # pgvector vs Qdrant
            t_stat_p, p_val_p = stats.ttest_ind(p95_p, p95_q, equal_var=False)
            # ChromaDB vs Qdrant
            t_stat_c, p_val_c = stats.ttest_ind(p95_c, p95_q, equal_var=False)
            
            # 5. Teste de Mann-Whitney U (não paramétrico, como robustez extra)
            mw_p = stats.mannwhitneyu(p95_p, p95_q, alternative='two-sided')
            mw_c = stats.mannwhitneyu(p95_c, p95_q, alternative='two-sided')
            
            results.append({
                "Scale": f"{scale:,}",
                "Shapiro_Q": format_pvalue(shapiro_q.pvalue),
                "Shapiro_P": format_pvalue(shapiro_p.pvalue),
                "Shapiro_C": format_pvalue(shapiro_c.pvalue),
                "Welch_p_pgvector": format_pvalue(p_val_p),
                "Welch_p_chromadb": format_pvalue(p_val_c),
                "MW_p_pgvector": format_pvalue(mw_p.pvalue),
                "MW_p_chromadb": format_pvalue(mw_c.pvalue)
            })
            
        # 6. Imprimir resultados formatados prontos para a tese
        df_results = pd.DataFrame(results)
        print(f"\n✅ Resultados para Tabela 4.{'1' if metric == 'l2' else '2'} (Coluna 'p-valor'):")
        print("-" * 50)
        print(df_results[["Scale", "Welch_p_pgvector", "Welch_p_chromadb"]].to_string(index=False))
        print("-" * 50)
        
        # 7. Guardar em CSV
        out_file = os.path.join(BASE_DIR, f"analise_estatistica_{metric}.csv")
        df_results.to_csv(out_file, index=False)
        print(f"💾 Resultados detalhados (com todos os testes) guardados em:\n   {out_file}")

if __name__ == "__main__":
    # Verificar dependências
    try:
        import pandas
        import scipy
    except ImportError:
        print(" Erro: é preciso instalar as dependências necessárias executando: pip install pandas scipy")
        exit(1)
        
    run_statistical_analysis()
