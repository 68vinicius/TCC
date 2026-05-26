import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
import logging

import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from engine.motor import TextAnalyzer 
from engine.processadores import TextCleaner 
from engine.extrator import DocumentFactory

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Sistema de Auditoria Textual",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GLOBAL CORPUS
# =========================================================

CORPUS_REFERENCE = [
    """
    processamento linguagem natural modelos vetoriais
    similaridade textual recuperação informação tfidf
    embeddings semânticos análise estrutural
    """
]


# =========================================================
# ENGINE INIT
# =========================================================

@st.cache_resource
def init_engine():

    logger.info("Inicializando motor analítico")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=10000
    )

    cleaner = TextCleaner()

    return TextAnalyzer(
        vectorizer=vectorizer,
        cleaner=cleaner,
        corpus_reference=CORPUS_REFERENCE
    )


engine = init_engine()


# =========================================================
# HEADER
# =========================================================

st.title("Sistema Computacional de Auditoria Textual")

st.markdown("""
Motor híbrido de análise textual baseado em:

- TF-IDF
- Similaridade de Cosseno
- Índice de Jaccard
- Sentence Embeddings
- Arquitetura Modular
- Strategy Pattern
""")

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Configuração Experimental")

show_clean_text = st.sidebar.checkbox(
    "Exibir textos normalizados",
    value=True
)

show_embeddings = st.sidebar.checkbox(
    "Exibir score semântico",
    value=True
)

show_telemetry = st.sidebar.checkbox(
    "Exibir telemetria",
    value=True
)


st.sidebar.divider()

st.sidebar.markdown("""
### Arquitetura

- Factory Method
- Strategy Pattern
- Modular Pipeline
- Semantic Embeddings
- Benchmark Driven
""")

st.sidebar.divider()

st.sidebar.markdown("""
### Autor

Vinicius de Oliveira Lima
""")

st.sidebar.markdown("""
### Repositório

https://github.com/68vinicius/TCC
""")


# =========================================================
# FILE UPLOAD
# =========================================================

col_a, col_b = st.columns(2)

file_a = col_a.file_uploader(
    "Documento de Referência",
    type=["txt", "pdf", "docx"]
)

file_b = col_b.file_uploader(
    "Documento de Comparação",
    type=["txt", "pdf", "docx"]
)


# =========================================================
# EXECUTION
# =========================================================

if file_a and file_b:

    execute = st.button(
        "Executar Auditoria Analítica",
        use_container_width=True
    )

    if execute:

        with st.spinner("Executando pipeline analítico..."):

            try:

                # =====================================
                # EXTRACTION
                # =====================================

                raw_a = DocumentFactory.get_text(file_a)

                raw_b = DocumentFactory.get_text(file_b)

                # =====================================
                # ANALYSIS
                # =====================================

                result = engine.compare(
                    raw_a,
                    raw_b
                )

                if result["status"] != "success":

                    st.error(result["message"])

                    st.stop()

                # =====================================
                # MAIN METRICS
                # =====================================

                st.subheader("Resultados Analíticos")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "TF-IDF + Cosseno",
                    f"{result['cosseno']*100:.2f}%"
                )

                c2.metric(
                    "Jaccard N-Gram",
                    f"{result['jaccard']*100:.2f}%"
                )

                c3.metric(
                    "Embedding Semântico",
                    f"{result['embedding']*100:.2f}%"
                )

                c4.metric(
                    "Score Híbrido",
                    f"{result['score_final']*100:.2f}%"
                )

                # =====================================
                # CLASSIFICATION
                # =====================================

                st.divider()

                classification = result["classificacao"]

                if classification == "ALTO RISCO":

                    st.error(
                        f"CLASSIFICAÇÃO: {classification}"
                    )

                elif classification == "SIMILARIDADE ELEVADA":

                    st.warning(
                        f"CLASSIFICAÇÃO: {classification}"
                    )

                elif classification == "SIMILARIDADE MODERADA":

                    st.info(
                        f"CLASSIFICAÇÃO: {classification}"
                    )

                else:

                    st.success(
                        f"CLASSIFICAÇÃO: {classification}"
                    )

                # =====================================
                # SCORE VISUALIZATION
                # =====================================

                st.progress(
                    result["score_final"],
                    text=f"Score Final: {result['score_final']*100:.2f}%"
                )

                # =====================================
                # COMPARATIVE TABLE
                # =====================================

                st.divider()

                st.subheader("Matriz de Similaridade")

                df = pd.DataFrame({

                    "Métrica": [
                        "Cosseno",
                        "Jaccard",
                        "Embedding",
                        "Final"
                    ],

                    "Score": [

                        result["cosseno"],

                        result["jaccard"],

                        result["embedding"],

                        result["score_final"]
                    ]
                })

                st.dataframe(
                    df,
                    use_container_width=True
                )

                # =====================================
                # CLEAN TEXT
                # =====================================

                if show_clean_text:

                    st.divider()

                    st.subheader("Textos Saneados")

                    txt1, txt2 = st.columns(2)

                    txt1.text_area(
                        "Documento A",
                        result["t1_limpo"][:3000],
                        height=300
                    )

                    txt2.text_area(
                        "Documento B",
                        result["t2_limpo"][:3000],
                        height=300
                    )

                # =====================================
                # TELEMETRY
                # =====================================

                if show_telemetry:

                    st.divider()

                    st.subheader("Telemetria Computacional")

                    telemetry = pd.DataFrame({

                        "Métrica": [
                            "Latência",
                            "Consumo RAM"
                        ],

                        "Valor": [
                            f"{result['tempo']:.4f}s",
                            f"{result['ram_mb']} MB"
                        ]
                    })

                    st.table(telemetry)

                # =====================================
                # TECHNICAL LOGS
                # =====================================

                st.divider()

                st.subheader("Logs Técnicos")

                st.code(

                    f"""
STATUS: SUCCESS
ENGINE: HYBRID_ANALYTICAL_ENGINE
VECTOR_MODEL: TFIDF
SEMANTIC_MODEL: MiniLM-L12-v2
PIPELINE: MODULAR_PIPELINE
ARCHITECTURE: STRATEGY_PATTERN
DOC_A: {file_a.name}
DOC_B: {file_b.name}
LATENCY: {result['tempo']:.4f}s
RAM_USAGE: {result['ram_mb']} MB
CLASSIFICATION: {result['classificacao']}
                    """,

                    language="bash"
                )

                # =====================================
                # EXPORT REPORT
                # =====================================

                st.divider()

                st.subheader("Exportação")

                report_json = json.dumps(
                    result,
                    indent=4,
                    ensure_ascii=False
                )

                st.download_button(
                    label="Baixar Laudo Técnico",
                    data=report_json,
                    file_name="laudo_analitico.json",
                    mime="application/json",
                    use_container_width=True
                )

            except Exception as e:

                logger.exception("Falha crítica")

                st.error(
                    f"Falha crítica na execução: {str(e)}"
                )
