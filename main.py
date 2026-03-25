import streamlit as st
import json
from pypdf import PdfReader
from engine import TextAnalyzer

# --- INFRAESTRUTURA DE EXTRAÇÃO ---
def extrair_texto(arquivo):
    """extrai texto de pdf ou txt com tratamento de exceção."""
    try:
        if arquivo.type == "application/pdf":
            reader = PdfReader(arquivo)
            return "".join([p.extract_text() for p in reader.pages if p.extract_text()])
        return arquivo.read().decode("utf-8")
    except Exception as e:
        st.error(f"falha na extração de dados: {e}")
        return None

# --- CONFIG UI ---
st.set_page_config(
    page_title="Sistema de Comparação de Conteúdo de Textos", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# inicialização persistente do motor 
@st.cache_resource
def load_engine():
    return TextAnalyzer()

engine = load_engine()

# --- HEADER ---
st.title("Sistema de Comparação de Conteúdo de Textos")
st.markdown("---")

# --- UPLOAD ---
col_a, col_b = st.columns(2)
file_a = col_a.file_uploader("Documento de Referência (A)", type=['txt', 'pdf'])
file_b = col_b.file_uploader("Documento de Comparação (B)", type=['txt', 'pdf'])

if file_a and file_b:
    if st.button("🚀 Executar Análise Multimetodológica", use_container_width=True):
        with st.spinner("processando vetores e calculando distâncias..."):
            raw_a = extrair_texto(file_a)
            raw_b = extrair_texto(file_b)

            if raw_a and raw_b:
                # execução do motor com telemetria
                res = engine.get_similarities(raw_a, raw_b)
                
                # --- DASHBOARD DE MÉTRICAS (TELEMETRIA) ---
                st.subheader("Resultados da Análise")
                c1, c2, c3 = st.columns(3)
                
                # métrica de cosseno (semântica)
                c1.metric(
                    label="Similaridade de Cosseno (TF-IDF)", 
                    value=f"{round(res['cosseno']*100, 2)}%",
                    help="mede a orientação vetorial temática. ideal para identificar paráfrases."
                )
                
                # métrica de jaccard (estrutural)
                c2.metric(
                    label="Índice de Jaccard (N-gram)", 
                    value=f"{round(res['jaccard']*100, 2)}%",
                    help="mede a sobreposição estrutural de caracteres. detecta cópias literais e typos."
                )
                
                # telemetria de performance
                c3.metric(
                    label="Latência de Processamento", 
                    value=f"{round(res['tempo_execucao'], 4)}s",
                    help="tempo gasto pelo motor analítico para saneamento e cálculo."
                )

                st.progress(res['cosseno'], text=f"Score de Veracidade: {round(res['cosseno']*100, 1)}%")

                # --- DOCUMENTAÇÃO E EXPORTAÇÃO ---
                st.markdown("### Evidências Técnicas")
                
                tab1, tab2, tab3 = st.tabs(["Processamento Interno", "Logs de Auditoria", "Exportar Relatório"])
                
                with tab1:
                    st.caption("amostra dos dados normalizados (primeiros 500 caracteres):")
                    txt_col1, txt_col2 = st.columns(2)
                    txt_col1.text_area("Texto A Saneado", res['t1_limpo'][:500], height=150)
                    txt_col2.text_area("Texto B Saneado", res['t2_limpo'][:500], height=150)

                with tab2:
                    st.code(
                        f"TIMESTAMP: 2026-03-24\n"
                        f"STATUS: {res['status'].upper()}\n"
                        f"ENGINE_LATENCY: {res['tempo_execucao']:.4f}s\n"
                        f"ALGORITHM_1: TF-IDF + COSINE_SIMILARITY\n"
                        f"ALGORITHM_2: JACCARD_CHARACTER_NGRAMS (n=3)",
                        language="bash"
                    )
                    st.success("fluxo de execução validado sem exceções críticas.")

                with tab3:
                    # exportação em json 
                    report_data = json.dumps(res, indent=4)
                    st.download_button(
                        label="Baixar Laudo Técnico (JSON)",
                        data=report_data,
                        file_name="laudo_auditoria_similaridade.json",
                        mime="application/json"
                    )
                    st.info("o laudo em JSON permite a integração desses resultados com outros sistemas.")
            else:
                st.error("não foi possível extrair conteúdo válido dos arquivos.")

# --- FOOTER ST ---
st.sidebar.markdown(f"""
**Versão:** v2.0  
**Status da Aplicação:** Online  
**Metodologia:** TF-IDF & N-Grams         
<br><br>               
<a href="https://www.linkedin.com/in/vinicius-o/" target="_blank"
style="display:inline-block;padding:8px 12px;background-color:#0A66C2;color:white;
text-decoration:none;border-radius:6px;">
Autor do Projeto
</a> 
<a href="https://github.com/68vinicius/TCC" target="_blank"
style="display:inline-block;padding:8px 12px;background-color:#0A66C2;color:white;
text-decoration:none;border-radius:6px;">
Projeto no GitHub
</a>
""", unsafe_allow_html=True)