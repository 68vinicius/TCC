import streamlit as st
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from engine.core import TextAnalyzer
from engine.processors import TextCleaner
from engine.handlers import DocumentFactory

# --- CONFIG UI ---
st.set_page_config(
    page_title="Comparação de Conteúdo de Textos", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- SETUP 
@st.cache_resource
def init_engine():
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    pt_stops = stopwords.words('portuguese')
    
    # termos ( stopwords acadêmicas)
    pt_stops.extend([
        'figura', 'tabela', 'capítulo', 'seção', 'página', 'autor', 'ano',
        'referências', 'bibliográficas', 'anexo', 'apêndice', 'introdução', 
        'conclusão', 'metodologia', 'resultados', 'discussão', 'et', 'al', 
        'disponível', 'acesso', 'citado', 'conforme', 'segundo', 'visto'
    ])
    
    vectorizer = TfidfVectorizer(
        stop_words=pt_stops, 
        ngram_range=(1, 2), 
        sublinear_tf=True
    )
    cleaner = TextCleaner()
    return TextAnalyzer(vectorizer, cleaner)

engine = init_engine()


# --- HEADER 
st.title("Sistema de Comparação de Conteúdo de Textos")
st.markdown("---")

# --- UPLOAD 
col_a, col_b = st.columns(2)
file_a = col_a.file_uploader("Documento de Referência (A)", type=['txt', 'pdf', 'docx'])
file_b = col_b.file_uploader("Documento de Comparação (B)", type=['txt', 'pdf', 'docx'])

if file_a and file_b:
    if st.button("Executar Análise", use_container_width=True):
        with st.spinner("Processando extração e análise vetorial..."):
            try:
                # 1. extração via factory 
                raw_a = DocumentFactory.get_text(file_a)
                raw_b = DocumentFactory.get_text(file_b)

                # 2. execução via Engine Modular
                res = engine.compare(raw_a, raw_b)
                
                if res["status"] == "success":
                    # --- DASHBOARD DE MÉTRICAS 
                    st.subheader("Resultados da Análise")
                    c1, c2, c3 = st.columns(3)
                    
                    c1.metric(
                        label="Similaridade de Cosseno (TF-IDF)", 
                        value=f"{res['cosseno']*100:.2f}%",
                        help="mede a orientação vetorial temática."
                    )
                    
                    c2.metric(
                        label="Índice de Jaccard (N-gram)", 
                        value=f"{res['jaccard']*100:.2f}%",
                        help="mede a sobreposição estrutural de caracteres."
                    )
                    
                    c3.metric(
                        label="Latência de Processamento", 
                        value=f"{res['tempo']:.4f}s",
                        help="tempo gasto pelo motor analítico."
                    )

                    st.progress(res['cosseno'], text=f"Score de Veracidade: {res['cosseno']*100:.1f}%")

                    # --- EVIDÊNCIAS 
                    st.markdown("### Evidências Técnicas")
                    tab1, tab2, tab3 = st.tabs(["Processamento Interno", "Logs de Auditoria", "Exportar Relatório"])
                    
                    with tab1:
                        st.caption("Amostra dos dados normalizados (primeiros 500 caracteres):")
                        txt_col1, txt_col2 = st.columns(2)
                        txt_col1.text_area("Texto A Saneado", res['t1_limpo'][:500], height=150)
                        txt_col2.text_area("Texto B Saneado", res['t2_limpo'][:500], height=150)

                    with tab2:
                        st.code(
                            f"STATUS: SUCCESS\n"
                            f"ENGINE_MODE: MODULAR_FACTORY\n"
                            f"LATENCY: {res['tempo']:.4f}s\n"
                            f"DOC_A_EXT: {file_a.name.split('.')[-1]}\n"
                            f"DOC_B_EXT: {file_b.name.split('.')[-1]}",
                            language="bash"
                        )
                        st.success("Execução validada via Injeção de Dependência.")

                    with tab3:
                        report_data = json.dumps(res, indent=4)
                        st.download_button(
                            label="Baixar Laudo Técnico (JSON)",
                            data=report_data,
                            file_name="laudo_auditoria_similaridade.json",
                            mime="application/json"
                        )
                else:
                    st.error(f"Erro no Motor: {res['message']}")
                    
            except Exception as e:
                st.error(f"Falha Crítica na Execução: {e}")

# --- FOOTER         
st.sidebar.markdown(f"""
**Versão:** v2.0  
**Status da Aplicação:** Online  
**Metodologia:** TF-IDF & N-Grams    
**Arquitetura:** Factory Pattern  
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