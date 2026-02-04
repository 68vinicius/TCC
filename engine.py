import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calcular_similaridade(texto1, texto2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([texto1, texto2])
    
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

st.set_page_config(page_title="Analizador de Texto TCC", layout="centered")

st.title("Comparador de Similitude de Textos")
# st.markdown("Protótipo de Engenharia de Sistema para TCC")

col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("Documento A", type=['txt'])
with col2:
    file2 = st.file_uploader("Documento B", type=['txt'])

if file1 and file2:
    t1 = file1.read().decode("utf-8")
    t2 = file2.read().decode("utf-8")
    
    if st.button("Analisar Correlação"):
        score = calcular_similaridade(t1, t2)
        porcentagem = round(score * 100, 2)
        
        st.divider()
        st.subheader(f"Índice de Correlação: {porcentagem}%")
        st.progress(score)
        
        if score > 0.8:
            st.success("Alta correlação detectada.")
        elif score > 0.4:
            st.warning("Correlação moderada.")
        else:
            st.info("Baixa correlação entre os textos.")