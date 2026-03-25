import re
import string
import nltk
import logging
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÃO DE LOGGING ESTRUTURADO ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [TEXT-ENGINE] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO DE AMBIENTE ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("baixando stopwords do nltk...")
    nltk.download('stopwords')

class TextAnalyzer:
    def __init__(self):
        """
        inicializa o motor de análise com parâmetros de alta performance.
        configurado para lidar com corpus reduzido e documentos acadêmicos.
        """
        try:
            self.pt_stopwords = stopwords.words('portuguese')
            
            # configuração do vetorizador com refinamento de outliers
            self.vectorizer = TfidfVectorizer(
                stop_words=self.pt_stopwords, 
                ngram_range=(1, 2),   # captura contexto de bigramas
                max_df=0.9,           # ignora termos que aparecem em mais de 90% do texto
                min_df=1,             # requisito mínimo de ocorrência
                smooth_idf=True,      # evita divisões por zero
                sublinear_tf=True     # aplica escala logarítmica (1 + log(tf))
            )
            logger.info("engine inicializado: vetorizador tf-idf configurado.")
        except Exception as e:
            logger.error(f"erro ao inicializar o motor: {str(e)}")
            raise

    def limpar_texto(self, texto):
        """
        pipeline de saneamento de dados.
        converte para minúsculo, remove pontuação, números e espaços excedentes.
        """
        if not texto:
            return ""
            
        try:
            texto = texto.lower()
            texto = re.sub(f"[{re.escape(string.punctuation)}]", " ", texto)
            texto = re.sub(r'\d+', '', texto)
            limpo = " ".join(texto.split())
            return limpo
        except Exception as e:
            logger.warning(f"erro durante o saneamento do texto: {str(e)}")
            return ""

    def _get_char_ngrams(self, text, n=3):
        """gera n-grams de caracteres para análise estrutural jaccard."""
        return set([text[i:i+n] for i in range(len(text)-n+1)])

    def _null_response(self, t1, t2, duration=0.0):
        """retorna estrutura de dados padrão para casos de erro ou vazio."""
        return {
            "cosseno": 0.0, 
            "jaccard": 0.0, 
            "t1_limpo": t1, 
            "t2_limpo": t2,
            "tempo_execucao": duration,
            "status": "vazio_ou_erro"
        }

    def get_similarities(self, t1_raw, t2_raw):
        """
        executa a comparação multimetodológica com telemetria.
        calcula similaridade de cosseno (semântica) e índice de jaccard (estrutural).
        """
        start_time = time.perf_counter()
        logger.info("iniciando novo ciclo de análise de similaridade.")

        try:
            # 1. saneamento
            t1 = self.limpar_texto(t1_raw)
            t2 = self.limpar_texto(t2_raw)

            if not t1 or not t2:
                logger.warning("análise abortada: um ou ambos os documentos estão vazios após limpeza.")
                return self._null_response(t1, t2)

            # 2. cálculo de cosseno (tf-idf)
            # transformamos os textos em matriz vetorial
            tfidf_matrix = self.vectorizer.fit_transform([t1, t2])
            cos_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

            # 3. cálculo de jaccard (caracteres)
            set1 = self._get_char_ngrams(t1)
            set2 = self._get_char_ngrams(t2)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jac_sim = float(intersection / union if union != 0 else 0)

            end_time = time.perf_counter()
            duration = end_time - start_time

            logger.info(f"análise concluída com sucesso em {duration:.4f}s.")
            logger.info(f"resultados - cosseno: {cos_sim:.4f} | jaccard: {jac_sim:.4f}")

            return {
                "cosseno": cos_sim,
                "jaccard": jac_sim,
                "t1_limpo": t1,
                "t2_limpo": t2,
                "tempo_execucao": duration,
                "status": "sucesso"
            }

        except Exception as e:
            logger.error(f"falha catastrófica durante o cálculo de similaridade: {str(e)}", exc_info=True)
            return self._null_response("", "", 0.0)