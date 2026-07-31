# Sistema Computacional de Auditoria Textual

Projeto desenvolvido como Trabalho de Conclusão de Curso, com o objetivo de comparar documentos acadêmicos (PDF, DOCX ou TXT) e identificar o grau de similaridade entre eles.

O sistema recebe dois ou mais documentos e calcula o quanto eles se parecem entre si, combinando três formas diferentes de medir similaridade: comparação de vocabulário (TF-IDF com similaridade de cosseno), comparação estrutural (Jaccard sobre n-grams de caracteres) e comparação de significado (embeddings de sentenças). Cada uma dessas métricas captura um tipo diferente de semelhança, a ideia de usar as três juntas é justamente cobrir casos que uma métrica sozinha deixaria passar, como uma paráfrase bem feita, que muda o vocabulário mas mantém o mesmo conteúdo.

O resultado final é um score entre 0 e 1, que classifica o par de documentos em quatro níveis: baixa similaridade, similaridade moderada, similaridade elevada ou alto risco.

## Tecnologias utilizadas

- Python 3.11
- Streamlit, para a interface
- scikit-learn, para TF-IDF e similaridade de cosseno
- NLTK, para stopwords e stemming em português
- sentence-transformers, para os embeddings semânticos
- pdfplumber e python-docx, para extração de texto dos arquivos
- pytest, para os testes

## Como executar

Instalar as dependências:

```
pip install -r requirements.txt
```

Rodar a interface:

```
streamlit run app/interface.py
```
