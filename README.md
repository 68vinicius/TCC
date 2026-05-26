# Sistema Computacional de Auditoria Textual

Este repositório contém um **Sistema Computacional de Auditoria Textual**, desenvolvido como parte integrante do Trabalho de Conclusão de Curso (TCC).

O sistema automatiza a análise de similaridade textual entre documentos acadêmicos utilizando uma abordagem híbrida baseada em:

* Processamento de Linguagem Natural (PLN)
* Modelos Vetoriais TF-IDF
* Similaridade de Cosseno
* Índice de Jaccard
* Sentence Embeddings
* Pipeline Modular de Processamento
* Estratégias Analíticas Híbridas

O objetivo central é identificar níveis de similaridade textual, paráfrases técnicas e potenciais indícios de reaproveitamento de conteúdo acadêmico.

---

# Estrutura do Projeto

```text 
TCC/
├── dataset/
│   ├── benchmark_01/
│   ├── benchmark_02/
│   ├── benchmark_03/
│   ├── benchmark_04/
│   └── benchmark_05/
├── src/
│   ├── app/
│   │   └── main.py                 
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── extrator.py             
│   │   ├── motor.py                
│   │   └── processadores.py        
│   └── experimentos/
│       └── executar.py             
├── .gitignore
├── packages.txt
├── requirements.txt
└── README.md                      
```

---

# Arquitetura do Sistema

O projeto foi desenvolvido seguindo princípios de:

* Clean Code
* Modularização
* Design Patterns
* Desacoplamento Arquitetural
* Reprodutibilidade Experimental

---

## Design Patterns Utilizados

### Factory Method

Utilizado na ingestão de múltiplos formatos:

* PDF
* DOCX
* TXT

---

### Strategy Pattern

Utilizado na implementação desacoplada das métricas de similaridade:

* Similaridade de Cosseno
* Índice de Jaccard
* Embeddings Semânticos

---

# Fundamentação Teórica

## 1. Processamento de Linguagem Natural (PLN)

O sistema utiliza técnicas clássicas de PLN para normalização textual:

* Lowercase
* Remoção de URLs
* Remoção de pontuação
* Remoção de números
* Stopwords em português
* Stemming (RSLP Stemmer)

Essas etapas reduzem ruídos linguísticos e aumentam a capacidade discriminatória dos modelos vetoriais.

---

## 2. Modelo Vetorial TF-IDF

O TF-IDF (*Term Frequency – Inverse Document Frequency*) é utilizado para representar documentos em espaço vetorial de alta dimensionalidade.

O modelo privilegia termos com maior poder semântico e reduz o impacto de palavras excessivamente frequentes.

---

## 3. Métricas de Similaridade

### Similaridade de Cosseno

Mede a proximidade angular entre vetores TF-IDF.

Indicada para:

* comparação temática
* análise semântica superficial
* documentos de tamanhos diferentes

---

### Índice de Jaccard com N-Grams

Baseado em interseção e união de conjuntos de caracteres.

Indicado para:

* detecção de paráfrases próximas
* padrões ortográficos
* reaproveitamento estrutural

---

### Sentence Embeddings

O sistema utiliza:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

para captura de similaridade semântica profunda.

Essa abordagem permite identificar:

* equivalência contextual
* paráfrases complexas
* semântica implícita

---

## 4. Ensemble Híbrido

O score final combina múltiplas métricas:

```text
Score Final =
(0.4 × Cosseno)
+ (0.2 × Jaccard)
+ (0.4 × Embedding)
```

Essa estratégia híbrida aumenta robustez analítica e reduz falsos positivos.

---

# Tecnologias Utilizadas

| Tecnologia           | Finalidade            |
| -------------------- | --------------------- |
| Python 3.11+         | Linguagem principal   |
| Streamlit            | Dashboard interativo  |
| Scikit-Learn         | Vetorização TF-IDF    |
| SentenceTransformers | Embeddings semânticos |
| NLTK                 | PLN e stemming        |
| Pandas               | Manipulação analítica |
| pdfplumber           | Extração PDF          |
| python-docx          | Extração DOCX         |

---

# Funcionalidades

## Dashboard Interativo

O sistema fornece interface web via Streamlit contendo:

* Upload de documentos
* Métricas analíticas
* Classificação automática
* Visualização de scores
* Logs técnicos
* Telemetria computacional
* Exportação JSON

---

## Pipeline Analítico

Fluxo completo:

```text
Upload
   ↓
Extração textual
   ↓
Normalização
   ↓
Vetorização TF-IDF
   ↓
Cálculo de Similaridade
   ↓
Embeddings Semânticos
   ↓
Score Híbrido
   ↓
Classificação
```

---

# Benchmark Experimental

O projeto possui ambiente experimental automatizado para execução de benchmarks em larga escala.

Os experimentos permitem:

* validação científica
* análise estatística
* comparação entre métricas
* geração de datasets analíticos
* exportação CSV/JSON

---

# Classificação Analítica

| Score Final | Classificação         |
| ----------- | --------------------- |
| ≥ 0.85      | ALTO RISCO            |
| ≥ 0.65      | SIMILARIDADE ELEVADA  |
| ≥ 0.40      | SIMILARIDADE MODERADA |
| < 0.40      | BAIXA SIMILARIDADE    |

---

## Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
   ```