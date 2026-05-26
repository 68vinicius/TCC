# Benchmarks 

Este documento descreve a estrutura experimental utilizada para avaliação do sistema de comparação textual baseado em modelos vetoriais.

O objetivo desta seção é formalizar o ambiente de testes, a metodologia de geração de dados e os critérios de avaliação adotados, garantindo reprodutibilidade e rigor científico.

---

# 1. Objetivo da Avaliação

O sistema foi projetado para analisar similaridade textual entre documentos por meio de:

- Vetorização TF-IDF
- Similaridade de Cosseno (orientação vetorial)
- Índice de Jaccard com n-grams (similaridade estrutural)

A avaliação experimental tem como objetivo verificar o comportamento do modelo frente a diferentes níveis de distorção textual.

---

# 2. Hipótese Experimental

A hipótese central do experimento é que:

> O aumento progressivo da distorção textual entre dois documentos resulta na redução monotônica das métricas de similaridade, tanto no espaço vetorial quanto estrutural.

---

# 3. Variáveis do Experimento

### 3.1 Variável Independente
- Nível de distorção textual aplicado ao documento

### 3.2 Variáveis Dependentes
- Similaridade de Cosseno (TF-IDF)
- Índice de Jaccard (n-grams)
- Tempo de processamento (latência do sistema)

### 3.3 Variáveis Controladas
- Documento base de referência
- Pipeline de normalização textual
- Modelo de vetorização (TF-IDF)
- Configuração do sistema de comparação

---

# 4. Geração dos Datasets Sintéticos

Os datasets utilizados foram gerados automaticamente a partir de um documento base (`base.txt`), através de um pipeline controlado de transformação textual.

## 4.1 Estratégia de Geração

O processo de geração segue cinco níveis progressivos de transformação:

- **Identical:** duplicação exata do documento original
- **Paraphrase Light:** substituições lexicais pontuais e pequenas alterações estruturais
- **Paraphrase Medium:** reorganização sintática com preservação parcial do significado
- **Paraphrase Heavy:** reestruturação profunda com preservação temática aproximada
- **Unrelated:** geração de texto semanticamente independente

---

## 4.2 Controle Experimental

Esse processo garante que:

- O documento base permanece constante
- Apenas o nível de distorção varia
- O ambiente experimental é reprodutível
- A variável independente é isolada corretamente

---

# 5. Estrutura dos Experimentos (Benchmarks)

Os experimentos são organizados em cinco cenários principais:

| Benchmark | Categoria | Nível de Distorção | Objetivo Experimental |
| :--- | :--- | :--- | :--- |
| 01 | Identical | Nulo | Estabelecer limite superior de similaridade |
| 02 | Paraphrase Light | Baixo | Avaliar sensibilidade a pequenas variações lexicais |
| 03 | Paraphrase Medium | Médio | Avaliar robustez a reestruturação sintática |
| 04 | Paraphrase Heavy | Alto | Avaliar degradação progressiva da similaridade |
| 05 | Unrelated | Máximo | Estabelecer baseline inferior (ausência de similaridade) |

---