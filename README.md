Este repositório contém um **Sistema de Comparação de Conteúdo de Textos**, desenvolvido como parte integrante do meu Trabalho de Conclusão de Curso (TCC). 

O sistema automatiza a verificação de integridade e originalidade de documentos acadêmicos através de modelos vetoriais.

---

## Fundamentação

* **2.1 PLN e Normalização:** Compreende processos como *lowercase*, remoção de *stopwords* e tokenização para reduzir o ruído nos modelos vetoriais.
* **2.2 Modelo Vetorial e TF-IDF:** O TF-IDF permite identificar palavras-chave com alto poder discriminatório, mitigando a influência de termos onipresentes.
* **2.3 Métricas de Similaridade:**
    * **Similitude de Cosseno:** Foca na orientação dos vetores (tema), sendo ideal para comparar textos de tamanhos diferentes.
    * **Índice de Jaccard e *N-grams*:** Foca na forma e na escrita, identificando variações ortográficas e paráfrases próximas.
* **2.4 Padrões de Projeto:** O uso do *Factory Method* garante o desacoplamento entre o formato de origem (PDF, DOCX, TXT) e o motor de processamento.  
---

## Tecnologias e Arquitetura

O projeto foi construído seguindo princípios de **Clean Code** e **Design Patterns**:

* **Linguagem:** Python 3.11+
* **Interface:** Streamlit (Dashboard de alta usabilidade)
* **Engine:** Scikit-Learn & NLTK
* **Design Pattern:** *Factory Method* para ingestão de múltiplos formatos (PDF, DOCX, TXT)

---

## Estrutura do Repositório

```text
TCC/
├── datasets/            # Amostras (.txt) 
├── engine/              # sistema analítico
│   ├── core.py          # vetorização e métricas matemáticas
│   ├── handlers.py      # Factory Pattern para múltiplos formatos
│   └── processors.py    # Pipeline de normalização
├── .gitignore           # ignorados pelo versionamento
├── main.py              # Streamlit Dashboard
├── packages.txt         # Gerenciador de dependências 
├── requirements.txt     # Dependências Python 
└── README.md            # Documentação do projeto
```

---

**Tabela 1 – Resultados de Similaridade por Caso de Teste**
| Caso de Teste | Descrição do Cenário | Similitude de Cosseno | Índice de Jaccard |
| :--- | :--- | :--- | :--- |
| Teste 01 | Documentos 100% Idênticos | 100,00% | 100,00% |
| Teste 02 | Paráfrase Técnica | 78,42% | 54,21% |
| Teste 03 | Temas Distintos | 5,20% | 2,15% |
*Fonte: Autoria  (2026).*

---

## Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
