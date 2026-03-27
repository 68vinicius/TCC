Este repositório contém o protótipo funcional de um **Sistema de Comparação de Conteúdo de Textos**, desenvolvido como parte integrante do meu Trabalho de Conclusão de Curso (TCC). 

O sistema utiliza técnicas de **Processamento de Linguagem Natural (PLN)** para calcular o nível de correlação entre dois documentos de texto.

---

## 🚀 Tecnologias Utilizadas

* **Python 3.11**: Linguagem base do projeto.
* **Streamlit**: Framework para a criação da interface web.
* **Scikit-Learn**: Biblioteca para processamento de dados e cálculo matemático.
    * **TF-IDF Vectorizer**: Para conversão de texto em vetores numéricos.
    * **Cosine Similarity**: Para o cálculo da distância entre os vetores.

---

## Estrutura do Projeto

```text
TCC/
├── .github/              # Configurações do repositório
├── datasets/             # Base de dados para validação
├── engine.py             # Script principal com a lógica de backend e interface
├── requirements.txt      # Gerenciador de dependências do projeto
└── README.md             # Documentação técnica
```

---

## Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
