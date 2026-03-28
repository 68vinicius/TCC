# Benchmarks de Performance

A arquitetura do sistema foi submetida a uma bateria de testes controlados (**Benchmarks**) para validar a sensibilidade dos algoritmos de similaridade híbrida. O objetivo é mapear o comportamento do motor analítico frente a diferentes níveis de sobreposição textual e semântica.

---

### Resumo Executivo dos Resultados

| Benchmark | Tipo de Teste | Documento A | Documento B | Cosseno (Tema) | Jaccard (Estrutura) | Veredito Técnico |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | **Divergência Total** | `b1_clima.txt` | `b1_vendas.txt` | **0.00%** | **15.81%** | Baseline: Ortogonalidade Vetorial |
| **02** | **Paráfrase Técnica** | `b2_orig.txt` | `b2_para.txt` | **9.73%** | **29.17%** | Detecção de Âncoras Temáticas |
| **03** | **Similaridade Crítica** | `b3_orig.txt` | `b3_copy.txt` | **71.91%** | **85.62%** | **Alerta Crítico de Auditoria** |

---

### Detalhamento dos Cenários

#### Benchmark 01: Divergência Temática
![Evidência do Benchmark 01](https://raw.githubusercontent.com/68vinicius/TCC/main/datasets/benchmark_01/b1_screenshot.jpg)
* **Objetivo:** Estabelecer a linha de base (piso) do sistema e validar a ausência de falsos-positivos em domínios ortogonais.
* **Análise:** As métricas convergem para valores mínimos, confirmando que o motor de análise não gera associações indevidas entre temas distintos (Climatologia vs. Vendas).
* **Conclusão:** Alta especificidade confirmada em temas ortogonais, garantindo a integridade analítica do motor.
* **Status:** ✅ Validado.

#### Benchmark 02: Paráfrase Técnica 
![Evidência do Benchmark 02](https://raw.githubusercontent.com/68vinicius/TCC/main/datasets/benchmark_02/b2_screenshot.jpg)
* **Objetivo:** Avaliar a capacidade do sistema em distinguir **Conceito Temático** de **Estrutura Sintática**.
* **Análise:** O incremento no Cosseno em relação ao B1 indica a identificação de tokens técnicos comuns (microserviços, software, escalabilidade), enquanto o Jaccard moderado reflete a reestruturação profunda das sentenças.
* **Conclusão:** Valida a tese de **hibridização de métricas**, demonstrando que a análise isolada de apenas um algoritmo seria insuficiente para um diagnóstico de similaridade.
* **Status:** ✅ Validado.

#### Benchmark 03: Similaridade Crítica 
![Evidência do Benchmark 03](https://raw.githubusercontent.com/68vinicius/TCC/main/datasets/benchmark_03/b3_screenshot.jpg)
* **Objetivo:** Validar a detecção de cópias quase literais com tentativas de ofuscação (substituição de sinônimos e siglas).
* **Análise:** Ambos os índices apresentam alta correlação e crescimento simultâneo, demonstrando que o sistema é resiliente a alterações superficiais (ex: "BI" por "Business Intelligence") em textos de maior volume.
* **Conclusão:** Identificação de plágio estrutural com alta confiabilidade.
* **Status:** ✅ Validado.

---

### Estrutura de Evidências de Auditoria
Para garantir a reprodutibilidade dos resultados, cada teste realizado possui um **Audit Trail** (Trilha de Auditoria) na pasta `datasets/`, composto por:

* **Documentos (.txt):** O par de arquivos brutos submetidos ao processamento.
* **Snapshot (.jpg):** Evidência visual do dashboard, capturando o estado da UI e os scores finais.
* **Laudo Técnico (.json):** Metadados brutos, incluindo vetores de similaridade e latência de processamento.

---
*Este projeto faz parte dos testes de validação para o Trabalho de Conclusão de Curso em Engenharia de Software.*
