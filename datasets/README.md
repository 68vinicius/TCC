Os arquivos foram projetados para testar o comportamento do sistema frente a diferentes tipos de sobreposição textual, desde a total divergência até o plágio quase literal.

### Benchmark 01: Divergência Total (Linha de Base)
* **Arquivos:** `b1_vendas.txt` vs `b1_clima.txt`
* **Objetivo:** Validar o "piso" do sistema e a ausência de falsos-positivos entre temas sem correlação.
* **Comportamento Esperado:** Ambas as métricas devem apresentar valores próximos a **0%**, dado que não há interseção léxica ou semântica relevante.

### Benchmark 02: Paráfrase Técnica (O "Pulo do Gato")
* **Arquivos:** `b2_engenharia_orig.txt` vs `b2_engenharia_para.txt`
* **Objetivo:** Avaliar a capacidade do sistema em distinguir **Tema** de **Escrita**.
* **Comportamento Esperado:** * **Cosseno:** Elevado (ex: > 75%), pois os vetores apontam para a mesma direção temática (microserviços).
    * **Jaccard:** Moderado/Baixo (ex: < 55%), pois a estrutura das frases e os termos foram significativamente alterados.
* **Conclusão:** Este cenário prova a eficácia da hibridização de métricas proposta no TCC.

### Benchmark 03: Similaridade Crítica (Plágio/Cópia)
* **Arquivos:** `b3_governance_orig.txt` vs `b3_governance_copy.txt`
* **Objetivo:** Validar a detecção de cópias quase literais com alterações mínimas (sinônimos pontuais).
* **Comportamento Esperado:** Ambas as métricas devem ser extremamente altas (ex: > 90%), disparando o alerta máximo de auditoria.
