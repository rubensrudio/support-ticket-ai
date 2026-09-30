# Decisões — Intelligent Support Ticket Classifier (initial_feature)

Etapa: gate de lacunas do spec-writer (`/generate-plan`, Passo 2).

## Decisões do humano

### LAC-01 — Fonte do dataset?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) dataset público real de tickets rotulados (ex.: Kaggle "Customer IT Support Ticket"), com as filas mapeadas para as categorias B) dataset sintético gerado por templates ou LLM C) público real + aumento sintético nas categorias raras
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-02 — Conjunto fechado de categorias?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) access, infrastructure, billing, bug + other B) categorias ou filas nativas do dataset C) lista definida no treino, e a API devolve o que o modelo conhece
- **Recomendação do spec-writer**: A
- **Escolha**: A — `access`, `infrastructure`, `billing`, `bug`, `other`
- **Observações do humano**: —

### LAC-03 — Escala de prioridade?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) low, medium, high B) low, medium, high, critical C) escala nativa do dataset
- **Recomendação do spec-writer**: A
- **Escolha**: A — `low`, `medium`, `high`
- **Observações do humano**: —

### LAC-04 — Idioma dos tickets?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) só inglês B) só português C) multilíngue (inglês + português)
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-06 — Tratamento de baixa confiança?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) devolve a classe prevista + `needs_review=true` abaixo de um limiar configurável (default 0.6) B) troca a categoria por "unknown" (triagem manual) C) sem tratamento, só devolve a confiança
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-07 — Como registrar a classificação correta?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) `/predict` devolve `prediction_id` e grava entrada+previsão; `POST /feedback` recebe `prediction_id` + rótulos corretos B) `POST /feedback` recebe título, descrição e rótulos, sem vínculo com previsão C) aceita as duas formas
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-17 — Segundo feedback para a mesma previsão?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) o mais recente substitui o anterior B) rejeita com 409 C) guarda todos e o retreino usa o mais recente
- **Recomendação do spec-writer**: C
- **Escolha**: C
- **Observações do humano**: —

### LAC-13 — `/predict` devolve também a distribuição de probabilidades?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) só classe + confiança para categoria e prioridade B) A + top-3 categorias com probabilidade C) A + distribuição completa
- **Recomendação do spec-writer**: B
- **Escolha**: B
- **Observações do humano**: —

### LAC-08 — Autenticação da API?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) nenhuma (demo local, risco documentado) B) chave de API em todos os endpoints, exceto health C) chave de API só no `/feedback`; predict e health abertos
- **Recomendação do spec-writer**: C
- **Escolha**: C
- **Observações do humano**: —

### LAC-09 — Dado pessoal no texto dos tickets?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) mascarar e-mail, telefone, URL e nº de documento no pré-processamento (treino e inferência) e persistir só o texto mascarado B) não mascarar e documentar que os dados são públicos/sintéticos C) não persistir texto vindo da API, só rótulos e hash
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-10 — Retreinamento e promoção de versão?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) disparo manual; treino original + feedbacks; avaliação no test set congelado; promoção manual B) disparo manual; promoção automática só se Macro F1 da nova versão >= atual no test set congelado C) disparo automático a cada N feedbacks + promoção automática
- **Recomendação do spec-writer**: B
- **Escolha**: B
- **Observações do humano**: —

### LAC-11 — Qual modelo a API serve?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) sempre o Transformer B) o melhor Macro F1 de categoria na validação (baseline ou Transformer) C) a versão marcada como promovida no registro de versões
- **Recomendação do spec-writer**: C
- **Escolha**: C — versão inicial promovida é o Transformer
- **Observações do humano**: —

### LAC-12 — Limites de entrada?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) title obrigatório 1-200 chars; description obrigatória 1-5000 chars; excedente de tokens truncado B) só title obrigatório, description opcional C) sem limite
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-16 — Devolver equipe de destino além da categoria?
- **Data**: 2026-09-30
- **Opções apresentadas**: A) não; só categoria, roteamento para equipe fora de escopo B) sim, com mapeamento categoria->equipe configurável
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

## Decisões do humano — escrita do spec (spec-writer, `Modo: ESCREVER`)

Etapa: lacunas surgidas ao escrever os critérios do `spec.md` (Passo 2, 2ª passada).

### LAC-18 — Padrões de PII cobertos pelo Mascaramento
- **Data**: 2026-09-30
- **Opções apresentadas**: A) e-mail→`[EMAIL]`, URL→`[URL]`, telefone→`[PHONE]`, sequência de 6+ dígitos→`[NUMBER]` B) só e-mail e telefone C) A + nomes de pessoas por reconhecimento de entidades
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-19 — Retenção e exclusão de Previsões e Feedbacks
- **Data**: 2026-09-30
- **Opções apresentadas**: A) retenção indefinida, documentada no README B) Previsões sem Feedback apagadas após 90 dias; Previsões com Feedback mantidas como dado de treino C) fora de escopo
- **Recomendação do spec-writer**: B
- **Escolha**: B — cria critério novo P2 (expurgo)
- **Observações do humano**: —

### LAC-20 — API sobe sem Chave de API configurada para o `/feedback`
- **Data**: 2026-09-30
- **Opções apresentadas**: A) API sobe; `/feedback` recusa toda requisição com 401 (falha fechada); `/predict` e `/health` funcionam B) API não sobe
- **Recomendação do spec-writer**: A
- **Escolha**: A
- **Observações do humano**: —

### LAC-21..LAC-30 — Pacote de recomendações (aceito em bloco)
- **Data**: 2026-09-30
- **Opções apresentadas**: A) aceitar todas as recomendações B) revisar uma a uma
- **Recomendação do spec-writer**: A
- **Escolha**: A — aplicadas as recomendações abaixo:
  - **LAC-21**: sem Versão promovida ao iniciar → API sobe; `/health` responde 503 com status `"unavailable"`; `/predict` responde 503 `"Model not available. Try again later."`
  - **LAC-22**: `needs_review=true` quando a confiança de category OU a de priority fica abaixo do limiar.
  - **LAC-23**: gate de promoção = Macro F1 de category >= atual E Macro F1 de priority >= atual, no Test set congelado.
  - **LAC-24**: versão recém-promovida passa a ser servida no próximo reinício da API.
  - **LAC-25**: retreino sem nenhum Feedback aborta sem criar versão (`"No feedback records available. Retraining aborted."`).
  - **LAC-26**: cada category precisa de >= 100 exemplos no train; senão a preparação falha e informa a categoria.
  - **LAC-27**: duplicatas exatas (title+description normalizados) removidas antes da divisão.
  - **LAC-28**: reprodutibilidade — baseline com métricas idênticas; Transformer com diferença de Macro F1 <= 1 pp.
  - **LAC-29**: falha ao gravar a Previsão → `/predict` responde 503 sem devolver previsão.
  - **LAC-30**: retreino disparado com outro em execução é recusado (`"A retraining run is already in progress."`).
- **Observações do humano**: —

## Decisões do humano — plano (plan-architect)

Etapa: `/generate-plan`, Passo 3.

### LAC-32 — Conversão dos rótulos do dataset em `category`
- **Data**: 2026-09-30
- **Contexto**: dataset HF `Tobi-Bueck/customer-support-tickets`, arquivo `dataset-tickets-multi-lang-4-20k.csv` (revisão `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`), 11.923 linhas em inglês, 10 valores de `queue` e tags livres `tag_1..tag_8`. Nenhuma `queue` corresponde a `access` nem a `bug`. Licença CC BY-NC 4.0; o card indica possível origem sintética (registrado como risco no README).
- **Opções apresentadas**: A) regra composta `queue` + tags em TOML versionado (`[priority]`, `[category.queue]` cobrindo todo valor de origem, `[[category.rules]]` ordenadas, primeira que casar vence; tags fora das regras ignoradas; DATA-91 vale para `queue` e `priority`) B) como A, mas toda tag mapeada na tabela C) mapear só por `queue`
- **Recomendação do plan-architect**: A
- **Escolha**: A — contagens medidas: other 4770, bug 3137, infrastructure 1343, billing 1194, access 446
- **Observações do humano**: —

## Premissas assumidas (lacunas não bloqueantes)

### LAC-31 — Textos exatos das mensagens
- **Premissa**: textos em inglês conforme seção 9 do `spec.md`.
- **Reversibilidade**: alta
- **Onde impacta**: `spec.md` seção 9, mensagens de erro da API e do retreino
- **Jev**: não aplicado (regra de `lacunas.md`: texto de mensagem é não bloqueante)

### LAC-05 — Como medir "superar claramente o baseline"?
- **Premissa**: Macro F1 de categoria do Transformer >= baseline + 5 pp no test set, e Macro F1 de prioridade não inferior ao baseline.
- **Reversibilidade**: alta
- **Onde impacta**: spec.md (critério de sucesso), relatório comparativo de métricas
- **Jev**: sem_criterio=0.64 (dúvida); demais <= 0.29

### LAC-14 — Divisão train/validation/test
- **Premissa**: 70/15/15, estratificada por categoria, semente fixa.
- **Reversibilidade**: alta
- **Onde impacta**: preparação do dataset, reprodutibilidade dos experimentos
- **Jev**: sem_criterio=0.36; demais <= 0.12

### LAC-15 — Meta de latência
- **Premissa**: p95 <= 500 ms por ticket em CPU.
- **Reversibilidade**: alta
- **Onde impacta**: NFR de desempenho da API `/predict`
- **Jev**: sem_criterio=0.60 (dúvida); demais <= 0.15

## Decisões do humano — implementação (`/implement`)

### LAC-33 — Telefone seguido de outro grupo de dígitos fica sem máscara
- **Data**: 2026-09-30
- **Etapa/Task**: review da onda 3 · TASK-005 (`src/ticket_classifier/preprocessing.py`)
- **Opções apresentadas**: A) candidato com mais de 15 dígitos mascara o telefone contido nele (ou o candidato inteiro) B) regra literal; vazamento vira risco aceito
- **Recomendação do hm-reviewer**: A (Jev phone_leak=0.48, dúvida)
- **Escolha**: A
- **Observações**: requer atualização de spec/plan (DA-6).

### LAC-34 — Cartão/documento/conta com separadores não é mascarado
- **Data**: 2026-09-30
- **Etapa/Task**: review da onda 3 · TASK-005
- **Opções apresentadas**: A) sequência de grupos de dígitos separados por espaço, "." ou "-" com mais de 15 dígitos no total vira `[NUMBER]` B) DA-6 literal ("dígitos seguidos"); risco aceito
- **Recomendação do hm-reviewer**: A (Jev card_required=0.56, dúvida)
- **Escolha**: A
- **Observações**: requer atualização de spec/plan (DA-6, DATA-07).

### LAC-35 — Banco SQLite inacessível no startup da API
- **Data**: 2026-09-30
- **Etapa/Task**: QA da onda 10 · TASK-023 (`src/ticket_classifier/api/app.py`, `_init_database` no lifespan)
- **Opções apresentadas**: A) manter: banco inacessível impede o startup; API-94 cobre só "Versão promovida carregável" B) tolerar: API sobe e `/health` responde 503
- **Recomendação do hm-qa**: A (Jev api94_db=0.52, dúvida; reviewer: spec não pede tolerância)
- **Escolha**: A
- **Observações**: sem mudança de spec/plan.

### LAC-36 — Campo só com caracteres de controle no `/predict`
- **Data**: 2026-09-30
- **Etapa/Task**: QA da onda 11 · TASK-024 (`src/ticket_classifier/services/prediction_service.py`) × TASK-022 (`src/ticket_classifier/api/schemas.py`)
- **Opções apresentadas**: A) API-92 vale para campo vazio após o pré-processamento DATA-08 → 422 blank, nada gravado B) "vazio" = só após strip; aceitar 200 com título gravado vazio
- **Recomendação do hm-qa**: nenhuma (Jev api92_letter=0.51, dúvida); orquestrador recomendou A
- **Escolha**: A
- **Observações**: requer atualização de spec/plan (API-92, 8.3).
