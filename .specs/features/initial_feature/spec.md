# Especificação — Intelligent Support Ticket Classifier

## 1. Problema

Equipes de suporte recebem tickets em texto livre (título e descrição) e hoje
precisam ler cada um para decidir a categoria (acesso, infraestrutura,
cobrança, bug...) e a prioridade. Essa triagem manual atrasa o primeiro
atendimento, varia de pessoa para pessoa e manda chamados para a fila errada.

A feature entrega um classificador de tickets baseado em NLP/Deep Learning.
Dado um título e uma descrição em inglês, ele devolve a categoria, a
prioridade e a confiança de cada previsão, e sinaliza quando a previsão não é
confiável o bastante para dispensar revisão humana. A inferência fica exposta
por uma API REST (`POST /predict`). Quem corrige uma classificação registra o
rótulo correto (`POST /feedback`), e esse rótulo alimenta retreinamentos
posteriores.

O projeto também serve de portfólio de AI Engineer. Ele precisa mostrar, de
forma reproduzível, que um modelo Transformer com fine-tuning supera
claramente um baseline clássico (TF-IDF + Logistic Regression), com métricas
adequadas a classes desbalanceadas (Macro F1).

Repositório greenfield: não há feature anterior nem código existente.

## 2. Objetivos

- [ ] Um ticket novo (title + description em inglês) enviado a `POST /predict` recebe categoria, prioridade, confiança e sinal de revisão em uma única resposta.
- [ ] O relatório comparativo no test set mostra o Transformer com Macro F1 de categoria >= baseline + 5 pp e Macro F1 de prioridade >= baseline (premissa LAC-05).
- [ ] Reexecutar a preparação de dados e o treino com a mesma semente e configuração reproduz as métricas dentro da tolerância definida (MODEL-06).
- [ ] Toda classificação corrigida via `POST /feedback` fica disponível para o retreinamento seguinte.
- [ ] A API roda em container e informa a saúde e a versão de modelo servida em `GET /health`.

## 3. Fora de Escopo

| Item | Motivo da exclusão |
|---|---|
| Interface gráfica (web ou desktop) para triagem | O consumidor é a API. A demonstração usa a documentação interativa da própria API ou chamadas HTTP. |
| Devolver a equipe de destino ou fazer o roteamento automático para equipes | Decisão LAC-16: a resposta traz só a categoria. O mapeamento categoria → equipe fica para evolução futura. |
| Integração com sistemas de ticketing (Jira, Zendesk, ServiceNow...) | Não pedido. A API é o ponto de integração. |
| Tickets em idiomas diferentes de inglês | Decisão LAC-04: só inglês. Ticket em outro idioma é classificado sem garantia de qualidade. |
| Classificação em lote (vários tickets por requisição) | Não pedido. `POST /predict` recebe um ticket por chamada. |
| Autenticação em `POST /predict` e `GET /health` | Decisão LAC-08: só `POST /feedback` exige chave de API. |
| Gestão de usuários, papéis ou múltiplas chaves de API | Uma única chave de API configurada basta para o portfólio (LAC-08). |
| Disparo automático de retreinamento (agendado ou por volume de feedback) | Decisão LAC-10: o disparo é manual. |
| Dataset sintético ou aumento sintético de dados | Decisão LAC-01: dataset público real. |
| Mascaramento de nomes de pessoas por reconhecimento de entidades | Decisão LAC-18: o Mascaramento cobre só e-mail, URL, telefone e sequências de 6+ dígitos. |
| Rate limiting e proteção contra abuso de `POST /predict` | Não pedido. Escala de demonstração. |

## 4. Glossário de Domínio

Repositório greenfield: todos os termos são novos. Os nomes que viram código
estão em inglês, entre crases.

| Termo | Significado | Onde aparece no código |
|---|---|---|
| Ticket (novo) | Chamado de suporte com `title` e `description` em texto livre, em inglês | a definir no plan |
| `title` (novo) | Título do ticket; 1 a 200 caracteres após remover espaços nas pontas (LAC-12) | a definir no plan |
| `description` (novo) | Descrição do ticket; 1 a 5000 caracteres após remover espaços nas pontas (LAC-12) | a definir no plan |
| `category` (novo) | Categoria do ticket, conjunto fechado: `access`, `infrastructure`, `billing`, `bug`, `other` (LAC-02) | a definir no plan |
| `priority` (novo) | Prioridade/severidade do ticket, conjunto fechado: `low`, `medium`, `high` (LAC-03) | a definir no plan |
| Confiança (novo) | Probabilidade, entre 0 e 1, que o modelo atribui à classe prevista (`category_confidence`, `priority_confidence`) | a definir no plan |
| `top_categories` (novo) | As 3 categorias de maior probabilidade, em ordem decrescente, cada uma com sua probabilidade (LAC-13) | a definir no plan |
| Limiar de baixa confiança (novo) | Valor configurável, default 0.6, abaixo do qual a previsão é marcada para revisão (LAC-06) | a definir no plan |
| `needs_review` (novo) | Indicador booleano de que a previsão tem baixa confiança e deve ser revisada por uma pessoa (LAC-06) | a definir no plan |
| Previsão (novo) | Registro gravado a cada resposta 200 de `POST /predict`: texto mascarado, rótulos previstos, confianças, versão do modelo e data/hora | a definir no plan |
| `prediction_id` (novo) | Identificador único de uma Previsão, devolvido por `POST /predict` e usado em `POST /feedback` (LAC-07) | a definir no plan |
| Feedback (novo) | Registro da categoria e da prioridade corretas de uma Previsão, enviado por `POST /feedback` (LAC-07) | a definir no plan |
| Feedback vigente (novo) | O Feedback mais recente de uma Previsão, pela data/hora de recebimento. É o único usado no retreino (LAC-17) | a definir no plan |
| Mascaramento de PII (novo) | Substituição de dados pessoais do texto por marcadores fixos antes de treinar, inferir ou gravar (LAC-09) | a definir no plan |
| Baseline (novo) | Modelo clássico TF-IDF + Logistic Regression usado como referência de comparação | a definir no plan |
| Modelo Transformer (novo) | Modelo Transformer pré-treinado em inglês, com fine-tuning para prever categoria e prioridade | a definir no plan |
| Split (novo) | Um dos três conjuntos disjuntos do dataset: `train`, `validation`, `test` | a definir no plan |
| Test set congelado (novo) | O split `test` gerado na preparação inicial. Não recebe dados de feedback e é a base de toda comparação e promoção (LAC-10) | a definir no plan |
| Versão de modelo (novo) | Modelo treinado e registrado, com identificador, tipo, métricas, versão dos dados, semente e hiperparâmetros | a definir no plan |
| Versão promovida (novo) | A única Versão de modelo que a API serve em um dado momento (LAC-11) | a definir no plan |
| Macro F1 (novo) | Média simples do F1 de cada classe, sem ponderar pelo suporte. Métrica principal por causa do desbalanceamento | a definir no plan |
| Tabela de mapeamento de rótulos (novo) | Tabela versionada no repositório que converte cada rótulo de origem do dataset público em `category` e `priority` | a definir no plan |
| Chave de API (novo) | Segredo configurado no servidor e exigido no cabeçalho das chamadas a `POST /feedback` (LAC-08) | a definir no plan |

## 5. Atores e Permissões

| Ator | Ação | Condição / restrição |
|---|---|---|
| Consumidor da API (sistema ou pessoa) | Chamar `POST /predict` e `GET /health` | Sem autenticação (LAC-08). |
| Agente de suporte / sistema de triagem com a chave de API | Chamar `POST /feedback` | Precisa enviar a Chave de API válida. Sem chave válida recebe 401 e nada é gravado (FDBK-90). |
| Engenheiro de ML (operador) | Preparar o dataset, treinar, avaliar, analisar erros e disparar o retreinamento | Por execução local ou no container, com acesso ao repositório. Não há endpoint HTTP para treino ou retreino. |
| Processo de expurgo | Apagar Previsões sem Feedback com mais de 90 dias | Automático (LAC-19). Nunca apaga Previsão com Feedback nem Feedback. |
| Processo de retreinamento | Promover automaticamente uma nova Versão de modelo | Só quando a nova versão passa no gate de promoção (RETR-03). Não pode promover sem avaliar no Test set congelado. |

Não há controle de permissão existente: repositório greenfield.

## 6. Histórias de Usuário

### P1: Preparar o dataset e dividir em train/validation/test ⭐ MVP

**História**: Como engenheiro de ML, quero transformar um dataset público de tickets em splits limpos, rotulados com as categorias e prioridades do sistema, para treinar e avaliar modelos de forma reproduzível.

**Por que P1**: Nenhum modelo pode ser treinado ou comparado sem dados rotulados e divididos, e o critério de sucesso depende de um test set estável.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `DATA-01` | WHEN a preparação do dataset é executada sobre o dataset público de origem (LAC-01) THEN o sistema SHALL gerar os splits `train`, `validation` e `test`, e cada registro SHALL conter `title`, `description`, `category` e `priority` |
| `DATA-02` | WHEN um rótulo de origem é convertido THEN o sistema SHALL usar a Tabela de mapeamento de rótulos versionada no repositório para produzir `category` em {`access`, `infrastructure`, `billing`, `bug`, `other`} e `priority` em {`low`, `medium`, `high`} |
| `DATA-03` | WHEN um registro de origem não está em inglês THEN o sistema SHALL excluí-lo dos splits e contabilizá-lo no relatório de preparação (LAC-04) |
| `DATA-04` | WHEN a preparação divide os dados THEN o sistema SHALL usar a proporção 70/15/15 (`train`/`validation`/`test`), estratificada por `category`, com semente fixa configurável (premissa LAC-14) |
| `DATA-05` | WHEN a divisão termina THEN a proporção de cada `category` em cada split SHALL diferir no máximo 1 ponto percentual da proporção no dataset completo |
| `DATA-06` | WHEN a preparação é executada duas vezes com a mesma origem e a mesma semente THEN o sistema SHALL produzir splits idênticos, registro a registro |
| `DATA-07` | WHEN `title` ou `description` é pré-processado THEN o sistema SHALL aplicar o Mascaramento de PII: e-mails viram `[EMAIL]`, URLs viram `[URL]`, números de telefone viram `[PHONE]` e qualquer outra sequência de 6 ou mais dígitos vira `[NUMBER]` (LAC-09, LAC-18) |
| `DATA-08` | WHEN um texto é pré-processado THEN o sistema SHALL remover espaços nas pontas, trocar sequências de espaços, tabulações e quebras de linha por um único espaço e remover caracteres de controle, usando exatamente a mesma rotina no treino e na inferência (mesma entrada gera a mesma saída nos dois) |
| `DATA-09` | WHEN dois ou mais registros têm `title` e `description` idênticos depois do pré-processamento (DATA-07, DATA-08) THEN o sistema SHALL manter só a primeira ocorrência antes da divisão e contabilizar as removidas no relatório de preparação (LAC-27) |
| `DATA-10` | WHEN alguma `category` tem menos de 100 exemplos no split `train` THEN a preparação SHALL falhar sem gravar splits, informando a categoria e a contagem (LAC-26) |
| `DATA-11` | WHEN a preparação termina com sucesso THEN o sistema SHALL emitir um relatório com a contagem por `category` e por `priority` em cada split e o número de registros descartados por motivo |
| `DATA-12` | WHEN alguém clona o repositório THEN o README SHALL trazer a fonte do dataset público, a licença, a versão ou data de obtenção e os passos para obtê-lo e rodar a preparação |

**Teste independente**: rodar a preparação sobre uma amostra fixa do dataset de origem e verificar os três splits, os campos obrigatórios, a estratificação, o relatório e a identidade entre duas execuções com a mesma semente.

### P1: Treinar o baseline e o Transformer e comparar as métricas ⭐ MVP

**História**: Como engenheiro de ML, quero treinar um baseline TF-IDF + Logistic Regression e um Transformer com fine-tuning e compará-los no mesmo test set, para provar com métricas reproduzíveis que o modelo de Deep Learning é melhor.

**Por que P1**: O critério de sucesso da feature é superar claramente o baseline, e a API precisa de uma Versão de modelo treinada para servir.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `MODEL-01` | WHEN o treino do Baseline é executado THEN o sistema SHALL treinar TF-IDF + Logistic Regression para `category` e para `priority` usando só o split `train` e escolher hiperparâmetros só pelo split `validation` |
| `MODEL-02` | WHEN o treino do Modelo Transformer é executado THEN o sistema SHALL fazer fine-tuning de um Transformer pré-treinado em inglês para prever `category` e `priority` usando só o split `train` e escolher o melhor checkpoint só pelo split `validation` |
| `MODEL-03` | WHEN a avaliação é executada no split `test` THEN o sistema SHALL calcular, para cada modelo e para cada alvo (`category`, `priority`), Accuracy, Precision, Recall e F1 por classe, Macro F1 e a Confusion Matrix |
| `MODEL-04` | WHEN os dois modelos foram avaliados THEN o sistema SHALL gerar um relatório comparativo com todas as métricas de MODEL-03 lado a lado e a diferença de Macro F1 em pontos percentuais, por alvo |
| `MODEL-05` | WHEN o relatório comparativo é gerado THEN o sistema SHALL indicar `PASS` se o Macro F1 de `category` do Transformer for >= o do Baseline + 5 pp e o Macro F1 de `priority` do Transformer for >= o do Baseline, e `FAIL` caso contrário (premissa LAC-05) |
| `MODEL-06` | WHEN o treino e a avaliação são reexecutados com os mesmos splits, semente e configuração THEN as métricas do Baseline SHALL ser idênticas às anteriores e o Macro F1 do Transformer SHALL diferir no máximo 1 ponto percentual do anterior, por alvo (LAC-28) |
| `MODEL-07` | WHEN qualquer treino ou seleção de hiperparâmetros é executado THEN o sistema SHALL não usar nenhum registro do split `test` |
| `MODEL-08` | WHEN um treino termina THEN o sistema SHALL registrar uma Versão de modelo com identificador único, tipo (`baseline` ou `transformer`), métricas de `validation` e `test`, identificação da versão dos splits, semente e hiperparâmetros |
| `MODEL-09` | WHEN a primeira Versão de modelo do tipo `transformer` é registrada e não existe Versão promovida THEN o sistema SHALL marcá-la como Versão promovida (LAC-11) |

**Teste independente**: com os splits de uma amostra pequena e fixa, rodar os dois treinos e a avaliação e verificar que o relatório comparativo traz todas as métricas, o veredito `PASS`/`FAIL` e as Versões de modelo registradas, com o Transformer promovido.

### P1: Classificar um ticket novo via POST /predict ⭐ MVP

**História**: Como consumidor da API, quero enviar o título e a descrição de um ticket e receber a categoria, a prioridade e a confiança, para encaminhar o chamado sem triagem manual.

**Por que P1**: É o valor central da feature e o ponto demonstrável do critério de sucesso ("classificação de tickets novos via API").

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `API-01` | WHEN `POST /predict` recebe `title` e `description` válidos THEN o sistema SHALL responder 200 com `prediction_id`, `category`, `category_confidence`, `priority`, `priority_confidence`, `top_categories`, `needs_review` e `model_version` |
| `API-02` | WHEN a resposta de `POST /predict` é montada THEN `category_confidence` e `priority_confidence` SHALL estar entre 0 e 1, `top_categories` SHALL ter 3 itens distintos em ordem decrescente de probabilidade e o primeiro item SHALL ser igual a `category` com probabilidade igual a `category_confidence` |
| `API-03` | WHEN `category_confidence` ou `priority_confidence` fica abaixo do Limiar de baixa confiança (LAC-22) THEN o sistema SHALL devolver `needs_review` = `true`, mantendo a classe prevista; caso contrário SHALL devolver `needs_review` = `false` (LAC-06) |
| `API-04` | WHEN o Limiar de baixa confiança não é configurado THEN o sistema SHALL usar 0.6; WHEN é configurado com outro valor entre 0 e 1 THEN o sistema SHALL usar esse valor sem alteração de código (LAC-06) |
| `API-05` | WHEN `POST /predict` recebe um ticket THEN o sistema SHALL aplicar o mesmo pré-processamento do treino (DATA-07 e DATA-08) antes da inferência |
| `API-06` | WHEN o texto pré-processado excede o limite de tokens do modelo THEN o sistema SHALL truncá-lo e responder 200 normalmente (LAC-12) |
| `API-07` | WHEN `POST /predict` responde 200 THEN o sistema SHALL gravar a Previsão com `prediction_id`, `title` e `description` mascarados, rótulos previstos, confianças, `model_version` e data/hora, e SHALL não gravar o texto original sem mascaramento (LAC-09) |
| `API-08` | WHEN `POST /predict` é chamado THEN o sistema SHALL não exigir Chave de API (LAC-08) |
| `API-09` | WHEN a API está em execução THEN ela SHALL servir a Versão promovida, e `model_version` na resposta SHALL ser o identificador dessa versão (LAC-11); uma versão promovida depois do início da API SHALL passar a ser servida no próximo reinício (LAC-24) |

**Teste independente**: com uma Versão promovida disponível (pode ser um modelo pequeno treinado na amostra de teste), subir a API, chamar `POST /predict` com um ticket válido e conferir os campos, os limites de API-02, o `needs_review` com limiares 0.0 e 1.0 e a Previsão gravada com texto mascarado.

### P1: Registrar a classificação correta ⭐ MVP

**História**: Como agente de suporte (ou sistema de triagem) com a chave de API, quero informar a categoria e a prioridade corretas de uma previsão, para que o modelo aprenda com os erros no próximo retreinamento.

**Por que P1**: É requisito funcional explícito da descrição e a origem dos dados do retreinamento.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `FDBK-01` | WHEN `POST /feedback` recebe Chave de API válida, `prediction_id` existente, `category` e `priority` válidas THEN o sistema SHALL gravar o Feedback com `prediction_id`, rótulos corretos e data/hora de recebimento e responder 201 com `feedback_id` (LAC-07) |
| `FDBK-02` | WHEN chega um novo Feedback para um `prediction_id` que já tem Feedback THEN o sistema SHALL gravar o novo sem apagar nem alterar os anteriores e responder 201 (LAC-17) |
| `FDBK-03` | WHEN os Feedbacks de um `prediction_id` são lidos para retreino THEN o sistema SHALL considerar vigente só o de data/hora de recebimento mais recente (LAC-17) |
| `FDBK-04` | WHEN o Feedback informa rótulos iguais aos previstos THEN o sistema SHALL aceitá-lo como confirmação e gravá-lo como qualquer outro Feedback |

**Teste independente**: com uma Previsão gravada, chamar `POST /feedback` com e sem chave, com `prediction_id` inexistente e duas vezes para o mesmo `prediction_id`, e conferir as respostas e os registros gravados.

### P2: Operar a API em container com health check

**História**: Como engenheiro de ML, quero subir a API em um container e consultar sua saúde, para demonstrar e implantar o classificador de forma repetível.

**Por que P2**: É diferencial pedido para o portfólio e facilita a demo, mas a classificação funciona sem container.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `OPS-01` | WHEN `GET /health` é chamado com a Versão promovida carregada THEN o sistema SHALL responder 200 com `status` = `ok` e `model_version`, sem exigir Chave de API |
| `OPS-02` | WHEN a imagem de container é construída e iniciada com os comandos documentados no README THEN `GET /health` SHALL responder 200 e `POST /predict` SHALL classificar um ticket válido |
| `OPS-03` | WHEN a suíte de testes automatizados é executada com o comando documentado no README THEN ela SHALL rodar sem GPU e sem acesso à internet |

**Teste independente**: construir a imagem, iniciar o container e chamar `GET /health` e `POST /predict`.

### P2: Rastrear experimentos e versionar modelos

**História**: Como engenheiro de ML, quero que todo treino registre parâmetros, métricas e artefatos em uma ferramenta de rastreamento de experimentos, para comparar execuções e saber qual versão está em produção.

**Por que P2**: É diferencial de portfólio. O MVP já grava Versões de modelo (MODEL-08), mas sem histórico navegável.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `OPS-04` | WHEN qualquer treino (Baseline, Transformer ou retreino) é executado THEN o sistema SHALL registrar no rastreamento de experimentos os hiperparâmetros, a semente, a identificação da versão dos splits, todas as métricas de MODEL-03, a Confusion Matrix como artefato e o modelo treinado |
| `OPS-05` | WHEN as Versões de modelo são listadas THEN o sistema SHALL mostrar identificador, tipo, Macro F1 de `category` e de `priority` no Test set congelado, data e se é a Versão promovida |
| `OPS-06` | WHEN uma Versão de modelo é promovida THEN a versão promovida anterior SHALL deixar de ser promovida, de modo que exista exatamente uma Versão promovida |

**Teste independente**: rodar um treino na amostra de teste e verificar o registro no rastreamento de experimentos e a listagem de versões.

### P2: Analisar erros do modelo

**História**: Como engenheiro de ML, quero ver onde o modelo erra e como a confiança se relaciona com o acerto, para ajustar o limiar de baixa confiança e explicar as limitações no README.

**Por que P2**: É diferencial pedido (análise de erros, tratamento de baixa confiança), mas não bloqueia a classificação.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `MODEL-10` | WHEN a análise de erros é executada com a Versão promovida sobre o split `test` THEN o sistema SHALL gerar a lista de tickets classificados errado, com rótulo real, rótulo previsto e confiança, e os 5 pares (real, previsto) de `category` mais frequentes entre os erros |
| `MODEL-11` | WHEN a análise de erros é executada THEN o sistema SHALL informar, para o Limiar de baixa confiança em vigor, a fração de tickets do `test` com `needs_review` = `true` e a Accuracy de `category` separada para `needs_review` = `true` e `needs_review` = `false` |

**Teste independente**: rodar a análise com um modelo treinado na amostra e verificar a lista de erros, os pares de confusão e as métricas por faixa de confiança.

### P2: Expurgar Previsões antigas sem Feedback

**História**: Como responsável pelo sistema, quero que Previsões sem Feedback sejam apagadas depois de 90 dias, para não guardar texto de tickets além do necessário (LGPD).

**Por que P2**: Vem da decisão LAC-19. Reduz dado pessoal residual (nomes não são mascarados), mas não bloqueia a classificação nem o feedback.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `OPS-07` | WHEN o expurgo é executado THEN o sistema SHALL apagar toda Previsão sem Feedback gravada há mais de 90 dias e SHALL manter as Previsões com pelo menos um Feedback, seus Feedbacks e as Previsões com 90 dias ou menos (LAC-19) |
| `OPS-08` | WHEN a API está em operação THEN o expurgo SHALL rodar automaticamente, sem ação manual, e cada execução SHALL registrar quantas Previsões apagou (LAC-19) |

**Teste independente**: gravar Previsões com datas de 89, 90 e 91 dias, uma delas com Feedback, executar o expurgo e conferir que só a de 91 dias sem Feedback foi apagada.

### P3: Retreinar com feedbacks e promover nova versão

**História**: Como engenheiro de ML, quero disparar um retreinamento que use os feedbacks registrados e só coloque a nova versão em produção se ela não piorar, para melhorar o modelo continuamente sem risco de regressão.

**Por que P3**: Fecha o ciclo de aprendizado, mas depende de volume de feedback que só existe depois do uso.

**Critérios de aceite**:

| ID | Critério |
|---|---|
| `RETR-01` | WHEN o retreinamento é disparado manualmente THEN o sistema SHALL treinar uma nova Versão de modelo do tipo `transformer` com o split `train` original mais o texto mascarado e os rótulos do Feedback vigente de cada Previsão com Feedback, usando o split `validation` original para seleção (LAC-10, LAC-17) |
| `RETR-02` | WHEN a nova versão é avaliada THEN o sistema SHALL usar o Test set congelado, o mesmo usado pela Versão promovida, e SHALL nunca incluir dados de Feedback no Test set congelado |
| `RETR-03` | WHEN a nova versão tem, no Test set congelado, Macro F1 de `category` >= e Macro F1 de `priority` >= os da Versão promovida (LAC-23) THEN o sistema SHALL promovê-la automaticamente; caso contrário SHALL registrá-la sem promover e manter a Versão promovida atual (LAC-10) |
| `RETR-04` | WHEN o retreinamento termina THEN o sistema SHALL informar o número de Feedbacks vigentes usados, as métricas da nova versão e da Versão promovida no Test set congelado e a decisão (`promoted` ou `not promoted`) |

**Teste independente**: com Previsões e Feedbacks gravados na amostra de teste, disparar o retreino e verificar o conjunto de treino montado, a avaliação no Test set congelado e a decisão de promoção nos dois cenários (melhor e pior).

## 7. Estados e Transições

Entidade com ciclo de vida: **Versão de modelo**.

| De | Evento | Para | Quem pode disparar | Efeito colateral |
|---|---|---|---|---|
| — | Treino ou retreino termina com sucesso | `registered` | Engenheiro de ML (treino) ou processo de retreinamento | Métricas e metadados gravados (MODEL-08) |
| `registered` | Primeira versão `transformer` sem Versão promovida existente (MODEL-09) ou aprovada no gate de promoção (RETR-03) | `promoted` | Processo de treino ou de retreinamento | A versão antes promovida passa a `retired` (OPS-06) |
| `promoted` | Outra versão é promovida | `retired` | Processo de retreinamento | A API passa a servir a nova versão no próximo reinício (LAC-24) |

- Estados terminais: `retired`, e `registered` quando a versão reprovou no gate (fica como candidata, sem promoção futura automática).
- Transições proibidas: `retired` → `promoted` e `registered` → `promoted` sem avaliação no Test set congelado. Voltar a uma versão anterior (rollback) está fora desta feature.
- Invariante: existe no máximo uma versão `promoted`, e exatamente uma depois do primeiro treino Transformer.

Previsão e Feedback não têm ciclo de vida: são registros de inclusão (append-only).

## 8. Casos de Borda e Erros

| ID | Situação | Comportamento esperado |
|---|---|---|
| `DATA-90` | Registro de origem com `title` ou `description` vazio | WHEN o registro de origem tem `title` ou `description` vazio depois do pré-processamento THEN o sistema SHALL descartá-lo e contabilizá-lo no relatório de preparação |
| `DATA-91` | Rótulo de origem ausente da Tabela de mapeamento | WHEN um rótulo de origem não consta na Tabela de mapeamento de rótulos THEN a preparação SHALL falhar sem gravar splits, informando o rótulo não mapeado |
| `DATA-92` | Dataset de origem indisponível | WHEN o arquivo do dataset de origem não existe ou não pode ser lido THEN a preparação SHALL falhar sem gravar splits parciais, informando o caminho esperado |
| `MODEL-90` | Split vazio | WHEN o split `train`, `validation` ou `test` está vazio ou ausente THEN o treino ou a avaliação SHALL falhar antes de treinar, informando o split |
| `API-90` | Campo obrigatório ausente | WHEN `POST /predict` recebe corpo sem `title` ou sem `description` THEN o sistema SHALL responder 422 indicando o campo, sem gravar Previsão |
| `API-91` | Limite superior de tamanho | WHEN `title` tem mais de 200 caracteres ou `description` tem mais de 5000 caracteres THEN o sistema SHALL responder 422 indicando o campo e o limite, sem gravar Previsão (LAC-12) |
| `API-92` | Texto só com espaços | WHEN `title` ou `description` fica vazio depois de remover os espaços nas pontas THEN o sistema SHALL responder 422 indicando o campo, sem gravar Previsão |
| `API-93` | Corpo inválido | WHEN `POST /predict` recebe corpo que não é JSON ou traz `title`/`description` de tipo diferente de texto THEN o sistema SHALL responder 422 sem gravar Previsão |
| `API-94` | Nenhuma Versão promovida disponível | WHEN a API inicia sem Versão promovida carregável THEN o sistema SHALL iniciar mesmo assim, `GET /health` SHALL responder 503 com `status` = `unavailable` e `POST /predict` SHALL responder 503 sem gravar Previsão (LAC-21) |
| `API-95` | Concorrência em /predict | WHEN várias requisições válidas chegam a `POST /predict` ao mesmo tempo THEN cada uma SHALL receber um `prediction_id` distinto e a Previsão gravada SHALL corresponder ao seu próprio texto |
| `API-96` | Falha ao gravar a Previsão | WHEN a gravação da Previsão falha THEN o sistema SHALL responder 503 sem devolver a previsão (LAC-29) |
| `FDBK-90` | Chave de API ausente ou inválida | WHEN `POST /feedback` chega sem Chave de API ou com chave diferente da configurada THEN o sistema SHALL responder 401 sem gravar Feedback (LAC-08) |
| `FDBK-91` | Previsão inexistente | WHEN `POST /feedback` traz `prediction_id` que não existe THEN o sistema SHALL responder 404 sem gravar Feedback |
| `FDBK-92` | Rótulo inválido | WHEN `POST /feedback` traz `category` fora de {`access`, `infrastructure`, `billing`, `bug`, `other`} ou `priority` fora de {`low`, `medium`, `high`}, ou omite um dos dois THEN o sistema SHALL responder 422 listando os valores aceitos, sem gravar Feedback |
| `FDBK-93` | Servidor sem Chave de API configurada | WHEN a API está em execução sem Chave de API configurada THEN `POST /feedback` SHALL responder 401 a toda requisição sem gravar Feedback, e `POST /predict` e `GET /health` SHALL continuar funcionando (LAC-20) |
| `FDBK-94` | Feedbacks concorrentes | WHEN dois Feedbacks válidos para o mesmo `prediction_id` chegam ao mesmo tempo THEN o sistema SHALL gravar os dois, cada um com sua data/hora de recebimento, e o Feedback vigente SHALL ser o de data/hora mais recente |
| `FDBK-95` | Falha ao gravar o Feedback | WHEN a gravação do Feedback falha THEN o sistema SHALL responder 503 e não SHALL deixar Feedback gravado pela metade |
| `OPS-90` | Feedback durante o expurgo | WHEN um Feedback chega para uma Previsão que o expurgo está apagando THEN o sistema SHALL manter a Previsão com o Feedback ou responder 404 sem gravar o Feedback, e SHALL nunca existir Feedback sem Previsão |
| `RETR-90` | Nenhum Feedback registrado | WHEN o retreinamento é disparado sem nenhum Feedback THEN o sistema SHALL abortar sem registrar Versão de modelo e sem alterar a Versão promovida (LAC-25) |
| `RETR-91` | Falha durante o retreino | WHEN o retreinamento falha antes de terminar a avaliação THEN o sistema SHALL não promover nenhuma versão e SHALL manter a Versão promovida atual |
| `RETR-92` | Retreinos concorrentes | WHEN um retreinamento é disparado enquanto outro está em execução THEN o sistema SHALL recusar o novo disparo sem afetar o retreino em execução (LAC-30) |

Permissão ausente: coberta por FDBK-90. `POST /predict` e `GET /health` não exigem permissão (LAC-08).

## 9. Mensagens ao Usuário

Idioma: inglês. Canal: corpo JSON da resposta HTTP (API) ou saída do comando
(preparação, treino, retreino). Os textos exatos são premissa (LAC-31).

| Situação | Mensagem | Tom / canal |
|---|---|---|
| Campo obrigatório ausente (API-90) | "Field '<field>' is required." | neutro / JSON 422 |
| Limite de tamanho excedido (API-91) | "Field '<field>' must be between 1 and <max> characters." | neutro / JSON 422 |
| Texto só com espaços (API-92) | "Field '<field>' must not be blank." | neutro / JSON 422 |
| Corpo inválido (API-93) | "Invalid request body." | neutro / JSON 422 |
| Chave de API ausente ou inválida (FDBK-90) | "Invalid or missing API key." | neutro / JSON 401 |
| Previsão inexistente (FDBK-91) | "Prediction '<prediction_id>' not found." | neutro / JSON 404 |
| Rótulo inválido (FDBK-92) | "Field '<field>' must be one of: <allowed values>." | neutro / JSON 422 |
| Falha ao gravar Feedback (FDBK-95) | "Service temporarily unavailable. Please try again later." | neutro / JSON 503 |
| Rótulo de origem não mapeado (DATA-91) | "Unmapped source label '<label>'. Add it to the label mapping table." | técnico / saída do comando |
| Dataset de origem indisponível (DATA-92) | "Source dataset not found at '<path>'. See README for download instructions." | técnico / saída do comando |
| Split vazio (MODEL-90) | "Split '<split>' is empty or missing. Run data preparation first." | técnico / saída do comando |
| Resultado do retreino (RETR-04) | "Retraining finished: <n> feedback records used. Decision: <promoted or not promoted>." | técnico / saída do comando |
| Sem Versão promovida (API-94) | "Model not available. Try again later." (em `GET /health`: `status` = `unavailable`) | neutro / JSON 503 |
| Falha ao gravar Previsão (API-96) | "Service temporarily unavailable. Please try again later." | neutro / JSON 503 |
| Servidor sem Chave de API (FDBK-93) | "Invalid or missing API key." na resposta; na inicialização: "API key not configured: /feedback will reject all requests." | neutro / JSON 401 e log de inicialização |
| Retreino sem Feedback (RETR-90) | "No feedback records available. Retraining aborted." | técnico / saída do comando |
| Retreino concorrente (RETR-92) | "A retraining run is already in progress." | técnico / saída do comando |
| Categoria com poucos exemplos (DATA-10) | "Category '<category>' has <n> training examples; minimum is 100." | técnico / saída do comando |
| Resultado do expurgo (OPS-08) | "Purge finished: <n> predictions without feedback older than 90 days deleted." | técnico / log |

## 10. Requisitos Não-Funcionais

| Eixo | Requisito |
|---|---|
| Performance | `POST /predict` com p95 <= 500 ms por ticket, em CPU, com a Versão promovida já carregada (premissa LAC-15). Treino sem meta de tempo. |
| Volume | Sem meta de throughput: escala de demonstração de portfólio. O dataset tem o tamanho do dataset público escolhido após o filtro de idioma. |
| Concorrência | `POST /predict` e `POST /feedback` aceitam requisições simultâneas sem misturar dados (API-95, FDBK-94). Retreinos concorrentes: RETR-92. Feedback durante o expurgo: OPS-90. |
| Segurança | Só `POST /feedback` exige Chave de API (LAC-08). A chave vem de configuração e nunca aparece em log, resposta ou repositório. Comportamento sem chave configurada: FDBK-93. |
| Privacidade / LGPD | Todo texto passa pelo Mascaramento de PII antes de treinar, inferir ou gravar (LAC-09). Só o texto mascarado é gravado (API-07). Previsões sem Feedback são apagadas automaticamente depois de 90 dias; Previsões com Feedback e seus Feedbacks ficam guardados como dado de treino (LAC-19, OPS-07, OPS-08). |
| Acessibilidade | Não se aplica: sem interface gráfica. |
| i18n / formato | Inglês na interface (API), nas mensagens e nos textos de saída; não há e-mails. Tickets em inglês (LAC-04). Datas e horas gravadas em UTC, formato ISO 8601. Confianças são números decimais entre 0 e 1. |
| Observabilidade | Cada Previsão fica rastreável pelo `prediction_id` e pela `model_version`. Treinos ficam rastreáveis pela Versão de modelo e pelo registro de experimentos (MODEL-08, OPS-04). `GET /health` expõe a versão servida (OPS-01). |
| Compatibilidade | Greenfield: nada existente a preservar. Contrato estável: os valores de `category` e `priority` são conjuntos fechados (LAC-02, LAC-03). A stack citada na descrição (Python, Pandas, Scikit-learn, PyTorch, Hugging Face Transformers, FastAPI, Pydantic, MLflow, Docker) é insumo para o plan, que decide. |

## 11. Rastreabilidade

| ID | História | Prioridade | Status |
|---|---|---|---|
| `DATA-01` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-02` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-03` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-04` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-05` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-06` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-07` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-08` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-09` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-10` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-11` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-12` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-90` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-91` | P1: Preparar o dataset | P1 | Pendente |
| `DATA-92` | P1: Preparar o dataset | P1 | Pendente |
| `MODEL-01` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-02` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-03` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-04` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-05` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-06` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-07` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-08` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-09` | P1: Treinar e comparar modelos | P1 | Pendente |
| `MODEL-90` | P1: Treinar e comparar modelos | P1 | Pendente |
| `API-01` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-02` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-03` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-04` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-05` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-06` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-07` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-08` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-09` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-90` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-91` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-92` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-93` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-94` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-95` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `API-96` | P1: Classificar ticket via POST /predict | P1 | Pendente |
| `FDBK-01` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-02` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-03` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-04` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-90` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-91` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-92` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-93` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-94` | P1: Registrar a classificação correta | P1 | Pendente |
| `FDBK-95` | P1: Registrar a classificação correta | P1 | Pendente |
| `OPS-01` | P2: Operar a API em container | P2 | Pendente |
| `OPS-02` | P2: Operar a API em container | P2 | Pendente |
| `OPS-03` | P2: Operar a API em container | P2 | Pendente |
| `OPS-04` | P2: Rastrear experimentos e versionar | P2 | Pendente |
| `OPS-05` | P2: Rastrear experimentos e versionar | P2 | Pendente |
| `OPS-06` | P2: Rastrear experimentos e versionar | P2 | Pendente |
| `MODEL-10` | P2: Analisar erros do modelo | P2 | Pendente |
| `MODEL-11` | P2: Analisar erros do modelo | P2 | Pendente |
| `OPS-07` | P2: Expurgar Previsões antigas | P2 | Pendente |
| `OPS-08` | P2: Expurgar Previsões antigas | P2 | Pendente |
| `OPS-90` | P2: Expurgar Previsões antigas | P2 | Pendente |
| `RETR-01` | P3: Retreinar e promover | P3 | Pendente |
| `RETR-02` | P3: Retreinar e promover | P3 | Pendente |
| `RETR-03` | P3: Retreinar e promover | P3 | Pendente |
| `RETR-04` | P3: Retreinar e promover | P3 | Pendente |
| `RETR-90` | P3: Retreinar e promover | P3 | Pendente |
| `RETR-91` | P3: Retreinar e promover | P3 | Pendente |
| `RETR-92` | P3: Retreinar e promover | P3 | Pendente |

**Cobertura**: 69 requisitos no total (P1: 51, P2: 11, P3: 7).

## 12. Lacunas

Decididas pelo humano em `decisions.md` (convertidas em critérios): LAC-01,
LAC-02, LAC-03, LAC-04, LAC-06, LAC-07, LAC-08, LAC-09, LAC-10, LAC-11,
LAC-12, LAC-13, LAC-16, LAC-17, LAC-18, LAC-19, LAC-20, LAC-21, LAC-22,
LAC-23, LAC-24, LAC-25, LAC-26, LAC-27, LAC-28, LAC-29, LAC-30.

Nenhuma lacuna bloqueante em aberto. Premissas não bloqueantes abaixo.

[LACUNA:LAC-05|NAO_BLOQUEANTE]
Pergunta: Como medir "superar claramente o baseline"?
Opções:
  A) Macro F1 de `category` do Transformer >= Baseline + 5 pp no test set, e Macro F1 de `priority` não inferior ao Baseline
  B) ganho >= 3 pp de Macro F1 em `category` e em `priority`
  C) qualquer ganho positivo de Macro F1 de `category`
Recomendação: A — ganho claro no alvo principal, sem regressão em prioridade.
Impacto se errado: o veredito `PASS`/`FAIL` (MODEL-05) muda; basta ajustar o limiar.
Premissa aplicada: A (MODEL-05, Objetivos). Jev: sem_criterio=0.64 (dúvida); demais <= 0.29.

[LACUNA:LAC-14|NAO_BLOQUEANTE]
Pergunta: Proporção e forma da divisão train/validation/test?
Opções:
  A) 70/15/15 estratificado por `category`, semente fixa
  B) 80/10/10 aleatório
  C) 60/20/20 estratificado
Recomendação: A — preserva classes raras em todos os splits.
Impacto se errado: métricas levemente diferentes; reversível.
Premissa aplicada: A (DATA-04). Jev: sem_criterio=0.36; demais <= 0.12.

[LACUNA:LAC-15|NAO_BLOQUEANTE]
Pergunta: Qual a meta de latência de inferência?
Opções:
  A) p95 <= 500 ms por ticket em CPU
  B) p95 <= 200 ms em CPU
  C) sem meta
Recomendação: A — alcançável com Transformer destilado em CPU.
Impacto se errado: exigência de hardware diferente; ajustável.
Premissa aplicada: A (seção 10, Performance). Jev: sem_criterio=0.60 (dúvida); demais <= 0.15.

[LACUNA:LAC-31|NAO_BLOQUEANTE]
Pergunta: Qual o texto exato das mensagens de erro e de saída?
Opções:
  A) os textos em inglês da seção 9
  B) textos definidos na implementação
Recomendação: A — o texto já fica fixado e testável.
Impacto se errado: só redação; reversível.
Premissa aplicada: A (seção 9). Classificação por regra de `lacunas.md` (texto exato de mensagem = NAO_BLOQUEANTE), sem Jev.

## 13. Critérios de Sucesso da Feature

- [ ] O relatório comparativo no Test set congelado mostra `PASS` em MODEL-05 (Macro F1 de `category` do Transformer >= Baseline + 5 pp e Macro F1 de `priority` >= Baseline).
- [ ] Um ticket novo, fora do dataset, enviado a `POST /predict` na API em container recebe resposta 200 com todos os campos de API-01 em até 500 ms (p95, CPU).
- [ ] Reexecutar preparação, treino e avaliação a partir do README em uma máquina limpa reproduz as métricas publicadas dentro da tolerância de MODEL-06.
- [ ] Um Feedback registrado via `POST /feedback` aparece no conjunto de treino do retreinamento seguinte (RETR-01).
- [ ] O README explica a arquitetura, as decisões técnicas, a obtenção do dataset e traz as métricas comparativas e a análise de erros.

## Autoverificação do spec

- [x] Todo critério de aceite tem ID único e formato WHEN/THEN/SHALL — 69 IDs, 0 duplicados, todas as linhas com WHEN, THEN e SHALL (verificado com as funções `parse_spec` do `check_plan.py`).
- [x] Toda história P1 é demonstrável isoladamente — as 4 histórias P1 têm "Teste independente" próprio. A de `POST /predict` usa um modelo pequeno treinado na amostra de teste.
- [x] Seção "Fora de Escopo" tem pelo menos um item — 11 itens.
- [x] Glossário usa termos que existem no codebase — repositório greenfield, sem código (verificado): todos os termos estão marcados `(novo)`.
- [x] Casos de borda cobrem vazio (DATA-90, API-92, MODEL-90, RETR-90), limite (API-91, DATA-10, OPS-07), inválido (API-93, FDBK-92, DATA-91), concorrência (API-95, FDBK-94, RETR-92, OPS-90), falha de dependência (DATA-92, API-94, API-96, FDBK-95, RETR-91) e permissão (FDBK-90, FDBK-93).
- [x] Toda seção de NFR está preenchida ou marcada "não se aplica" — 9 eixos; Acessibilidade = não se aplica.
- [x] Toda lacuna tem severidade, opções e recomendação — LAC-01..LAC-04, LAC-06..LAC-13, LAC-16..LAC-30 decididas em `decisions.md`; LAC-05, LAC-14, LAC-15 e LAC-31 com bloco `NAO_BLOQUEANTE` completo. Nenhuma lacuna bloqueante nem marcador de decisão pendente restante.
- [x] Nenhuma decisão técnica de implementação vazou para o spec — só aparecem contratos (rotas, campos, códigos HTTP) e os modelos exigidos pela descrição (TF-IDF + Logistic Regression, Transformer). A stack fica como insumo do plan (seção 10). Não há nome de classe nem caminho de arquivo.

**Resultado**: 8/8 itens OK. Rastreabilidade: 69/69 IDs na seção 11 (P1: 51, P2: 11, P3: 7).
