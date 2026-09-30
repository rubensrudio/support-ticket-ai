# Tarefas — Intelligent Support Ticket Classifier

Caminhos de produção relativos à raiz do repositório. Contratos `CT-n` e
decisões `DA-n` estão em `plan.md` (seções 8.2 e 5.3). Comandos de gate:
`plan.md` seção 3.

### TASK-001 — Scaffold Python project with uv and CLI entry point
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `OPS-03`
- **Tipo**: infra
- **Risco**: médio
- **Perfil**: infra
- **Depende de**: —
- **Arquivos de produção**:
  - `pyproject.toml`
  - `src/ticket_classifier/cli.py`
- **Arquivos de teste**:
  - `tests/conftest.py`
  - `tests/unit/test_cli.py`
- **Wiring permitido**:
  - `src/ticket_classifier/__init__.py` (criar; só `__version__ = "0.1.0"`)
  - `.gitignore` (criar; entradas: `.venv/`, `reports/`, `dist/`, `data/raw/`, `data/processed/`, `artifacts/`, `var/`, `mlflow.db`, `mlartifacts/`, `mlruns/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`)
  - `uv.lock` (gerado por `uv lock`; não editar à mão)
  - `README.md` (criar em inglês; seções "Overview", "Requirements", "Setup", "Development" com os comandos de lint/typecheck/test/build da seção 3 do plan, e "Tests run offline" explicando OPS-03)
- **Reusa**: —
- **Contrato**:
  - CT-28 — `main(argv: Sequence[str] | None = None) -> int`; `build_parser() -> argparse.ArgumentParser`; padrão `_add_<nome>_command(subparsers)` (produz)
- **Testes**: unit
- **Descrição**: Criar o projeto `uv` (DA-1) com `[project]` Python `>=3.12,<3.13`, dependências da seção 11 do plan (runtime, extra `train` com `mlflow`, `[dependency-groups] dev`), `torch` pelo índice CPU (DA-2: `[tool.uv.sources]` + `[[tool.uv.index]]` `name = "pytorch-cpu"`, `explicit = true`), build `hatchling` com pacote `src/ticket_classifier`, `[project.scripts] ticket-classifier = "ticket_classifier.cli:main"`. Configurar `[tool.ruff]` (line-length 100, regras `E,F,I,B,UP`), `[tool.mypy]` (`python_version = "3.12"`, `disallow_untyped_defs = true`, `ignore_missing_imports = true`) e `[tool.pytest.ini_options]` (`testpaths = ["tests"]`, `addopts = "-q --junitxml=reports/junit.xml -m 'not container'"`, marcador `container`). `cli.py` só com parser, subparsers vazios e `main`. `tests/conftest.py` define no import `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=""`.
- **Done when**:
  - [ ] `uv sync --all-extras` termina com exit 0 e gera `uv.lock`
  - [ ] `uv run python -c "import torch; assert torch.version.cuda is None"` termina com exit 0
  - [ ] `uv run ticket-classifier --help` termina com exit 0
  - [ ] `uv run pytest` termina com exit 0 e cria `reports/junit.xml`
  - [ ] `uv run ruff check . --output-format=concise` e `uv run mypy src` terminam com exit 0
  - [ ] `uv build --wheel --out-dir dist` cria um arquivo `dist/*.whl`
  - [ ] `tests/unit/test_cli.py` prova que `main(["--help"])` levanta `SystemExit(0)` e que `main([])` devolve código diferente de 0 sem exceção não tratada
  - [ ] `README.md` contém os títulos `Overview`, `Requirements`, `Setup`, `Development` e `Tests run offline`
- **Não fazer**:
  - Não criar subcomandos nem módulos de domínio (labels, settings etc.)
  - Não capturar `PipelineError` em `main` (é da TASK-002)
  - Não criar `Dockerfile`

---

### TASK-002 — Define label sets and pipeline error type
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `DATA-02`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-001
- **Arquivos de produção**:
  - `src/ticket_classifier/labels.py`
  - `src/ticket_classifier/errors.py`
- **Arquivos de teste**:
  - `tests/unit/test_labels.py`
  - `tests/unit/test_errors.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas importar `PipelineError` e capturá-lo em `main`: `message` em stderr, retorno 1)
- **Reusa**:
  - `src/ticket_classifier/cli.py` → `main`, `build_parser` (CT-28)
- **Contrato**:
  - CT-1 — `CATEGORIES`, `PRIORITIES`, `Category`, `Priority`, `TARGETS` (produz)
  - CT-2 — `class PipelineError(Exception)` com `message: str` (produz)
  - CT-28 (consome)
- **Testes**: unit
- **Descrição**: Constantes e tipos `Literal` exatamente como em CT-1 (ordem fixa, usada em mensagens e desempates). `PipelineError` guarda `message` e `str(exc) == message`. Em `cli.main`, envolver a execução do handler: `PipelineError` → imprime `message` em stderr e devolve 1 (DA-17).
- **Done when**:
  - [ ] `CATEGORIES == ("access", "infrastructure", "billing", "bug", "other")` e `PRIORITIES == ("low", "medium", "high")` verificados em teste
  - [ ] `typing.get_args(Category)` é igual a `CATEGORIES` e `typing.get_args(Priority)` é igual a `PRIORITIES`
  - [ ] Teste com handler que levanta `PipelineError("boom")` (via `monkeypatch` de `build_parser`) mostra `main` devolvendo 1 e `boom` em stderr (`capsys`)
  - [ ] `uv run pytest tests/unit/test_labels.py tests/unit/test_errors.py` passa
- **Não fazer**:
  - Não criar enum paralelo nem listas de rótulos em outros módulos
  - Não registrar subcomandos na CLI

---

### TASK-003 — Define classifier protocol and probability output
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `API-02`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-002
- **Arquivos de produção**:
  - `src/ticket_classifier/models/base.py`
- **Arquivos de teste**:
  - `tests/unit/test_models_base.py`
- **Wiring permitido**:
  - `src/ticket_classifier/models/__init__.py` (criar vazio)
- **Reusa**:
  - `src/ticket_classifier/labels.py` → `CATEGORIES`, `PRIORITIES`
- **Contrato**:
  - CT-3 — `ClassProbabilities`, `TicketClassifier(Protocol)`, `top_label(probs, order) -> tuple[str, float]` (produz)
  - CT-1 (consome)
- **Testes**: unit
- **Descrição**: `ClassProbabilities` congelada; `TicketClassifier` como `typing.Protocol` com `kind`, `predict_proba`, `save`. `top_label` devolve o rótulo de maior probabilidade e o valor exato (sem arredondar); empate resolvido pela posição em `order`.
- **Done when**:
  - [ ] `top_label({"access": 0.2, "bug": 0.5, "other": 0.3}, CATEGORIES) == ("bug", 0.5)`
  - [ ] Empate `{"billing": 0.4, "access": 0.4, "other": 0.2}` com `order=CATEGORIES` devolve `("access", 0.4)`
  - [ ] Uma classe de teste com `kind`, `predict_proba` e `save` passa em `isinstance(obj, TicketClassifier)` com o Protocol marcado `@runtime_checkable`
  - [ ] `uv run pytest tests/unit/test_models_base.py` passa
- **Não fazer**:
  - Não implementar classificadores concretos (TASK-012, TASK-013)

---

### TASK-004 — Load runtime settings from environment
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `API-04`, `FDBK-93`
- **Tipo**: config
- **Risco**: crítico
- **Âncora de risco**: AS-7 (segredos — `src/ticket_classifier/settings.py` lê a chave de API). Acima do teto de `config` porque manuseia o segredo da chave de API.
- **Perfil**: backend
- **Depende de**: TASK-001
- **Arquivos de produção**:
  - `src/ticket_classifier/settings.py`
- **Arquivos de teste**:
  - `tests/unit/test_settings.py`
- **Wiring permitido**: —
- **Reusa**:
  - pydantic-settings → `BaseSettings`, `SettingsConfigDict(env_prefix="TICKET_")`, `SecretStr`
- **Contrato**:
  - CT-4 — `Settings(BaseSettings)` + `get_settings() -> Settings` (produz)
- **Testes**: unit
- **Descrição**: Campos e defaults exatamente como CT-4. `TICKET_API_KEY` vazio ou só espaços vira `None` (FDBK-93 trata como não configurado). Limiar validado em [0, 1] (API-04). `get_settings()` cria instância nova a cada chamada (sem cache) para os testes.
- **Done when**:
  - [ ] Sem variáveis `TICKET_*`, `get_settings().review_threshold == 0.6`, `api_key is None`, `db_path == Path("var/tickets.db")`, `artifacts_dir == Path("artifacts")`, `retention_days == 90`, `purge_interval_seconds == 86400`
  - [ ] Com `TICKET_REVIEW_THRESHOLD=0.3` o valor é `0.3`; com `1.5` e com `-0.1` `get_settings()` levanta `pydantic.ValidationError`
  - [ ] Com `TICKET_API_KEY=""` e com `TICKET_API_KEY="   "` o campo é `None`
  - [ ] Com `TICKET_API_KEY=s3cr3t`, `repr(settings)` e `str(settings)` não contêm `s3cr3t` e `settings.api_key.get_secret_value() == "s3cr3t"`
  - [ ] `uv run pytest tests/unit/test_settings.py` passa
- **Não fazer**:
  - Não validar nem comparar a chave de API (TASK-025)
  - Não logar valores de configuração

---

### TASK-005 — Normalize text and mask PII
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `DATA-07`, `DATA-08`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-001
- **Arquivos de produção**:
  - `src/ticket_classifier/preprocessing.py`
- **Arquivos de teste**:
  - `tests/unit/test_preprocessing.py`
- **Wiring permitido**: —
- **Reusa**: —
- **Contrato**:
  - CT-5 — `normalize_text`, `mask_pii`, `preprocess_text` (produz)
- **Testes**: unit
- **Descrição**: Implementar DA-6 (plan): normalização (caracteres `Cc` viram espaço, colapso de espaços, strip, caixa preservada) e máscara na ordem EMAIL → URL → PHONE → NUMBER com os marcadores de CT-5. `preprocess_text` é idempotente e é a única rotina usada no treino e na inferência.
- **Done when**:
  - [ ] `preprocess_text("  Hi\t\tthere\n\nfriend \x07 ")` == `"Hi there friend"`
  - [ ] `preprocess_text("Mail john.doe@acme.com now")` == `"Mail [EMAIL] now"`
  - [ ] `preprocess_text("See https://acme.com/x?a=1 and www.acme.org")` == `"See [URL] and [URL]"`
  - [ ] `preprocess_text("Call +1 (555) 123-4567 today")` == `"Call [PHONE] today"` e `preprocess_text("Call 555-123-4567")` == `"Call [PHONE]"`
  - [ ] `preprocess_text("Order 12345678 failed")` == `"Order [NUMBER] failed"` e `preprocess_text("Code 12345 ok")` == `"Code 12345 ok"`
  - [ ] Para todas as entradas acima, `preprocess_text(preprocess_text(x)) == preprocess_text(x)`
  - [ ] `uv run pytest tests/unit/test_preprocessing.py` passa
- **Não fazer**:
  - Não mascarar nomes de pessoas (fora de escopo, LAC-18)
  - Não converter para minúsculas nem remover pontuação

---

### TASK-006 — Add pipeline configuration file and loader
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `DATA-04`
- **Tipo**: config
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-002
- **Arquivos de produção**:
  - `src/ticket_classifier/pipeline_config.py`
  - `configs/pipeline.toml`
- **Arquivos de teste**:
  - `tests/unit/test_pipeline_config.py`
- **Wiring permitido**: —
- **Reusa**:
  - `tomllib` (stdlib)
  - `src/ticket_classifier/errors.py` → `PipelineError`
- **Contrato**:
  - CT-6 — `DataConfig`, `BaselineConfig`, `TransformerConfig`, `PipelineConfig`, `load_pipeline_config(path: Path) -> PipelineConfig` (produz)
  - CT-2 (consome)
- **Testes**: unit
- **Descrição**: Dataclasses congeladas com os campos e valores de `configs/pipeline.toml` da seção 7.5 do plan (`tag_columns` como `tuple[str, ...]`, `c_grid` como `tuple[float, ...]`, caminhos como `Path`). Chave ausente ou soma de ratios diferente de 1.0 (tolerância 1e-9) → `PipelineError` citando a chave.
- **Done when**:
  - [ ] `load_pipeline_config(Path("configs/pipeline.toml"))` devolve `seed == 42`, `data.train_ratio == 0.70`, `data.min_train_examples_per_category == 100`, `transformer.model_name == "distilbert-base-uncased"`, `transformer.max_length == 256`
  - [ ] TOML sem `[transformer]` levanta `PipelineError` com `transformer` na mensagem
  - [ ] TOML com ratios 0.7/0.2/0.2 levanta `PipelineError`
  - [ ] `uv run pytest tests/unit/test_pipeline_config.py` passa
- **Não fazer**:
  - Não ler variáveis de ambiente aqui (isso é `settings.py`)

---

### TASK-007 — Implement label mapping table and loader
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `DATA-02`, `DATA-91`
- **Tipo**: config
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-002
- **Arquivos de produção**:
  - `src/ticket_classifier/data/label_mapping.py`
  - `configs/label_mapping.toml`
- **Arquivos de teste**:
  - `tests/unit/test_label_mapping.py`
- **Wiring permitido**:
  - `src/ticket_classifier/data/__init__.py` (criar vazio se não existir)
- **Reusa**:
  - `tomllib` (stdlib)
  - `src/ticket_classifier/labels.py` → `CATEGORIES`, `PRIORITIES`
  - `src/ticket_classifier/errors.py` → `PipelineError`
- **Contrato**:
  - CT-7 — `UnmappedLabelError`, `LabelMapping`, `load_label_mapping`, `map_category(queue, tags)`, `map_priority(source)` (produz)
  - CT-1, CT-2 (consome)
- **Testes**: unit
- **Descrição**: Decisão LAC-32 = A. Implementar o formato da seção 7.5 do plan (DA-4): regras `[[category.rules]]` em ordem (primeira que casar vence; `queue_in` compara `queue`, `any_tag_in` compara qualquer tag não vazia), senão `[category.queue]`. `queue` ausente de `[category.queue]` ou prioridade ausente de `[priority]` → `UnmappedLabelError` com a mensagem de 8.3. Destino fora de CT-1 → `PipelineError` na carga.
- **Done when**:
  - [ ] Com o `configs/label_mapping.toml` do repositório: `map_category("Billing and Payments", ["Login"]) == "billing"`, `map_category("Technical Support", ["Login"]) == "access"`, `map_category("IT Support", ["Crash"]) == "bug"`, `map_category("Product Support", ["Network"]) == "infrastructure"`, `map_category("Customer Service", ["Feedback"]) == "other"`
  - [ ] `map_priority("very_low") == "low"` e `map_priority("critical") == "high"`
  - [ ] `map_category("Unknown Queue", [])` levanta `UnmappedLabelError` com mensagem `Unmapped source label 'Unknown Queue'. Add it to the label mapping table.`
  - [ ] `map_priority("urgent")` levanta `UnmappedLabelError` com `'urgent'` na mensagem
  - [ ] TOML com destino `"hardware"` levanta `PipelineError` na carga
  - [ ] `uv run pytest tests/unit/test_label_mapping.py` passa
- **Não fazer**:
  - Não ler o CSV de origem nem aplicar filtros (TASK-009)
  - Não adicionar categorias fora de CT-1

---

### TASK-008 — Store and version dataset splits
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `MODEL-90`
- **Tipo**: crud-padrão
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-002
- **Arquivos de produção**:
  - `src/ticket_classifier/data/splits.py`
- **Arquivos de teste**:
  - `tests/unit/test_splits.py`
- **Wiring permitido**:
  - `src/ticket_classifier/data/__init__.py` (criar vazio se não existir)
- **Reusa**:
  - `src/ticket_classifier/errors.py` → `PipelineError`
- **Contrato**:
  - CT-8 — `SPLIT_NAMES`, `EmptySplitError`, `write_splits`, `load_split`, `splits_version` (produz)
  - CT-2 (consome)
- **Testes**: unit
- **Descrição**: Formato da seção 7.2 do plan. `write_splits` grava os três JSONL e `prepare_report.json` (com `splits_version` inserido) num diretório temporário irmão e move cada arquivo com `os.replace`. `load_split` levanta `EmptySplitError` (mensagem MODEL-90 de 8.3) se o arquivo não existe ou tem 0 linhas.
- **Done when**:
  - [ ] Gravar e ler de volta três DataFrames preserva linhas, colunas e ordem (`pandas.testing.assert_frame_equal`)
  - [ ] Duas gravações dos mesmos DataFrames produzem o mesmo `splits_version` e arquivos byte a byte iguais
  - [ ] `load_split(dir, "validation")` com arquivo ausente levanta `EmptySplitError` com mensagem `Split 'validation' is empty or missing. Run data preparation first.`; o mesmo com arquivo vazio
  - [ ] `prepare_report.json` gravado contém a chave `splits_version` igual ao valor devolvido
  - [ ] `uv run pytest tests/unit/test_splits.py` passa
- **Não fazer**:
  - Não fazer a divisão estratificada (TASK-009)

---

### TASK-009 — Implement dataset preparation command
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `DATA-01`, `DATA-03`, `DATA-04`, `DATA-05`, `DATA-06`, `DATA-09`, `DATA-10`, `DATA-11`, `DATA-12`, `DATA-90`, `DATA-92`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-005, TASK-006, TASK-007, TASK-008
- **Arquivos de produção**:
  - `src/ticket_classifier/data/prepare.py`
- **Arquivos de teste**:
  - `tests/support/__init__.py`
  - `tests/support/sample_data.py`
  - `tests/integration/test_prepare.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas registrar o subcomando `prepare` com `--config` e `--mapping`)
  - `README.md` (seção "Dataset": fonte HF, arquivo, revisão, SHA-256, licença CC BY-NC 4.0, aviso de possível origem sintética do risco da seção 15 do plan, comando `curl` para `data/raw/tickets.csv` e `uv run ticket-classifier prepare`)
- **Reusa**:
  - `src/ticket_classifier/preprocessing.py` → `preprocess_text`
  - `src/ticket_classifier/pipeline_config.py` → `load_pipeline_config`, `PipelineConfig`
  - `src/ticket_classifier/data/label_mapping.py` → `load_label_mapping`
  - `src/ticket_classifier/data/splits.py` → `write_splits`
  - scikit-learn → `train_test_split(stratify=..., random_state=seed)`
- **Contrato**:
  - CT-9 — `prepare_dataset(config: PipelineConfig, mapping_path: Path) -> dict[str, Any]`, `SourceDatasetNotFoundError`, `InsufficientExamplesError` (produz)
  - CT-5, CT-6, CT-7, CT-8, CT-28 (consome)
- **Testes**: integration
- **Descrição**: Pipeline na ordem de DA-5. Split: primeiro separa `test` (fração `test_ratio`), depois `validation` do restante (`validation_ratio / (1 - test_ratio)`), ambos estratificados por `category` com `random_state=seed`. Toda validação (DATA-92, DATA-91, DATA-10) antes de gravar. Relatório da seção 7.2 (inclui `source_sha256`). `tests/support/sample_data.py` expõe: `SAMPLE_PER_CATEGORY = 160`; `make_source_rows(per_category: int = SAMPLE_PER_CATEGORY, seed: int = 7) -> list[dict[str, str]]` (colunas do CSV de origem; cada categoria alcançável pelo `configs/label_mapping.toml`; mais 10 linhas `language="de"`, 5 duplicatas exatas, 3 com `subject` vazio, linhas com e-mail e telefone); `write_source_csv(path: Path, rows) -> Path`; `write_pipeline_config(tmp_path: Path, source_path: Path, model_name: str = "unused", epochs: int = 1, max_length: int = 32, batch_size: int = 16) -> Path` (`processed_dir` em `tmp_path`, `c_grid = [1.0]`, `num_threads = 1`); `prepare_sample(tmp_path: Path, model_name: str = "unused") -> PipelineConfig`.
- **Done when**:
  - [ ] `prepare_sample(tmp_path)` grava `train.jsonl`, `validation.jsonl`, `test.jsonl` com exatamente as chaves `title`, `description`, `category`, `priority` em cada linha
  - [ ] Relatório: `discarded.non_english == 10`, `discarded.duplicate == 5`, `discarded.empty_text == 3`, e contagem por `category` e `priority` em cada split
  - [ ] Tamanhos dos splits dentro de ±1 registro de 70/15/15 do total após descarte; proporção de cada `category` em cada split difere no máximo 1 pp da proporção total
  - [ ] Duas execuções com a mesma semente geram arquivos byte a byte iguais; semente diferente gera `test.jsonl` diferente
  - [ ] Nenhum texto gravado contém `@` de e-mail ou o telefone da fixture; contém `[EMAIL]` e `[PHONE]`
  - [ ] Origem ausente: `SourceDatasetNotFoundError` com mensagem DATA-92 de 8.3 e `processed_dir` sem arquivos
  - [ ] Fixture com uma `queue` não mapeada: `UnmappedLabelError` e `processed_dir` sem arquivos
  - [ ] Fixture com 60 linhas de `access`: `InsufficientExamplesError` com mensagem `Category 'access' has <n> training examples; minimum is 100.` e `processed_dir` sem arquivos
  - [ ] `uv run ticket-classifier prepare --config <tmp>/pipeline.toml` devolve 0; com origem ausente devolve 1 e imprime a mensagem DATA-92 em stderr
  - [ ] `README.md` contém a seção `Dataset` com a URL de download e o SHA-256 `9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635`
  - [ ] `uv run pytest tests/integration/test_prepare.py` passa
- **Não fazer**:
  - Não baixar o dataset por código
  - Não usar detector de idioma (DA-5 usa a coluna `language`)
  - Não commitar dados em `data/`

---

### TASK-010 — Compute classification metrics
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `MODEL-03`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-003
- **Arquivos de produção**:
  - `src/ticket_classifier/evaluation/metrics.py`
- **Arquivos de teste**:
  - `tests/unit/test_metrics.py`
- **Wiring permitido**:
  - `src/ticket_classifier/evaluation/__init__.py` (criar vazio)
- **Reusa**:
  - scikit-learn → `accuracy_score`, `precision_recall_fscore_support(labels=..., zero_division=0)`, `confusion_matrix(labels=...)`
  - `src/ticket_classifier/models/base.py` → `TicketClassifier`, `top_label`
- **Contrato**:
  - CT-10 — `ClassMetrics`, `TargetMetrics`, `compute_target_metrics`, `evaluate_classifier` (produz)
  - CT-1, CT-3 (consome)
- **Testes**: unit
- **Descrição**: Métricas por alvo com a lista de rótulos fixa (CT-1), inclusive rótulos sem suporte. `evaluate_classifier` chama `predict_proba` em lotes de 64 e usa `top_label` por alvo. `to_dict`/`from_dict` fazem ida e volta sem perda.
- **Done when**:
  - [ ] Caso conhecido (`y_true=["a","a","b","c"]`, `y_pred=["a","b","b","c"]`, `labels=["a","b","c"]`): `accuracy == 0.75`, `macro_f1` igual a `sklearn.metrics.f1_score(..., average="macro")`, `confusion_matrix == [[1,1,0],[0,1,0],[0,0,1]]`
  - [ ] Rótulo sem suporte aparece em `per_class` com `support == 0` e `f1 == 0.0`
  - [ ] `TargetMetrics.from_dict(m.to_dict()) == m`
  - [ ] `evaluate_classifier` com um classificador fake (implementa CT-3) devolve chaves `category` e `priority`
  - [ ] `uv run pytest tests/unit/test_metrics.py` passa
- **Não fazer**:
  - Não gerar relatórios nem arquivos (TASK-017)

---

### TASK-011 — Implement model version registry
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `MODEL-08`, `OPS-05`, `OPS-06`
- **Tipo**: crud-padrão
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-002
- **Arquivos de produção**:
  - `src/ticket_classifier/registry.py`
- **Arquivos de teste**:
  - `tests/unit/test_registry.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas registrar o subcomando `list-versions`)
- **Reusa**:
  - `src/ticket_classifier/errors.py` → `PipelineError`
  - `fcntl.flock`, `os.replace` (stdlib)
- **Contrato**:
  - CT-11 — `ModelVersion`, `ModelRegistry` (produz)
  - CT-2, CT-28 (consome)
- **Testes**: unit
- **Descrição**: Registro JSON da seção 7.3 do plan (DA-10). Escrita sob `flock` em `registry.lock` + arquivo temporário + `os.replace`; leitura sem lock; arquivo ausente = registro vazio. `promote` aplica as transições da seção 7 do spec. `list-versions` imprime as colunas de OPS-05 (Macro F1 de `test` com 4 casas, `promoted` = `yes`/`no`); o settings não é usado aqui: o subcomando recebe `--artifacts-dir` com default `artifacts`.
- **Done when**:
  - [ ] `new_version_id("baseline")` casa com `^baseline-\d{8}T\d{6}Z-[0-9a-f]{8}$` e duas chamadas seguidas devolvem valores diferentes
  - [ ] `register` + `get` devolvem `ModelVersion` igual ao gravado; `list_versions` ordena por `created_at`
  - [ ] Promover A e depois B deixa A `retired`, B `promoted` e exatamente uma versão `promoted`
  - [ ] `promote` de versão `retired` ou inexistente levanta `PipelineError` e não altera o arquivo
  - [ ] `get_promoted()` em diretório sem `registry.json` devolve `None`
  - [ ] `uv run ticket-classifier list-versions --artifacts-dir <tmp>` imprime `version_id`, `kind`, os dois Macro F1 de test, `created_at` e `yes`/`no` para cada versão
  - [ ] `uv run pytest tests/unit/test_registry.py` passa
- **Não fazer**:
  - Não usar MLflow Model Registry (DA-10)
  - Não implementar rollback de promoção (fora de escopo)

---

### TASK-012 — Train TF-IDF and Logistic Regression baseline
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `MODEL-01`, `MODEL-07`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-003, TASK-006
- **Arquivos de produção**:
  - `src/ticket_classifier/models/baseline.py`
- **Arquivos de teste**:
  - `tests/unit/test_baseline.py`
- **Wiring permitido**: —
- **Reusa**:
  - scikit-learn → `Pipeline`, `TfidfVectorizer`, `LogisticRegression`, `f1_score(average="macro")`
  - `joblib` → `dump`, `load`
  - `src/ticket_classifier/models/base.py` → `ClassProbabilities`
- **Contrato**:
  - CT-12 — `BaselineClassifier`, `train_baseline(train, validation, config, seed)` (produz)
  - CT-1, CT-3, CT-6 (consome)
- **Testes**: unit
- **Descrição**: Um `Pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, ngram_max), min_df, max_features, sublinear_tf=True), LogisticRegression(C, class_weight="balanced", max_iter=2000, random_state=seed))` por alvo, texto = `title + " " + description`. Para cada alvo, treina com `train` para cada `C` de `c_grid` e escolhe o maior Macro F1 em `validation` (empate: primeiro do grid). Probabilidades devolvidas com todas as chaves de CT-1. Salva `category.joblib`, `priority.joblib`, `baseline_meta.json` (DA-16, seção 7.4).
- **Done when**:
  - [ ] Com DataFrames sintéticos (5 categorias × 20 linhas em train e 5 × 6 em validation) `train_baseline` devolve hiperparâmetros com o `C` escolhido por alvo
  - [ ] `predict_proba` devolve, por ticket, dicts com exatamente as chaves de `CATEGORIES` e `PRIORITIES` e soma entre 0.999 e 1.001
  - [ ] Dois treinos com a mesma semente produzem probabilidades idênticas (`==`) no mesmo lote
  - [ ] `save` + `BaselineClassifier.load` reproduzem as mesmas probabilidades
  - [ ] A assinatura de `train_baseline` não recebe o split `test` (verificado por `inspect.signature`)
  - [ ] `uv run pytest tests/unit/test_baseline.py` passa
- **Não fazer**:
  - Não registrar versão nem logar no MLflow (TASK-016)
  - Não reajustar com train + validation juntos

---

### TASK-013 — Build multi-head Transformer classifier
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `MODEL-02`, `API-06`
- **Tipo**: lógica-negócio
- **Risco**: alto
- **Âncora de risco**: AS-8 (integração nova com o Hugging Face Hub — `src/ticket_classifier/models/transformer.py`)
- **Perfil**: backend
- **Depende de**: TASK-003
- **Arquivos de produção**:
  - `src/ticket_classifier/models/transformer.py`
- **Arquivos de teste**:
  - `tests/support/tiny_model.py`
  - `tests/unit/test_transformer.py`
- **Wiring permitido**:
  - `tests/support/__init__.py` (criar vazio se não existir)
- **Reusa**:
  - transformers → `AutoModel.from_pretrained`, `AutoTokenizer.from_pretrained`, `save_pretrained`
  - `src/ticket_classifier/models/base.py` → `ClassProbabilities`
- **Contrato**:
  - CT-13 — `TransformerClassifier.from_pretrained(model_name_or_path, max_length, seed)`, `.load(directory)`, `.save`, `.predict_proba`, `.encode` (produz)
  - CT-1, CT-3 (consome)
- **Testes**: unit
- **Descrição**: DA-7: módulo `torch.nn.Module` com encoder `AutoModel`, dropout 0.1 e duas cabeças `Linear(hidden_size, 5)` / `Linear(hidden_size, 3)` sobre o primeiro token. `encode` usa par (title, description), `truncation="longest_first"`, `max_length`, `padding=True`. `predict_proba` em `eval()` + `torch.inference_mode()`, lotes de 32, softmax, protegido por `threading.Lock` (DA-12). Layout de `save` da seção 7.4; `load` lê só do diretório (`heads.pt` com `weights_only=True`). `tests/support/tiny_model.py` expõe `build_tiny_model(directory: Path) -> Path` (DA-9: `BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)`, vocabulário local com tokens especiais, letras, dígitos e `##`-sufixos).
- **Done when**:
  - [ ] `TransformerClassifier.from_pretrained(str(build_tiny_model(tmp_path)), max_length=32, seed=1)` funciona com `HF_HUB_OFFLINE=1`
  - [ ] `predict_proba` devolve dicts com todas as chaves de CT-1 e soma entre 0.999 e 1.001
  - [ ] Ticket com description de 5000 caracteres gera `input_ids` com comprimento `<= 32` em `encode` e `predict_proba` não levanta erro (API-06)
  - [ ] `save` + `load` em outro diretório reproduzem as mesmas probabilidades (`torch.allclose`, `atol=1e-6`)
  - [ ] 8 threads chamando `predict_proba` ao mesmo tempo devolvem o mesmo resultado que chamadas sequenciais
  - [ ] `uv run pytest tests/unit/test_transformer.py` passa sem acesso à rede
- **Não fazer**:
  - Não escrever o loop de treino (TASK-014)
  - Não baixar modelo nos testes

---

### TASK-014 — Implement deterministic Transformer fine-tuning loop
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `MODEL-02`, `MODEL-06`, `MODEL-07`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-006, TASK-010, TASK-013
- **Arquivos de produção**:
  - `src/ticket_classifier/models/transformer_training.py`
- **Arquivos de teste**:
  - `tests/integration/test_transformer_training.py`
- **Wiring permitido**: —
- **Reusa**:
  - `src/ticket_classifier/models/transformer.py` → `TransformerClassifier`
  - `src/ticket_classifier/evaluation/metrics.py` → `compute_target_metrics`
  - `tests/support/tiny_model.py` → `build_tiny_model`
  - transformers → `get_linear_schedule_with_warmup`
- **Contrato**:
  - CT-14 — `set_determinism(seed, num_threads)`, `train_transformer(train, validation, config, seed)` (produz)
  - CT-6, CT-10, CT-13 (consome)
- **Testes**: integration
- **Descrição**: DA-8 e DA-16: `set_determinism` antes de criar o modelo; `AdamW(lr, weight_decay)`, schedule linear com `warmup_ratio`; perda = CE(category, pesos) + CE(priority, pesos), pesos = inverso da frequência no train; ao fim de cada época avalia em `validation` e guarda cópia do `state_dict` com a maior média dos dois Macro F1; devolve o modelo com esse estado. Só recebe `train` e `validation` (MODEL-07).
- **Done when**:
  - [ ] Com modelo minúsculo, `epochs=2`, `max_length=32` e DataFrames sintéticos (5 categorias × 20 linhas), `train_transformer` termina e o dict devolvido contém `best_epoch` em {1, 2} e os hiperparâmetros
  - [ ] Dois treinos com a mesma semente geram probabilidades iguais (`torch.allclose`, `atol=1e-6`) no mesmo lote
  - [ ] A assinatura de `train_transformer` não recebe o split `test` (verificado por `inspect.signature`)
  - [ ] `uv run pytest tests/integration/test_transformer_training.py` passa em CPU sem rede
- **Não fazer**:
  - Não usar `transformers.Trainer`
  - Não salvar artefatos nem registrar versão (TASK-016)

---

### TASK-015 — Log training runs to MLflow
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `OPS-04`
- **Tipo**: integração-externa
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-011
- **Arquivos de produção**:
  - `src/ticket_classifier/tracking.py`
- **Arquivos de teste**:
  - `tests/unit/test_tracking.py`
- **Wiring permitido**: —
- **Reusa**:
  - mlflow → `set_tracking_uri`, `set_experiment`, `start_run`, `log_params`, `log_metrics`, `log_dict`, `log_artifacts`
  - `src/ticket_classifier/registry.py` → `ModelVersion`
- **Contrato**:
  - CT-15 — `log_training_run(version, model_dir, experiment_name="support-ticket-classifier") -> str` (produz)
  - CT-11 (consome)
- **Testes**: unit
- **Descrição**: URI de `MLFLOW_TRACKING_URI` ou `sqlite:///mlflow.db`. Parâmetros: hiperparâmetros achatados, `seed`, `splits_version`, `kind`, `version_id`, `feedback_rows`. Métricas `{split}_{target}_accuracy`, `{split}_{target}_macro_f1` e `{split}_{target}_{label}_{precision|recall|f1}`. Artefatos: `confusion_matrix_{split}_{target}.json` e o diretório do modelo em `model/`.
- **Done when**:
  - [ ] Com `MLFLOW_TRACKING_URI=sqlite:///<tmp_path>/mlflow.db`, o `run_id` devolvido existe em `mlflow.get_run` com parâmetro `seed`, `splits_version`, `version_id` e métrica `test_category_macro_f1`
  - [ ] O run tem os artefatos `confusion_matrix_test_category.json` e `model/`
  - [ ] `uv run pytest tests/unit/test_tracking.py` passa sem rede
- **Não fazer**:
  - Não importar `tracking` em nenhum módulo de `api/`, `services/`, `storage/`, `registry.py` ou `models/loader.py`

---

### TASK-016 — Orchestrate training, evaluation and registration
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `MODEL-06`, `MODEL-07`, `MODEL-08`, `MODEL-09`, `MODEL-90`, `OPS-04`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-004, TASK-008, TASK-009, TASK-010, TASK-011, TASK-012, TASK-013, TASK-014, TASK-015
- **Arquivos de produção**:
  - `src/ticket_classifier/training.py`
  - `src/ticket_classifier/models/loader.py`
- **Arquivos de teste**:
  - `tests/integration/test_training.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas registrar `train-baseline` e `train-transformer`, ambos com `--config`)
  - `README.md` (seção "Training": comandos, onde ficam artefatos, `list-versions`, aviso de que o treino real baixa `distilbert-base-uncased`)
- **Reusa**:
  - `src/ticket_classifier/data/splits.py` → `load_split`, `splits_version`
  - `src/ticket_classifier/models/baseline.py` → `train_baseline`, `BaselineClassifier.load`
  - `src/ticket_classifier/models/transformer_training.py` → `train_transformer`
  - `src/ticket_classifier/models/transformer.py` → `TransformerClassifier.load`
  - `src/ticket_classifier/evaluation/metrics.py` → `evaluate_classifier`
  - `src/ticket_classifier/registry.py` → `ModelRegistry`, `ModelVersion`
  - `src/ticket_classifier/tracking.py` → `log_training_run`
  - `src/ticket_classifier/settings.py` → `get_settings`
  - `tests/support/sample_data.py` → `prepare_sample`
  - `tests/support/tiny_model.py` → `build_tiny_model`
- **Contrato**:
  - CT-16 — `TrainingOutcome`, `train_and_register(kind, config, settings, extra_train=None)`, `load_classifier(version, artifacts_dir)` (produz)
  - CT-4, CT-6, CT-8, CT-9, CT-10, CT-11, CT-12, CT-13, CT-14, CT-15, CT-28 (consome)
- **Testes**: integration
- **Descrição**: Ordem: carregar os três splits (MODEL-90 antes de treinar) → `train = concat(train, extra_train)` se houver → treinar → avaliar `validation` e `test` → salvar em `ModelRegistry.model_dir(id)` → `log_training_run` → `register` → promover se `kind == "transformer"`, `extra_train is None` e `get_promoted() is None` (MODEL-09). `ModelVersion.training_rows` e `feedback_rows = len(extra_train or [])`. `load_classifier` despacha por `kind`. CLI imprime `Registered model version '<id>' (<kind>). Promoted: yes|no.`
- **Done when**:
  - [ ] Com `prepare_sample` e `MLFLOW_TRACKING_URI` em `tmp_path`: `train_and_register("baseline", ...)` registra versão `baseline` não promovida, com métricas `validation` e `test` dos dois alvos, `splits_version`, `seed` e hiperparâmetros
  - [ ] `train_and_register("transformer", ...)` com o modelo minúsculo registra e promove (primeira transformer); segunda chamada registra sem promover
  - [ ] Duas execuções do baseline com a mesma configuração geram métricas de test idênticas (`==`); duas do transformer geram Macro F1 com diferença `<= 0.01` por alvo
  - [ ] Com `test.jsonl` vazio, levanta `EmptySplitError` e nenhuma versão é registrada nem diretório de modelo criado
  - [ ] Teste espiona `train_baseline`/`train_transformer` e prova que nenhum registro de `test.jsonl` é passado a eles
  - [ ] `load_classifier(version, artifacts_dir)` devolve classificador cujo `predict_proba` iguala o do modelo recém-treinado
  - [ ] `uv run ticket-classifier train-baseline --config <tmp>/pipeline.toml` devolve 0 e imprime `Registered model version`
  - [ ] `uv run pytest tests/integration/test_training.py` passa em CPU sem rede
- **Não fazer**:
  - Não implementar gate de promoção do retreino (TASK-027)
  - Não importar `training`/`tracking` a partir de `models/loader.py`

---

### TASK-017 — Generate baseline versus Transformer comparison report

- **Requisito**: `MODEL-04`, `MODEL-05`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-010, TASK-011
- **Arquivos de produção**:
  - `src/ticket_classifier/evaluation/report.py`
- **Arquivos de teste**:
  - `tests/unit/test_report.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas registrar `compare` com `--baseline`, `--transformer`, `--artifacts-dir`)
  - `README.md` (seção "Results": como gerar o relatório, caminho `artifacts/reports/comparison.md`, regra PASS/FAIL)
- **Reusa**:
  - `src/ticket_classifier/registry.py` → `ModelRegistry.latest`, `ModelRegistry.get`
  - `src/ticket_classifier/evaluation/metrics.py` → `TargetMetrics.from_dict`
- **Contrato**:
  - CT-17 — `ComparisonReport`, `build_comparison`, `render_markdown`, `write_report` (produz)
  - CT-10, CT-11, CT-28 (consome)
- **Testes**: unit
- **Descrição**: Lado a lado, por alvo, todas as métricas de MODEL-03 do split `test` (accuracy, P/R/F1 por classe, Macro F1, confusion matrix). `macro_f1_delta_pp[alvo] = round((transformer - baseline) * 100, 2)`. Veredito (LAC-05): `PASS` se `t_cat >= b_cat + 0.05 - 1e-9` e `t_pri >= b_pri - 1e-9`, senão `FAIL`. Splits diferentes → `PipelineError` (8.3). Grava `comparison.md` e `comparison.json` em `artifacts/reports/`.
- **Done when**:
  - [ ] Versões fake com Macro F1 (cat, pri) baseline (0.60, 0.50) e transformer (0.65, 0.50) → `PASS` e deltas `{"category": 5.0, "priority": 0.0}`
  - [ ] Transformer (0.649, 0.60) → `FAIL`; transformer (0.70, 0.49) → `FAIL`
  - [ ] `render_markdown` contém `Verdict: PASS` ou `Verdict: FAIL`, as duas matrizes de confusão por alvo e uma linha por classe com precision/recall/F1 dos dois modelos
  - [ ] `splits_version` diferente levanta `PipelineError` com `Model versions were evaluated on different splits.`
  - [ ] `uv run ticket-classifier compare --artifacts-dir <tmp>` sem versão `transformer` devolve 1 com `No 'transformer' model version registered. Train it first.`
  - [ ] `uv run pytest tests/unit/test_report.py` passa
- **Não fazer**:
  - Não reavaliar modelos: usar só as métricas gravadas no registro

---

### TASK-018 — Analyze errors of the promoted model

- **Requisito**: `MODEL-10`, `MODEL-11`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-004, TASK-016
- **Arquivos de produção**:
  - `src/ticket_classifier/evaluation/error_analysis.py`
- **Arquivos de teste**:
  - `tests/unit/test_error_analysis.py`
  - `tests/integration/test_error_analysis_cli.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas registrar `analyze-errors` com `--version` e `--config`)
- **Reusa**:
  - `src/ticket_classifier/models/loader.py` → `load_classifier`
  - `src/ticket_classifier/models/base.py` → `top_label`
  - `src/ticket_classifier/data/splits.py` → `load_split`
  - `src/ticket_classifier/registry.py` → `ModelRegistry.get_promoted`
  - `src/ticket_classifier/settings.py` → `get_settings` (limiar)
  - `tests/support/sample_data.py` → `prepare_sample`
- **Contrato**:
  - CT-18 — `ErrorAnalysis`, `analyze_errors`, `write_error_analysis` (produz)
  - CT-3, CT-4, CT-8, CT-10, CT-11, CT-16, CT-28 (consome)
- **Testes**: unit
- **Descrição**: Erro = `category` ou `priority` previsto diferente do real; cada item traz título, descrição, rótulos real e previsto e confianças dos dois alvos. Top-5 pares (real, previsto) de `category` entre erros de categoria, ordenados por contagem desc e depois pela ordem de CT-1. `needs_review` com a mesma regra de LAC-22 e limiar de `Settings`. Accuracy de `category` separada por `needs_review` (`None` se o grupo está vazio). Grava `error_analysis_<id>.{md,json}` em `artifacts/reports/`.
- **Done when**:
  - [ ] Classificador fake com saídas controladas: lista de erros com os campos acima e contagem exata; top-5 pares na ordem esperada
  - [ ] Limiar 0.0 → `needs_review_fraction == 0.0` e `category_accuracy_needs_review is None`; limiar 1.0 → fração `1.0`
  - [ ] Accuracy por grupo confere com cálculo manual no caso fake
  - [ ] CLI `analyze-errors` sem versão promovida devolve 1 com `No promoted model version. Train a transformer first.`; com baseline promovido em `tmp_path` devolve 0 e cria os dois arquivos
  - [ ] `uv run pytest tests/unit/test_error_analysis.py tests/integration/test_error_analysis_cli.py` passa
- **Não fazer**:
  - Não alterar o limiar em vigor nem a configuração

---

### TASK-019 — Create SQLite schema and prediction repository
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `API-07`, `API-95`
- **Tipo**: crud-padrão
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-002
- **Arquivos de produção**:
  - `src/ticket_classifier/storage/database.py`
  - `src/ticket_classifier/storage/predictions.py`
- **Arquivos de teste**:
  - `tests/unit/test_storage_predictions.py`
- **Wiring permitido**:
  - `src/ticket_classifier/storage/__init__.py` (criar vazio)
- **Reusa**:
  - `sqlite3` (stdlib)
  - `src/ticket_classifier/labels.py` → `CATEGORIES`, `PRIORITIES` (CHECKs)
- **Contrato**:
  - CT-19 — `connect`, `init_schema`, `transaction`, `utc_now`, `to_iso`, `PredictionRecord`, `insert_prediction`, `prediction_exists` (produz)
  - CT-1 (consome)
- **Testes**: unit
- **Descrição**: Esquema da seção 7.1 do plan (as duas tabelas e índices, idempotente), PRAGMAs de CT-19 (DA-11). `insert_prediction` grava os campos recebidos; o repositório não mascara (quem mascara é o serviço) e não aceita campo de texto original. Datas no formato fixo de 7.1.
- **Done when**:
  - [ ] `init_schema` chamado duas vezes no mesmo banco não levanta erro
  - [ ] `insert_prediction` + consulta direta devolvem os mesmos valores, com `created_at` no formato `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`
  - [ ] Inserir `category="hardware"` levanta `sqlite3.IntegrityError`
  - [ ] 20 threads, cada uma com sua conexão, inserem Previsões distintas ao mesmo tempo: 20 linhas gravadas, 20 `prediction_id` distintos, cada texto no seu `prediction_id`
  - [ ] `PRAGMA foreign_keys` devolve 1 e `PRAGMA journal_mode` devolve `wal` numa conexão de `connect`
  - [ ] `uv run pytest tests/unit/test_storage_predictions.py` passa
- **Não fazer**:
  - Não criar funções de feedback ou expurgo (TASK-020, TASK-021)
  - Não adicionar ORM

---

### TASK-020 — Implement append-only feedback repository
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `FDBK-01`, `FDBK-02`, `FDBK-03`, `FDBK-04`, `FDBK-94`
- **Tipo**: crud-padrão
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-019
- **Arquivos de produção**:
  - `src/ticket_classifier/storage/feedback.py`
- **Arquivos de teste**:
  - `tests/unit/test_storage_feedback.py`
- **Wiring permitido**: —
- **Reusa**:
  - `src/ticket_classifier/storage/database.py` → `transaction`, `utc_now`, `to_iso`
  - `src/ticket_classifier/storage/predictions.py` → `insert_prediction`, `PredictionRecord` (nos testes)
- **Contrato**:
  - CT-20 — `PredictionNotFoundError`, `FeedbackRecord`, `insert_feedback`, `FeedbackTrainingRow`, `list_current_feedback` (produz)
  - CT-19 (consome)
- **Testes**: unit
- **Descrição**: `insert_feedback` numa transação `BEGIN IMMEDIATE`: `IntegrityError` de FK ou Previsão ausente → `PredictionNotFoundError`; nunca apaga nem altera Feedback anterior. `list_current_feedback` devolve, por Previsão com Feedback, o de maior `(received_at, seq)` com os textos mascarados da Previsão, ordenado por `prediction_id`.
- **Done when**:
  - [ ] Dois Feedbacks para a mesma Previsão ficam gravados (2 linhas) e `list_current_feedback` devolve só o mais recente
  - [ ] Feedback com rótulos iguais aos previstos é gravado como os demais
  - [ ] `received_at` iguais: vence o de maior `seq`
  - [ ] `prediction_id` inexistente levanta `PredictionNotFoundError` e a tabela `feedback` continua com 0 linhas
  - [ ] 10 threads gravando Feedback para o mesmo `prediction_id` ao mesmo tempo: 10 linhas, `feedback_id` distintos, vigente = maior `(received_at, seq)`
  - [ ] `uv run pytest tests/unit/test_storage_feedback.py` passa
- **Não fazer**:
  - Não implementar autenticação nem rota HTTP (TASK-025)

---

### TASK-021 — Purge expired predictions without feedback
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `OPS-07`, `OPS-90`
- **Tipo**: crud-padrão
- **Risco**: crítico
- **Âncora de risco**: AS-3, AS-5 (dado pessoal e retenção LGPD — `src/ticket_classifier/storage/purge.py`). Acima do teto de `crud-padrão` porque apaga dado operacional por regra de retenção legal.
- **Perfil**: backend
- **Depende de**: TASK-004, TASK-020
- **Arquivos de produção**:
  - `src/ticket_classifier/storage/purge.py`
- **Arquivos de teste**:
  - `tests/unit/test_purge.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas registrar o subcomando `purge`)
- **Reusa**:
  - `src/ticket_classifier/storage/database.py` → `connect`, `transaction`, `utc_now`, `to_iso`
  - `src/ticket_classifier/storage/feedback.py` → `insert_feedback` (nos testes)
  - `src/ticket_classifier/settings.py` → `get_settings`
- **Contrato**:
  - CT-21 — `purge_expired_predictions(conn, now, retention_days=90) -> int`, `PURGE_MESSAGE` (produz)
  - CT-4, CT-19, CT-20, CT-28 (consome)
- **Testes**: unit
- **Descrição**: Um único `DELETE` dentro de `transaction` (BEGIN IMMEDIATE): apaga Previsões com `created_at < to_iso(now - timedelta(days=retention_days))` e sem nenhuma linha em `feedback`. A FK `ON DELETE RESTRICT` + transação garantem OPS-90. Subcomando `purge` usa `Settings.db_path` e `retention_days` e imprime `PURGE_MESSAGE`.
- **Done when**:
  - [ ] Previsões com 89, 90 e 91 dias sem Feedback e uma de 91 dias com Feedback: depois do expurgo só a de 91 dias sem Feedback sumiu; função devolve 1
  - [ ] Nenhum Feedback é apagado (contagem igual antes e depois)
  - [ ] Thread A segura transação de expurgo enquanto thread B tenta `insert_feedback` para a Previsão sendo apagada: ao fim, ou a Previsão existe com o Feedback, ou B recebeu `PredictionNotFoundError` e não há Feedback órfão (`SELECT` com `LEFT JOIN` devolve 0 órfãos)
  - [ ] `uv run ticket-classifier purge` com `TICKET_DB_PATH` em `tmp_path` devolve 0 e imprime `Purge finished: <n> predictions without feedback older than 90 days deleted.`
  - [ ] `uv run pytest tests/unit/test_purge.py` passa
- **Não fazer**:
  - Não agendar a execução automática (TASK-026)
  - Não apagar Previsão com Feedback nem Feedback

---

### TASK-022 — Define API schemas and validation error format
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `API-90`, `API-91`, `API-92`, `API-93`, `FDBK-92`
- **Tipo**: crud-padrão
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-002
- **Arquivos de produção**:
  - `src/ticket_classifier/api/schemas.py`
  - `src/ticket_classifier/api/errors.py`
- **Arquivos de teste**:
  - `tests/unit/test_api_errors.py`
- **Wiring permitido**:
  - `src/ticket_classifier/api/__init__.py` (criar vazio)
- **Reusa**:
  - pydantic → `StrictStr`, `field_validator`, `PydanticCustomError`
  - fastapi → `RequestValidationError`, `JSONResponse`
  - `src/ticket_classifier/labels.py` → `Category`, `Priority`, `CATEGORIES`, `PRIORITIES`
- **Contrato**:
  - CT-22 — modelos de `api/schemas.py`, `error_response`, `validation_message`, `register_error_handlers` (produz)
  - CT-1 (consome)
- **Testes**: unit
- **Descrição**: Modelos e formato `{"code", "message"}` da seção 8.1 do plan. `PredictRequest` faz strip, erro customizado `blank` e `too_long` (com `max`). `validation_message` escolhe o primeiro erro na ordem corpo → campos declarados e traduz: `missing` → required (exceto `category`/`priority`, que usam one-of); `literal_error` → one-of com valores de CT-1 separados por `, `; `json_invalid`, `model_attributes_type`, `dict_type`, `string_type` → `Invalid request body.`; `blank`/`too_long` → mensagens de 8.3. Testes montam um `FastAPI()` mínimo com rotas usando os modelos.
- **Done when**:
  - [ ] Corpo sem `title` → 422 `{"code": "VALIDATION_ERROR", "message": "Field 'title' is required."}`
  - [ ] `title` com 201 caracteres → `Field 'title' must be between 1 and 200 characters.`; `description` com 5001 → `... between 1 and 5000 characters.`; `title` de 200 caracteres com espaços extras nas pontas é aceito
  - [ ] `title = "   "` → `Field 'title' must not be blank.`
  - [ ] Corpo `not json`, corpo `[]` e `title = 5` → `Invalid request body.`
  - [ ] `FeedbackRequest` com `category = "hardware"` ou sem `category` → `Field 'category' must be one of: access, infrastructure, billing, bug, other.`; `priority` inválida → `Field 'priority' must be one of: low, medium, high.`
  - [ ] Nenhuma resposta 422 contém a chave `detail`
  - [ ] `uv run pytest tests/unit/test_api_errors.py` passa
- **Não fazer**:
  - Não criar as rotas reais (TASK-023 a TASK-025)

---

### TASK-023 — Create API factory with model loading and health endpoint
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `OPS-01`, `API-09`, `API-94`, `FDBK-93`
- **Tipo**: crud-padrão
- **Risco**: alto
- **Âncora de risco**: AS-6 (endpoint público sem autenticação — `src/ticket_classifier/api/routes_health.py`)
- **Perfil**: backend
- **Depende de**: TASK-004, TASK-009, TASK-011, TASK-016, TASK-019, TASK-022
- **Arquivos de produção**:
  - `src/ticket_classifier/api/app.py`
  - `src/ticket_classifier/api/routes_health.py`
- **Arquivos de teste**:
  - `tests/support/api_fixtures.py`
  - `tests/integration/test_api_health.py`
- **Wiring permitido**:
  - `README.md` (seções "Run the API" com o comando `uvicorn` da seção 3 do plan e "Configuration" com todas as variáveis `TICKET_*` e `MLFLOW_TRACKING_URI` da seção 11 do plan)
- **Reusa**:
  - `src/ticket_classifier/settings.py` → `get_settings`, `Settings`
  - `src/ticket_classifier/registry.py` → `ModelRegistry.get_promoted`
  - `src/ticket_classifier/models/loader.py` → `load_classifier`
  - `src/ticket_classifier/storage/database.py` → `connect`, `init_schema`
  - `src/ticket_classifier/api/errors.py` → `register_error_handlers`
  - `src/ticket_classifier/api/schemas.py` → `HealthResponse`
  - `tests/support/sample_data.py` → `prepare_sample`
- **Contrato**:
  - CT-23 — `AppState`, `create_app(settings=None)`, `get_app_state`, `get_connection`, `GET /health` (produz)
  - CT-3, CT-4, CT-9, CT-11, CT-16, CT-19, CT-22 (consome)
- **Testes**: integration
- **Descrição**: Lifespan (DA-12, LAC-21, LAC-24): `init_schema`; carrega a promovida uma vez (qualquer exceção → `classifier=None` e log de erro); se `api_key is None`, log WARNING `API key not configured: /feedback will reject all requests.`; guarda `AppState` em `app.state.ctx`. `/health` devolve 200 `{"status": "ok", "model_version": id}` ou 503 `{"status": "unavailable", "model_version": null}`, sem chave. Sem objeto `app` global em nível de módulo. `tests/support/api_fixtures.py` expõe `build_promoted_baseline(tmp_path: Path) -> Settings` (prepara amostra, treina baseline com `train_and_register`, promove via `ModelRegistry.promote`, devolve `Settings` com `artifacts_dir`, `db_path` em `tmp_path` e `api_key="test-key"`).
- **Done when**:
  - [ ] Com baseline promovido: `GET /health` → 200 com `model_version` igual ao `version_id` promovido, sem cabeçalho de chave
  - [ ] Sem `registry.json`: a app sobe e `GET /health` → 503 `{"status": "unavailable", "model_version": null}`
  - [ ] Com artefato da promovida apagado do disco: a app sobe e `/health` → 503
  - [ ] Promover outra versão com a app rodando não muda `/health`; nova `create_app` passa a devolver o novo `model_version`
  - [ ] Com `api_key=None`, o log de inicialização contém `API key not configured: /feedback will reject all requests.` (`caplog`)
  - [ ] `import ticket_classifier.api.app` não importa `mlflow` (`"mlflow" not in sys.modules` num subprocesso limpo)
  - [ ] `uv run pytest tests/integration/test_api_health.py` passa
- **Não fazer**:
  - Não criar `/predict` nem `/feedback` (TASK-024, TASK-025)
  - Não recarregar modelo em tempo de execução (LAC-24)

---

### TASK-024 — Implement POST /predict with review flag and persistence
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `API-01`, `API-02`, `API-03`, `API-04`, `API-05`, `API-06`, `API-07`, `API-08`, `API-90`, `API-91`, `API-92`, `API-93`, `API-94`, `API-95`, `API-96`
- **Tipo**: lógica-negócio
- **Risco**: crítico
- **Âncora de risco**: AS-3, AS-6 (texto de ticket com PII gravado e endpoint público — `src/ticket_classifier/services/prediction_service.py`, `src/ticket_classifier/api/routes_predict.py`)
- **Perfil**: backend
- **Depende de**: TASK-005, TASK-023
- **Arquivos de produção**:
  - `src/ticket_classifier/services/prediction_service.py`
  - `src/ticket_classifier/api/routes_predict.py`
- **Arquivos de teste**:
  - `tests/unit/test_prediction_service.py`
  - `tests/integration/test_api_predict.py`
- **Wiring permitido**:
  - `src/ticket_classifier/services/__init__.py` (criar vazio)
  - `src/ticket_classifier/api/app.py` (apenas `include_router` do router de predict)
- **Reusa**:
  - `src/ticket_classifier/preprocessing.py` → `preprocess_text`
  - `src/ticket_classifier/models/base.py` → `top_label`
  - `src/ticket_classifier/storage/database.py` → `transaction`, `utc_now`
  - `src/ticket_classifier/storage/predictions.py` → `insert_prediction`, `PredictionRecord`
  - `src/ticket_classifier/api/app.py` → `get_app_state`, `get_connection`
  - `src/ticket_classifier/api/errors.py` → `error_response`
  - `src/ticket_classifier/api/schemas.py` → `PredictRequest`, `PredictResponse`, `TopCategory`
  - `tests/support/api_fixtures.py` → `build_promoted_baseline`
- **Contrato**:
  - CT-24 — `ModelUnavailableError`, `PredictionStorageError`, `predict_ticket(state, conn, title, description)`, `POST /predict` (produz)
  - CT-1, CT-3, CT-5, CT-19, CT-22, CT-23 (consome)
- **Testes**: integration
- **Descrição**: Fluxo da seção 5.2 do plan. `needs_review = cat_conf < limiar or pri_conf < limiar` (LAC-22). `top_categories` = 3 maiores, desempate pela ordem de CT-1, primeiro item com o mesmo float de `category_confidence`. `prediction_id = str(uuid4())`. Grava só texto mascarado (resultado de `preprocess_text`). `ModelUnavailableError` → 503 `MODEL_UNAVAILABLE`; `sqlite3.Error` → `PredictionStorageError` → 503 `STORAGE_UNAVAILABLE`, sem corpo de previsão. Rota síncrona, sem autenticação. Log sem texto do ticket (seção 14).
- **Done when**:
  - [ ] Ticket válido → 200 com exatamente as chaves de API-01; `0 <= confidences <= 1`; `top_categories` com 3 categorias distintas em ordem decrescente e o primeiro item igual a (`category`, `category_confidence`)
  - [ ] Com `review_threshold=0.0` → `needs_review == false`; com `1.0` → `needs_review == true` (classificador fake com confianças < 1)
  - [ ] Ticket com `john@acme.com` e `+1 555 123 4567`: a linha gravada tem `[EMAIL]` e `[PHONE]` e nenhuma coluna da tabela contém `john@acme.com`
  - [ ] Chamada sem nenhum cabeçalho de chave → 200
  - [ ] Os casos 422 de API-90..API-93 via `POST /predict` devolvem as mensagens de 8.3 e a tabela `predictions` continua vazia
  - [ ] Sem versão promovida → 503 `Model not available. Try again later.` e nenhuma Previsão gravada
  - [ ] `insert_prediction` forçado a levantar `sqlite3.OperationalError` → 503 `Service temporarily unavailable. Please try again later.` sem `prediction_id` na resposta
  - [ ] 20 requisições concorrentes (`ThreadPoolExecutor` + `TestClient`) com textos distintos → 20 `prediction_id` distintos e cada linha gravada com o texto da sua requisição
  - [ ] Description de 5000 caracteres → 200
  - [ ] `uv run pytest tests/unit/test_prediction_service.py tests/integration/test_api_predict.py` passa
- **Não fazer**:
  - Não adicionar autenticação ao `/predict` (LAC-08)
  - Não devolver distribuição completa de probabilidades (LAC-13 = top-3)
  - Não logar `title`/`description`

---

### TASK-025 — Implement POST /feedback with API key
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `FDBK-01`, `FDBK-02`, `FDBK-04`, `FDBK-90`, `FDBK-91`, `FDBK-92`, `FDBK-93`, `FDBK-95`
- **Tipo**: crud-padrão
- **Risco**: crítico
- **Âncora de risco**: AS-1, AS-7 (autorização por chave e comparação do segredo — `src/ticket_classifier/api/security.py`). Acima do teto de `crud-padrão` porque decide autorização.
- **Perfil**: backend
- **Depende de**: TASK-020, TASK-023
- **Arquivos de produção**:
  - `src/ticket_classifier/api/security.py`
  - `src/ticket_classifier/api/routes_feedback.py`
- **Arquivos de teste**:
  - `tests/unit/test_security.py`
  - `tests/integration/test_api_feedback.py`
- **Wiring permitido**:
  - `src/ticket_classifier/api/app.py` (apenas `include_router` do router de feedback)
- **Reusa**:
  - `hmac.compare_digest` (stdlib)
  - `src/ticket_classifier/storage/feedback.py` → `insert_feedback`, `PredictionNotFoundError`
  - `src/ticket_classifier/api/app.py` → `get_app_state`, `get_connection`
  - `src/ticket_classifier/api/errors.py` → `error_response`, `validation_message`
  - `src/ticket_classifier/api/schemas.py` → `FeedbackRequest`, `FeedbackResponse`
  - `tests/support/api_fixtures.py` → `build_promoted_baseline`
- **Contrato**:
  - CT-25 — `API_KEY_HEADER`, `is_authorized(settings, provided) -> bool`, `POST /feedback` (produz)
  - CT-4, CT-20, CT-22, CT-23 (consome)
- **Testes**: integration
- **Descrição**: DA-13: a rota recebe `Request`, verifica `is_authorized` antes de ler o JSON (401 `UNAUTHORIZED`), depois valida com `FeedbackRequest.model_validate_json` (erro → 422 via `validation_message`), grava com `insert_feedback` (`PredictionNotFoundError` → 404; `sqlite3.Error` → 503) e responde 201 com `FeedbackResponse`. Schema publicado via `openapi_extra`. `is_authorized` devolve `False` se a chave não está configurada (LAC-20). Nunca logar a chave.
- **Done when**:
  - [ ] Chave válida + Previsão existente → 201 com `feedback_id`, `prediction_id`, `category`, `priority`, `received_at` e 1 linha em `feedback`
  - [ ] Segundo Feedback para a mesma Previsão → 201 e 2 linhas na tabela
  - [ ] Feedback com rótulos iguais aos previstos → 201
  - [ ] Sem cabeçalho, com chave errada e com corpo `not json` sem chave → 401 `Invalid or missing API key.` e 0 linhas gravadas
  - [ ] App com `api_key=None` → 401 para chave qualquer; `/predict` e `/health` continuam 200
  - [ ] `prediction_id` inexistente → 404 `Prediction '<id>' not found.` e 0 linhas
  - [ ] `category="hardware"` com chave válida → 422 com a lista de valores aceitos e 0 linhas
  - [ ] `insert_feedback` forçado a levantar `sqlite3.OperationalError` → 503 `Service temporarily unavailable. Please try again later.` e 0 linhas
  - [ ] `caplog` de todas as chamadas não contém o valor da chave
  - [ ] `/openapi.json` descreve o corpo de `POST /feedback` com `prediction_id`, `category` e `priority`
  - [ ] `uv run pytest tests/unit/test_security.py tests/integration/test_api_feedback.py` passa
- **Não fazer**:
  - Não exigir chave em `/predict` ou `/health`
  - Não suportar múltiplas chaves ou usuários

---

### TASK-026 — Schedule automatic purge in the API lifespan
- **Status**: ✅ APROVADA em 2026-09-30

- **Requisito**: `OPS-08`
- **Tipo**: infra
- **Risco**: crítico
- **Âncora de risco**: AS-5 (automação da retenção LGPD — `src/ticket_classifier/api/purge_scheduler.py`)
- **Perfil**: backend
- **Depende de**: TASK-021, TASK-023
- **Arquivos de produção**:
  - `src/ticket_classifier/api/purge_scheduler.py`
- **Arquivos de teste**:
  - `tests/unit/test_purge_scheduler.py`
- **Wiring permitido**:
  - `src/ticket_classifier/api/app.py` (apenas iniciar `start_purge_loop` no lifespan e cancelar a task no encerramento)
- **Reusa**:
  - `src/ticket_classifier/storage/purge.py` → `purge_expired_predictions`, `PURGE_MESSAGE`
  - `src/ticket_classifier/storage/database.py` → `connect`, `utc_now`
  - `asyncio.to_thread` (stdlib)
- **Contrato**:
  - CT-26 — `run_purge_once(settings) -> int`, `start_purge_loop(settings) -> asyncio.Task[None]` (produz)
  - CT-4, CT-19, CT-21, CT-23 (consome)
- **Testes**: unit
- **Descrição**: DA-14: `run_purge_once` abre conexão própria, executa o expurgo em `asyncio.to_thread`, loga `PURGE_MESSAGE` em INFO e devolve a contagem; exceção é logada em ERROR e devolve -1 sem derrubar o loop. `start_purge_loop` roda uma vez imediatamente e depois a cada `purge_interval_seconds`.
- **Done when**:
  - [ ] Banco com uma Previsão de 91 dias sem Feedback: `await run_purge_once(settings)` devolve 1 e o log contém `Purge finished: 1 predictions without feedback older than 90 days deleted.`
  - [ ] `db_path` inválido: `run_purge_once` devolve -1 e não levanta exceção
  - [ ] Com `purge_interval_seconds=1`, o loop executa ao menos 2 vezes em 1.5 s (contagem via `monkeypatch`) e a task termina sem erro ao ser cancelada
  - [ ] Subir e encerrar a app com `TestClient` (context manager) com a Previsão antiga no banco deixa 0 Previsões expiradas e não deixa task pendente
  - [ ] `uv run pytest tests/unit/test_purge_scheduler.py` passa
- **Não fazer**:
  - Não adicionar APScheduler nem cron
  - Não alterar a regra de expurgo (TASK-021)

---

### TASK-027 — Retrain with feedback and gated promotion

- **Requisito**: `RETR-01`, `RETR-02`, `RETR-03`, `RETR-04`, `RETR-90`, `RETR-91`, `RETR-92`
- **Tipo**: lógica-negócio
- **Risco**: médio
- **Perfil**: backend
- **Depende de**: TASK-016, TASK-020
- **Arquivos de produção**:
  - `src/ticket_classifier/retraining.py`
- **Arquivos de teste**:
  - `tests/integration/test_retraining.py`
- **Wiring permitido**:
  - `src/ticket_classifier/cli.py` (apenas registrar `retrain` com `--config`)
  - `README.md` (seção "Retraining": pré-requisitos, comando, regra de promoção)
- **Reusa**:
  - `src/ticket_classifier/training.py` → `train_and_register`
  - `src/ticket_classifier/registry.py` → `ModelRegistry.get_promoted`, `ModelRegistry.promote`
  - `src/ticket_classifier/storage/database.py` → `connect`
  - `src/ticket_classifier/storage/feedback.py` → `list_current_feedback`
  - `src/ticket_classifier/data/splits.py` → `splits_version`
  - `fcntl.flock` (stdlib)
  - `tests/support/sample_data.py` → `prepare_sample`
  - `tests/support/tiny_model.py` → `build_tiny_model`
- **Contrato**:
  - CT-27 — `RetrainInProgressError`, `NoFeedbackError`, `RetrainResult`, `run_retraining(config, settings)` (produz)
  - CT-4, CT-6, CT-8, CT-11, CT-16, CT-20, CT-28 (consome)
- **Testes**: integration
- **Descrição**: DA-15. Ordem: lock não bloqueante em `artifacts/retrain.lock` (falha → `RetrainInProgressError`) → promovida obrigatória → `splits_version` atual igual ao da promovida (senão `PipelineError` de 8.3) → `list_current_feedback` (vazio → `NoFeedbackError`, nada registrado) → `train_and_register("transformer", ..., extra_train=<feedback>)` → gate LAC-23 com métricas `test` → `promote` ou não → `RetrainResult`. CLI imprime a mensagem RETR-04 e as métricas de test (Macro F1 por alvo) da nova e da promovida. Qualquer exceção antes do `promote` mantém a promovida.
- **Done when**:
  - [ ] Com 3 Previsões e 4 Feedbacks (2 para a mesma Previsão), o DataFrame passado como `extra_train` (espião) tem 3 linhas com os rótulos dos Feedbacks vigentes e o texto mascarado
  - [ ] O `test.jsonl` usado na avaliação é o mesmo arquivo da promovida (mesmo `splits_version`) e não contém nenhum texto de Feedback
  - [ ] Cenário melhor (métricas de test da nova forçadas >= promovida via `monkeypatch`) → `decision == "promoted"` e a antiga fica `retired`; cenário pior → `decision == "not promoted"`, nova `registered`, promovida inalterada
  - [ ] Sem Feedback → `NoFeedbackError` `No feedback records available. Retraining aborted.`, número de versões inalterado
  - [ ] `train_and_register` forçado a levantar exceção → promovida inalterada e nenhuma versão nova `promoted`
  - [ ] Com o lock segurado por outro processo (`multiprocessing`), `run_retraining` levanta `RetrainInProgressError` `A retraining run is already in progress.` e o outro processo segue com o lock
  - [ ] `uv run ticket-classifier retrain --config <tmp>/pipeline.toml` imprime `Retraining finished: <n> feedback records used. Decision: ` seguido de `promoted` ou `not promoted`
  - [ ] `uv run pytest tests/integration/test_retraining.py` passa em CPU sem rede
- **Não fazer**:
  - Não criar endpoint HTTP de retreino
  - Não disparar retreino automaticamente
  - Não incluir Feedback em `validation` ou `test`

---

### TASK-028 — Package the API as a CPU-only Docker image

- **Requisito**: `OPS-02`
- **Tipo**: infra
- **Risco**: médio
- **Perfil**: infra
- **Depende de**: TASK-024, TASK-025, TASK-026, TASK-027
- **Arquivos de produção**:
  - `docker/Dockerfile`
  - `docker/Dockerfile.dockerignore`
- **Arquivos de teste**:
  - `tests/container/test_container_smoke.py`
- **Wiring permitido**:
  - `README.md` (seções "Run with Docker" com o comando da seção 3 do plan e "Architecture" com a visão de componentes, o fluxo do `/predict` e as decisões DA-1..DA-17 resumidas)
- **Reusa**:
  - `tests/support/api_fixtures.py` → `build_promoted_baseline`
- **Contrato**:
  - CT-23, CT-24, CT-25 (consome)
- **Testes**: e2e
- **Gate**: `uv run pytest --junitxml=reports/junit.xml -m container tests/container/test_container_smoke.py` (executar em `.`)
- **Descrição**: Imagem `python:3.12-slim`, `uv` copiado de `ghcr.io/astral-sh/uv:0.12.21`, `uv sync --frozen --no-dev` sem o extra `train` (camada de dependências antes do código), `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TICKET_ARTIFACTS_DIR=/app/artifacts TICKET_DB_PATH=/app/var/tickets.db`, usuário não-root, `EXPOSE 8000`, `CMD` uvicorn `--factory ticket_classifier.api.app:create_app --host 0.0.0.0 --port 8000`. O ignore exclui `.venv`, `data`, `artifacts`, `var`, `reports`, `mlflow.db`, `.git`, `.specs`. O smoke constrói a imagem, gera baseline promovido em `tmp_path`, sobe o container na porta 18080 com o volume de artefatos e `TICKET_API_KEY`, e remove o container no fim.
- **Done when**:
  - [ ] `docker build -f docker/Dockerfile -t support-ticket-ai .` termina com exit 0
  - [ ] `docker run --rm support-ticket-ai python -c "import torch, importlib.util; assert torch.version.cuda is None; assert importlib.util.find_spec('mlflow') is None"` termina com exit 0
  - [ ] No smoke, `GET http://localhost:18080/health` → 200 com `model_version` e `POST /predict` com ticket válido → 200 com `prediction_id`
  - [ ] `README.md` contém as seções `Run with Docker` e `Architecture`
  - [ ] `uv run pytest --junitxml=reports/junit.xml -m container tests/container/test_container_smoke.py` passa
  - [ ] `uv run pytest` (suíte padrão) não executa o smoke do container
- **Não fazer**:
  - Não copiar `artifacts/` nem modelos para dentro da imagem
  - Não criar `docker-compose.yml` nem pipeline de CI
