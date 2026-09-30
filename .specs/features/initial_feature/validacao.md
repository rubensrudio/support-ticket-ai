# Validações pré-entrega (geradas por check_plan.py — não editar à mão)

## Mapa de ondas (calculado)

| Onda | Tasks | Risco máx | Gatilhos previstos | QA previsto |
|---|---|---|---|---|
| 1 | TASK-001 | médio | — | — |
| 2 | TASK-004 | crítico | G1 TASK-004 | EXAUSTIVO |
| 3 | TASK-002, TASK-005 | médio | — | — |
| 4 | TASK-003, TASK-006, TASK-007, TASK-011 | médio | G2 CT-1,CT-2 | PADRAO |
| 5 | TASK-013 | alto | G1 TASK-013 | RIGOROSO |
| 6 | TASK-008, TASK-010, TASK-012, TASK-015 | médio | G2 CT-3,CT-1 | PADRAO |
| 7 | TASK-009, TASK-014, TASK-019, TASK-022 | médio | G2 CT-6,CT-1; G3 (1 história(s) P1) | PADRAO |
| 8 | TASK-016, TASK-020 | médio | — | — |
| 9 | TASK-021 | crítico | G1 TASK-021 | EXAUSTIVO |
| 10 | TASK-023 | alto | G1 TASK-023 | RIGOROSO |
| 11 | TASK-024 | crítico | G1 TASK-024; G3 (1 história(s) P1) | EXAUSTIVO |
| 12 | TASK-025 | crítico | G1 TASK-025; G3 (1 história(s) P1) | EXAUSTIVO |
| 13 | TASK-026 | crítico | G1 TASK-026 | EXAUSTIVO |
| 14 | TASK-017 | médio | G3 (1 história(s) P1) | PADRAO |
| 15 | TASK-018 | médio | — | — |
| 16 | TASK-027 | médio | — | — |
| 17 | TASK-028 | médio | — | — |

PRs previstos (uma fase termina numa onda com QA semântico; a última, no QA FEATURE): fase 1: ondas 1, 2 · fase 2: ondas 3, 4 · fase 3: ondas 5 · fase 4: ondas 6 · fase 5: ondas 7 · fase 6: ondas 8, 9 · fase 7: ondas 10 · fase 8: ondas 11 · fase 9: ondas 12 · fase 10: ondas 13 · fase 11: ondas 14 · fase 12: ondas 15, 16, 17

Caminho crítico: TASK-001 → TASK-002 → TASK-003 → TASK-013 → TASK-014 → TASK-016 → TASK-023 → TASK-026 → TASK-028 (9 tasks)
G4 (gate cego) e G5 (retry) só são conhecidos na execução.

## Cobertura de requisitos

| ID do spec | Task(s) | Status |
|---|---|---|
| `DATA-01` | TASK-009 | ✅ |
| `DATA-02` | TASK-002, TASK-007 | ✅ |
| `DATA-03` | TASK-009 | ✅ |
| `DATA-04` | TASK-006, TASK-009 | ✅ |
| `DATA-05` | TASK-009 | ✅ |
| `DATA-06` | TASK-009 | ✅ |
| `DATA-07` | TASK-005 | ✅ |
| `DATA-08` | TASK-005 | ✅ |
| `DATA-09` | TASK-009 | ✅ |
| `DATA-10` | TASK-009 | ✅ |
| `DATA-11` | TASK-009 | ✅ |
| `DATA-12` | TASK-009 | ✅ |
| `MODEL-01` | TASK-012 | ✅ |
| `MODEL-02` | TASK-013, TASK-014 | ✅ |
| `MODEL-03` | TASK-010 | ✅ |
| `MODEL-04` | TASK-017 | ✅ |
| `MODEL-05` | TASK-017 | ✅ |
| `MODEL-06` | TASK-014, TASK-016 | ✅ |
| `MODEL-07` | TASK-012, TASK-014, TASK-016 | ✅ |
| `MODEL-08` | TASK-011, TASK-016 | ✅ |
| `MODEL-09` | TASK-016 | ✅ |
| `API-01` | TASK-024 | ✅ |
| `API-02` | TASK-003, TASK-024 | ✅ |
| `API-03` | TASK-024 | ✅ |
| `API-04` | TASK-004, TASK-024 | ✅ |
| `API-05` | TASK-024 | ✅ |
| `API-06` | TASK-013, TASK-024 | ✅ |
| `API-07` | TASK-019, TASK-024 | ✅ |
| `API-08` | TASK-024 | ✅ |
| `API-09` | TASK-023 | ✅ |
| `FDBK-01` | TASK-020, TASK-025 | ✅ |
| `FDBK-02` | TASK-020, TASK-025 | ✅ |
| `FDBK-03` | TASK-020 | ✅ |
| `FDBK-04` | TASK-020, TASK-025 | ✅ |
| `OPS-01` | TASK-023 | ✅ |
| `OPS-02` | TASK-028 | ✅ |
| `OPS-03` | TASK-001 | ✅ |
| `OPS-04` | TASK-015, TASK-016 | ✅ |
| `OPS-05` | TASK-011 | ✅ |
| `OPS-06` | TASK-011 | ✅ |
| `MODEL-10` | TASK-018 | ✅ |
| `MODEL-11` | TASK-018 | ✅ |
| `OPS-07` | TASK-021 | ✅ |
| `OPS-08` | TASK-026 | ✅ |
| `RETR-01` | TASK-027 | ✅ |
| `RETR-02` | TASK-027 | ✅ |
| `RETR-03` | TASK-027 | ✅ |
| `RETR-04` | TASK-027 | ✅ |
| `DATA-90` | TASK-009 | ✅ |
| `DATA-91` | TASK-007 | ✅ |
| `DATA-92` | TASK-009 | ✅ |
| `MODEL-90` | TASK-008, TASK-016 | ✅ |
| `API-90` | TASK-022, TASK-024 | ✅ |
| `API-91` | TASK-022, TASK-024 | ✅ |
| `API-92` | TASK-022, TASK-024 | ✅ |
| `API-93` | TASK-022, TASK-024 | ✅ |
| `API-94` | TASK-023, TASK-024 | ✅ |
| `API-95` | TASK-019, TASK-024 | ✅ |
| `API-96` | TASK-024 | ✅ |
| `FDBK-90` | TASK-025 | ✅ |
| `FDBK-91` | TASK-025 | ✅ |
| `FDBK-92` | TASK-022, TASK-025 | ✅ |
| `FDBK-93` | TASK-004, TASK-023, TASK-025 | ✅ |
| `FDBK-94` | TASK-020 | ✅ |
| `FDBK-95` | TASK-025 | ✅ |
| `OPS-90` | TASK-021 | ✅ |
| `RETR-90` | TASK-027 | ✅ |
| `RETR-91` | TASK-027 | ✅ |
| `RETR-92` | TASK-027 | ✅ |

Cobertura: 69/69

## Dependências

| Task | Depende de | Onda | Ondas das dependências |
|---|---|---|---|
| TASK-001 | — | 1 | — |
| TASK-002 | TASK-001 | 3 | 1 |
| TASK-003 | TASK-002 | 4 | 3 |
| TASK-004 | TASK-001 | 2 | 1 |
| TASK-005 | TASK-001 | 3 | 1 |
| TASK-006 | TASK-002 | 4 | 3 |
| TASK-007 | TASK-002 | 4 | 3 |
| TASK-008 | TASK-002 | 6 | 3 |
| TASK-009 | TASK-005, TASK-006, TASK-007, TASK-008 | 7 | 3, 4, 4, 6 |
| TASK-010 | TASK-003 | 6 | 4 |
| TASK-011 | TASK-002 | 4 | 3 |
| TASK-012 | TASK-003, TASK-006 | 6 | 4, 4 |
| TASK-013 | TASK-003 | 5 | 4 |
| TASK-014 | TASK-006, TASK-010, TASK-013 | 7 | 4, 6, 5 |
| TASK-015 | TASK-011 | 6 | 4 |
| TASK-016 | TASK-004, TASK-008, TASK-009, TASK-010, TASK-011, TASK-012, TASK-013, TASK-014, TASK-015 | 8 | 2, 6, 7, 6, 4, 6, 5, 7, 6 |
| TASK-017 | TASK-010, TASK-011 | 14 | 6, 4 |
| TASK-018 | TASK-004, TASK-016 | 15 | 2, 8 |
| TASK-019 | TASK-002 | 7 | 3 |
| TASK-020 | TASK-019 | 8 | 7 |
| TASK-021 | TASK-004, TASK-020 | 9 | 2, 8 |
| TASK-022 | TASK-002 | 7 | 3 |
| TASK-023 | TASK-004, TASK-009, TASK-011, TASK-016, TASK-019, TASK-022 | 10 | 2, 7, 4, 8, 7, 7 |
| TASK-024 | TASK-005, TASK-023 | 11 | 3, 10 |
| TASK-025 | TASK-020, TASK-023 | 12 | 8, 10 |
| TASK-026 | TASK-021, TASK-023 | 13 | 9, 10 |
| TASK-027 | TASK-016, TASK-020 | 16 | 8, 8 |
| TASK-028 | TASK-024, TASK-025, TASK-026, TASK-027 | 17 | 11, 12, 13, 16 |

## Contratos (produtor × consumidores)

| Contrato | Produtor | Consumidores |
|---|---|---|
| CT-1 | TASK-002 | TASK-003, TASK-007, TASK-010, TASK-012, TASK-013, TASK-019, TASK-022, TASK-024 |
| CT-2 | TASK-002 | TASK-006, TASK-007, TASK-008, TASK-011 |
| CT-3 | TASK-003 | TASK-010, TASK-012, TASK-013, TASK-018, TASK-023, TASK-024 |
| CT-4 | TASK-004 | TASK-016, TASK-018, TASK-021, TASK-023, TASK-025, TASK-026, TASK-027 |
| CT-5 | TASK-005 | TASK-009, TASK-024 |
| CT-6 | TASK-006 | TASK-009, TASK-012, TASK-014, TASK-016, TASK-027 |
| CT-7 | TASK-007 | TASK-009 |
| CT-8 | TASK-008 | TASK-009, TASK-016, TASK-018, TASK-027 |
| CT-9 | TASK-009 | TASK-016, TASK-023 |
| CT-10 | TASK-010 | TASK-014, TASK-016, TASK-017, TASK-018 |
| CT-11 | TASK-011 | TASK-015, TASK-016, TASK-017, TASK-018, TASK-023, TASK-027 |
| CT-12 | TASK-012 | TASK-016 |
| CT-13 | TASK-013 | TASK-014, TASK-016 |
| CT-14 | TASK-014 | TASK-016 |
| CT-15 | TASK-015 | TASK-016 |
| CT-16 | TASK-016 | TASK-018, TASK-023, TASK-027 |
| CT-17 | TASK-017 | — |
| CT-18 | TASK-018 | — |
| CT-19 | TASK-019 | TASK-020, TASK-021, TASK-023, TASK-024, TASK-026 |
| CT-20 | TASK-020 | TASK-021, TASK-025, TASK-027 |
| CT-21 | TASK-021 | TASK-026 |
| CT-22 | TASK-022 | TASK-023, TASK-024, TASK-025 |
| CT-23 | TASK-023 | TASK-024, TASK-025, TASK-026, TASK-028 |
| CT-24 | TASK-024 | TASK-028 |
| CT-25 | TASK-025 | TASK-028 |
| CT-26 | TASK-026 | — |
| CT-27 | TASK-027 | — |
| CT-28 | TASK-001 | TASK-002, TASK-009, TASK-011, TASK-016, TASK-017, TASK-018, TASK-021, TASK-027 |

## Granularidade e testes

| Task | Produção | Teste | Testes | Paralelo-seguro | Agente |
|---|---|---|---|---|---|
| TASK-001 | 2 | 2 | unit | sim | hm-engineer |
| TASK-002 | 2 | 2 | unit | sim | hm-engineer |
| TASK-003 | 1 | 1 | unit | sim | hm-engineer |
| TASK-004 | 1 | 1 | unit | sim | hm-engineer |
| TASK-005 | 1 | 1 | unit | sim | hm-engineer |
| TASK-006 | 2 | 1 | unit | sim | hm-engineer |
| TASK-007 | 2 | 1 | unit | sim | hm-engineer |
| TASK-008 | 1 | 1 | unit | sim | hm-engineer |
| TASK-009 | 1 | 3 | integration | sim | hm-engineer |
| TASK-010 | 1 | 1 | unit | sim | hm-engineer |
| TASK-011 | 1 | 1 | unit | sim | hm-engineer |
| TASK-012 | 1 | 1 | unit | sim | hm-engineer |
| TASK-013 | 1 | 2 | unit | sim | hm-engineer |
| TASK-014 | 1 | 1 | integration | sim | hm-engineer |
| TASK-015 | 1 | 1 | unit | sim | hm-engineer |
| TASK-016 | 2 | 1 | integration | sim | hm-engineer |
| TASK-017 | 1 | 1 | unit | sim | hm-engineer |
| TASK-018 | 1 | 2 | unit | sim | hm-engineer |
| TASK-019 | 2 | 1 | unit | sim | hm-engineer |
| TASK-020 | 1 | 1 | unit | sim | hm-engineer |
| TASK-021 | 1 | 1 | unit | sim | hm-engineer |
| TASK-022 | 2 | 1 | unit | sim | hm-engineer |
| TASK-023 | 2 | 2 | integration | sim | hm-engineer |
| TASK-024 | 2 | 2 | integration | sim | hm-engineer |
| TASK-025 | 2 | 2 | integration | sim | hm-engineer |
| TASK-026 | 1 | 1 | unit | sim | hm-engineer |
| TASK-027 | 1 | 1 | integration | sim | hm-engineer |
| TASK-028 | 2 | 1 | e2e | não | hm-engineer |
