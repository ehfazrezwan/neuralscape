# ICEBench Report

**Harness:** ICEBench-v1
**Date:** 2026-07-08
**Machine:** ns-bench 8vCPU/31GB

This report presents performance and accuracy measurements for coding-agentic memory layers.
No established benchmark exists for this domain (design stance: coding-agentic memory is a
novel capability requiring novel evaluation).



## Methodology

**Harness:** ICEBench-v1
**Machine:** ns-bench 8vCPU/31GB
**Repo SHA:** 8a4ce842564ae94ab050062db8525196ad476c19
**Seed:** 42
**Quiescence:** Measured on ns-bench (8 vCPU / 31 GB) with ZERO competing benchmark stacks running (the nsbench factory stacks were stopped for the run).

Each measurement is the median of 3 repetitions with min/max reported.
Percentiles (p50/p95/p99) use the nearest-rank definition.


## Capability Matrix

System capabilities by operation class:

| Operation | cbm | graphify | ns-graphify | ns-ice | ns-ice-det |
| --- | --- | --- | --- | --- | --- |
| blast_radius | N/A (no impact-analysis op) | N/A (no impact-analysis op) | N/A (no impact-analysis op) | supported | supported |
| neighbors_1hop | supported | supported | supported | supported | supported |
| nl_locate | N/A (no NL→symbol retrieval) | N/A (no NL→symbol retrieval) | N/A (no NL→symbol retrieval) | supported | supported |
| path_le4 | supported | supported | supported | supported | supported |
| symbol_lookup | supported | supported | supported | supported | supported |

## Track P: Performance

### cbm

#### Corpus: small-py

| Operation | Wall (s) | Peak RSS (MB) | CPU (s) | Latency p50/p95/p99 (ms) | Bytes | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| export_snapshot | 0.02 (0.02-0.02) | N/A | N/A | N/A | 17235968 | - |
| index_cold | 0.13 (0.12-1.34) | 8.37 (8.36-310.70) | 0.03 (0.03-5.50) | N/A | N/A | - |
| index_incremental_1 | DNF | DNF | DNF | DNF | DNF | incremental_na |
| index_incremental_5 | DNF | DNF | DNF | DNF | DNF | incremental_na |
| index_second | 1.22 (1.22-1.23) | 72.22 (72.20-72.49) | 2.12 (2.09-2.14) | N/A | N/A | - |
| neighbors_1hop | N/A | N/A | N/A | 24.95/32.78/36.31 | N/A | - |
| path_le4 | N/A | N/A | N/A | 51.05/56.49/56.49 | N/A | - |
| store_size | N/A | N/A | N/A | N/A | 17235968 | - |
| symbol_lookup | N/A | N/A | N/A | 26.53/33.42/35.38 | N/A | - |

### graphify

#### Corpus: small-py

| Operation | Wall (s) | Peak RSS (MB) | CPU (s) | Latency p50/p95/p99 (ms) | Bytes | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| export_snapshot | 0.00 (0.00-0.00) | N/A | N/A | N/A | 1744143 | - |
| index_cold | 0.41 (0.39-2.65) | 36.48 (36.36-89.71) | 0.39 (0.38-2.62) | N/A | N/A | - |
| index_incremental_1 | DNF | DNF | DNF | DNF | DNF | incremental_na |
| index_incremental_5 | DNF | DNF | DNF | DNF | DNF | incremental_na |
| index_second | 0.41 (0.40-0.42) | 36.33 (36.05-36.46) | 0.39 (0.39-0.40) | N/A | N/A | - |
| neighbors_1hop | N/A | N/A | N/A | 314.58/329.82/343.03 | N/A | - |
| path_le4 | N/A | N/A | N/A | 326.26/339.81/339.81 | N/A | - |
| store_size | N/A | N/A | N/A | N/A | 5744079 | - |
| symbol_lookup | N/A | N/A | N/A | 312.26/322.51/338.29 | N/A | - |

### ns-graphify

#### Corpus: small-py

| Operation | Wall (s) | Peak RSS (MB) | CPU (s) | Latency p50/p95/p99 (ms) | Bytes | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| export_snapshot | 0.00 (0.00-0.00) | N/A | N/A | N/A | 252420 | - |
| index_cold | 0.42 (0.40-2.86) | 36.55 (36.27-87.35) | 0.39 (0.37-4.17) | N/A | N/A | - |
| index_incremental_1 | 0.45 (0.45-0.51) | 38.93 (38.68-39.12) | 0.43 (0.43-0.49) | N/A | N/A | - |
| index_incremental_5 | 0.66 (0.64-0.86) | 41.05 (40.84-44.07) | 0.63 (0.62-0.83) | N/A | N/A | - |
| index_second | 0.67 (0.67-0.67) | 44.37 (44.30-44.41) | 0.65 (0.65-0.65) | N/A | N/A | - |
| neighbors_1hop | N/A | N/A | N/A | N/A | N/A | No valid runs |
| path_le4 | N/A | N/A | N/A | N/A | N/A | No valid runs |
| store_size | N/A | N/A | N/A | N/A | 252420 | - |
| symbol_lookup | N/A | N/A | N/A | N/A | N/A | No valid runs |

### ns-ice

#### Corpus: small-py

| Operation | Wall (s) | Peak RSS (MB) | CPU (s) | Latency p50/p95/p99 (ms) | Bytes | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| blast_radius | N/A | N/A | N/A | 10446.55/14348.37/14742.90 | N/A | - |
| export_snapshot | N/A | N/A | N/A | N/A | N/A | No valid runs |
| index_cold | 123.57 (122.94-124.12) | 402.73 (401.61-404.95) | 16.56 (16.29-16.59) | N/A | N/A | - |
| index_incremental_1 | 49.45 (49.03-96.99) | 403.42 (402.21-403.54) | 8.19 (8.18-8.34) | N/A | N/A | - |
| index_incremental_5 | 58.08 (58.04-58.15) | 401.36 (401.18-402.02) | 9.30 (9.20-9.32) | N/A | N/A | - |
| index_second | 123.46 (122.87-124.63) | 403.19 (403.07-403.25) | 16.44 (16.39-17.08) | N/A | N/A | - |
| neighbors_1hop | N/A | N/A | N/A | 420.40/468.11/514.67 | N/A | - |
| nl_locate | N/A | N/A | N/A | 8249.97/8936.94/22662.53 | N/A | - |
| path_le4 | N/A | N/A | N/A | 21.80/25.23/25.23 | N/A | - |
| store_size | N/A | N/A | N/A | N/A | 2204035599 | - |
| symbol_lookup | N/A | N/A | N/A | 14.10/854.39/8156.28 | N/A | - |

### ns-ice-det

#### Corpus: small-py

| Operation | Wall (s) | Peak RSS (MB) | CPU (s) | Latency p50/p95/p99 (ms) | Bytes | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| blast_radius | N/A | N/A | N/A | 10380.49/14390.29/14555.00 | N/A | - |
| export_snapshot | N/A | N/A | N/A | N/A | N/A | No valid runs |
| index_cold | 100.10 (99.81-100.37) | 293.54 (293.34-293.78) | 15.05 (14.83-15.07) | N/A | N/A | - |
| index_incremental_1 | 24.93 (24.93-25.08) | 288.77 (288.62-288.86) | 6.14 (6.14-6.33) | N/A | N/A | - |
| index_incremental_5 | 34.37 (34.30-34.40) | 291.12 (290.96-291.19) | 7.55 (7.45-7.72) | N/A | N/A | - |
| index_second | 99.85 (99.79-99.89) | 293.32 (293.31-294.58) | 14.79 (14.67-14.88) | N/A | N/A | - |
| neighbors_1hop | N/A | N/A | N/A | 407.12/451.09/484.96 | N/A | - |
| nl_locate | N/A | N/A | N/A | 4071.48/4313.29/8197.75 | N/A | - |
| path_le4 | N/A | N/A | N/A | 20.74/27.20/27.20 | N/A | - |
| store_size | N/A | N/A | N/A | N/A | 2204066959 | - |
| symbol_lookup | N/A | N/A | N/A | 13.92/3516.53/8057.70 | N/A | - |


## Track Q: Accuracy

### cbm

#### Corpus: small-py

| Operation | Hits@1 | Hits@5 | Hits@10 | MRR | Hit Rate | Precision | Recall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neighbors_1hop | N/A | N/A | N/A | N/A | N/A | 0.552 | 0.503 |  |
| path_le4 | N/A | N/A | N/A | N/A | 0.400 | N/A | N/A |  |
| symbol_lookup | N/A | N/A | N/A | N/A | 0.510 | N/A | N/A |  |

### graphify

#### Corpus: small-py

| Operation | Hits@1 | Hits@5 | Hits@10 | MRR | Hit Rate | Precision | Recall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neighbors_1hop | N/A | N/A | N/A | N/A | N/A | 0.262 | 0.676 |  |
| path_le4 | N/A | N/A | N/A | N/A | 1.000 | N/A | N/A |  |
| symbol_lookup | N/A | N/A | N/A | N/A | 0.540 | N/A | N/A |  |

### ns-graphify

#### Corpus: small-py

| Operation | Hits@1 | Hits@5 | Hits@10 | MRR | Hit Rate | Precision | Recall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neighbors_1hop | N/A | N/A | N/A | N/A | N/A | N/A | N/A |  |
| path_le4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |  |
| symbol_lookup | N/A | N/A | N/A | N/A | N/A | N/A | N/A |  |

### ns-ice

#### Corpus: small-py

| Operation | Hits@1 | Hits@5 | Hits@10 | MRR | Hit Rate | Precision | Recall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neighbors_1hop | N/A | N/A | N/A | N/A | N/A | 0.002 | 0.013 |  |
| nl_locate | 0.813 | 0.813 | 0.820 | 0.814 | N/A | N/A | N/A |  |
| path_le4 | N/A | N/A | N/A | N/A | 0.100 | N/A | N/A |  |
| symbol_lookup | N/A | N/A | N/A | N/A | 0.000 | N/A | N/A |  |

### ns-ice-det

#### Corpus: small-py

| Operation | Hits@1 | Hits@5 | Hits@10 | MRR | Hit Rate | Precision | Recall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neighbors_1hop | N/A | N/A | N/A | N/A | N/A | 0.001 | 0.005 |  |
| nl_locate | 0.090 | 0.215 | 0.325 | 0.147 | N/A | N/A | N/A |  |
| path_le4 | N/A | N/A | N/A | N/A | 0.100 | N/A | N/A |  |
| symbol_lookup | N/A | N/A | N/A | N/A | 0.000 | N/A | N/A |  |


## DNF Log

DNF (Did Not Finish) events are first-class results indicating stability issues. Both Track-P (performance) and Track-Q (accuracy) DNFs are included:

| Track | System | Corpus | Operation | Rep | Reason |
| --- | --- | --- | --- | --- | --- |
| P | graphify | small-py | index_incremental_1 | 0 | incremental_na |
| P | graphify | small-py | index_incremental_5 | 0 | incremental_na |
| P | graphify | small-py | index_incremental_1 | 1 | incremental_na |
| P | graphify | small-py | index_incremental_5 | 1 | incremental_na |
| P | graphify | small-py | index_incremental_1 | 2 | incremental_na |
| P | graphify | small-py | index_incremental_5 | 2 | incremental_na |
| P | cbm | small-py | index_incremental_1 | 0 | incremental_na |
| P | cbm | small-py | index_incremental_5 | 0 | incremental_na |
| P | cbm | small-py | index_incremental_1 | 1 | incremental_na |
| P | cbm | small-py | index_incremental_5 | 1 | incremental_na |
| P | cbm | small-py | index_incremental_1 | 2 | incremental_na |
| P | cbm | small-py | index_incremental_5 | 2 | incremental_na |

## Caveats & Methodology Notes

**Sample sizes:** Each measurement is the median of 3 repetitions.

**Shared oracle bias:** All systems share the same ground-truth oracle (tree-sitter structural QA).

**NS-authored harness bias:** This harness is authored by the Neuralscape team. Mitigations: adapters are forbidden from encoding system-specific intelligence; per-operation N/A honesty (no fabricated numbers for unsupported operations).

**Novel domain:** No established benchmark exists for coding-agentic memory layers. This is an initial design stance for evaluating a novel capability.
