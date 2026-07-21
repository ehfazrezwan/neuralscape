# ICE Mission — Fable Independent Review

**Reviewer:** Fable (claude-fable-5) · **Date:** 2026-07-08 · **Tree reviewed:** `ice/benchmark` @ `a9e5aa3`
**Inputs:** ICE_MASTERPLAN.md, ICE_MISSION.md, all 11 merged ice/* PR diffs (`31ba3b4..HEAD`), ice-ckpt-1/2, engine-surface-gap.md, premise-verification.md, phase-III-runbook.md, ICE_BENCH_REPORT-ice-final.md, raw results (`ice-final.jsonl`, 2,266 rows; `ice-final.trackq.json`), systems.lock.json, status.jsonl, plus live spot-checks on the VM (suite re-run, graphify CLI, adapter/scorer/engine source).

**Bottom line:** The engineering work is real, well-tested, and the worst numbers for NS are reported against self-interest — that earns trust. But the final PR should **NOT be opened as-is**. I found one materially misleading headline number (nl_locate 81% is contaminated by docstring leakage), one wrong competitor number (CBM `index_cold` is a cache-hit no-op in 2 of 3 reps), two capability-matrix claims contradicted by the installed graphify binary, and a report whose Caveats section omits nearly every run-specific limitation the team itself documented elsewhere. All are fixable in hours. Details and required fixes below.

---

## 1. Per-PR verdicts

Independently verified: fast suite on the final tree = **2092 pass / 1 skip** (I re-ran it), matching the claimed floor; secrecy grep clean; container-gate images `ice-gate`/`ice-gate-rt` built 2026-07-08T23:23–25Z (post-#187 tree), consistent with the "gate green (2092)" status claim.

| PR | Task | Verdict | Rationale |
|----|------|---------|-----------|
| #177 | I1 dev merge | **KEEP** | Single conflict file documented; suite floor 1986→2043 in-container; fixed a dev-side standards-pool NameError regression *with* a regression test; container gate run on the merged tree. |
| #179 | I2 Louvain | **KEEP** | Premise (stub) verified before build; deterministic seed, stable-id tests, >200k-edge guard; `semantic_layer()` reads persisted props. Live run showed 10 communities persisted. |
| #178 | I4 liveness consumer | **KEEP** | Closed a real producer-without-consumer gap. Credit: Copilot found the first cut non-functional (wrong staged-row keys) and it was reworked to Qdrant scroll rather than papered over. |
| #180 | I3 CBM importer | **KEEP** (known issue) | Works against its assumed schema; H2 later found the real CBM schema differs (documented as task "I3-reconcile" in phase-III-runbook.md). Not benchmark-relevant (benchmark uses CBM standalone). Carry the known-issue note into the PR body. |
| #182 | E7 engine surface | **KEEP** | Real plan-premise gap ("native index via ingest queue" was false), honestly documented in engine-surface-gap.md, closed spec-aligned (F2 §1). Without this the benchmark could not have driven ns-ice at all. |
| #181 | H1 harness core | **KEEP** | SystemAdapter protocol, results schema, rail (cap/timeout→DNF) — and the rail was actually *enforced* only after a rework, which is recorded. |
| #183 | H3 Track-Q | **REWORK** (scoped) | Oracle independence and N/A honesty are good. Two defects: (a) `generate_nl_locate` says "strip the docstring" (trackq/generate.py:268) but nothing strips docstrings from the corpus the systems index → leakage, see §3.1; (b) LSP spot-check skipped — disclosed in trackq.json metadata but never surfaced in the report. |
| #184 | H2 competitor adapters | **REWORK** (scoped) | Installs/pins are real and verified. But the CBM adapter's `index_cold` never deletes the prior project (see §3.2), and CAPABILITIES.md's graphify N/A claims are contradicted by the installed 0.9.10 CLI (see §3.3). |
| #185 | H4 report gen | **KEEP** (small rework) | N/A≠0 and DNF-first-class rendering verified correct (ns-graphify nulls render as N/A, not 0). But the Caveats section is static boilerplate — it must accept/emit run-specific caveats (§4). |
| #186 | E8 smoke hardening | **KEEP** | The most important honest finding of the mission: E2–E6 had never executed live. Three real crash fixes, each with a regression test, live-validated end-to-end. |
| #187 | Phase-III wiring + DET-1 + Track-Q normalizers | **KEEP** (flags below) | Wiring and normalizers are real and tested (suite 2092). Flags: DET-1 flips `code_index_embeddings` default to OFF, which deviates from the mission's "feature flags default to prior behavior" rule — defensible as a product directive but must be called out in the final PR body, not buried in a config comment (config.py:315–322). Also the det locate is labeled "BM25" in status/config comments; `_locate_deterministic` (native_engine.py) is token-overlap on fqn/file, not BM25 — relabel it. |

No PR warrants revert. Review discipline (Copilot triage, rework-with-tests) was genuinely applied — I4 and H1 reworks are documented catches of real defects.

---

## 2. Methodology audit — what holds up

- **Competitors pinned and actually run, not mocked.** systems.lock.json pins graphify 0.9.10 @ `20bfdf60` and CBM 0.9.0 @ `ee68144`; binaries exist on disk at `/data/ice/tools/`; raw jsonl contains 400+ real competitor query answers with per-query latencies and 23:13Z timestamps; live adapter tests exist. Verified.
- **N/A vs DNF vs 0 discipline is mostly honest.** Unsupported ops render N/A with reasons; ns-graphify's null Track-Q rows (n_supported=0) render as N/A, not 0; CBM's 35 failed neighbors queries are counted `n_unsupported`, not scored as zeros. Two gaps: those 35 CBM query failures don't appear in the DNF log despite "DNF first-class" framing, and the report never explains *why* ns-graphify has "No valid runs" (the async ingest→queryable-graph_id gap is documented only in status/runbook).
- **Quiescence claim is corroborated.** status.jsonl shows the auto-resume loop *refusing to run* for 5 consecutive checks while 16 nsbench containers were up (17:39–19:39Z), then "QUIESCENCE ACHIEVED (Ehfaz stopped all nsbench stacks; 0 competitors)" at 20:30Z, with the measured window 20:45–23:16Z. nsbench's own status went idle at 09:22Z. I consider "measured with zero competing stacks" honest. (The `-p ice` stack itself was up during competitor host-process runs; negligible for sub-3s CPU-bound tasks, worth one sentence.)
- **NS's own bad numbers are reported against self-interest.** ns-ice index 123.6s vs graphify 0.41s; blast_radius p50 10.4s; symbol_lookup structural accuracy 0.000; det nl_locate 9%. Nobody fabricating numbers reports these. This is the strongest evidence of overall honesty.
- **Shared-oracle bias** is acknowledged in the report and trackq.json metadata, and the oracle is a genuinely separate module (trackq/oracle.py). The **LSP spot-check was skipped** (pyright/gopls not installable) — that is honestly recorded in trackq.json (`lsp_agreement_pct: null`) but **missing from the report's caveats**, where the masterplan promised an agreement number. Must be stated in the report.

---

## 3. Dishonest or overstated numbers — must fix

### 3.1 nl_locate 81% hits@1 (ns-ice) is contaminated — the single worst number
The masterplan (line 130) requires CodeSearchNet-style evaluation: *"strip docstrings from N≥150 sampled functions… use the docstring as the query."* The generator samples docstrings as queries but **never strips them from the corpus the systems index** (trackq/generate.py — the word "strip" appears only in the docstring of `generate_nl_locate`). ns-ice's symbol cards explicitly embed the docstring (`_build_symbol_card`, native_engine.py:1791–1799: `parts.append(f"Doc: {docstring}")`), so for the embeddings-ON leg **the query text is verbatim inside the target's embedded card**. 81%/MRR 0.81 is substantially a self-retrieval measurement, not NL→code retrieval.

The companion claim (status 23:40Z, likely headed for the PR body): *"cloud nl_locate 81% >> det BM25 9% quantifies semantic value"* is therefore doubly overstated: the cloud leg leaks the answer text, and the det leg isn't BM25 and doesn't search text at all (token overlap on fqn/file only) — so the 81-vs-9 gap mostly quantifies "one leg had the answer embedded, the other only matched names."

**Fix (pick one):** (a) re-index a docstring-stripped copy of the corpus and re-run nl_locate for both NS legs (~30–40 min); or (b) keep the numbers but caption them "upper bound — docstrings were NOT stripped from the indexed corpus; the query text is present in the embedded symbol cards" everywhere they appear, and drop the "quantifies semantic value" framing. (a) is strongly preferred; the current number will not survive an informed reader.

### 3.2 CBM index_cold is a cache-hit, not a cold index
Raw reps: rep0 = 1.34s / 310.7MB RSS / 5.5 CPU-s (real index); reps 1–2 = 0.12s / 8.4MB / 0.03 CPU-s. The CBM adapter's `index_cold` is just `self._index(corpus)` with no prior `delete_project`, and the runner never cleans between reps — CBM sees unchanged file hashes in its persistent cache (`/data/ice/tools/cbm_cache`) and no-ops. The reported median **0.13s / 8.37MB is not a cold index** (it's internally inconsistent with index_second = 1.22s). Direction of error flatters a *competitor*, so it's not self-serving — but it's still a wrong number in the headline table.

**Fix:** call `delete_project` (the adapter already has cleanup machinery, adapter.py:446ff) before each cold rep and re-run (CBM small-py indexing takes seconds), or report rep0 as the cold number with the min/max convention explained. Check ns-graphify's cold reps for the same pattern (median 0.42 vs max 2.86 — likely warmup, but verify the NS-side store is actually reset between reps too).

### 3.3 Capability matrix vs the installed graphify binary
CAPABILITIES.md declares blast_radius N/A for graphify because a "true blast radius… requires reasoning about call chains, data flow, and change propagation." Two problems, verified against the pinned 0.9.10 CLI on this VM:
- `graphify affected "X"` exists — help text: *"reverse traversal to find nodes impacted by X."* NS's own blast_radius is `_blast_radius_bfs` — a BFS over CALLS/IMPORTS edges with **no data-flow analysis either**. Excluding graphify's `affected` on a criterion NS itself does not meet is a double standard, and its effect is to make ns-ice the *only* system in the blast_radius row (at 10.4s p50, a number that would look very different next to a millisecond-scale graph traversal).
- `graphify update <path>` exists — *"re-extract code files and update the graph"* — while CAPABILITIES.md asserts "Graphify: No incremental mode → N/A" and the report logs 6 incremental DNFs for it.

**Fix:** test both commands; if they work as documented, benchmark them (both are sub-second-class ops; this is cheap) or, at minimum, correct CAPABILITIES.md and add a report footnote: "graphify `affected`/`update` were discovered post-run and not benchmarked." The current text reads as engineered N/A. (CBM's N/As are more defensible: `detect_changes` genuinely requires uncommitted git state; `semantic_query` returns chunks not symbols — though the report should still disclose that CBM *has* local vector search that was ruled out by the no-adapter-intelligence rule, since nl_locate is NS's flagship op.)

### 3.4 NS structural-QA 0.000 needs interpretation, not just a cell
symbol_lookup hit_rate 0.000 (n_supported=200) and neighbors precision ~0.002 are *real measurements through the native surface* — I sampled raw answers: most NS symbol_lookup responses are header-only ("Code graph search results for: X" with zero hits), and when results do come back they're member FQNs (`src.click.core.CommandCollection.list_commands`) rather than the definition. The executor's own status correctly calls this "NS v1 structural WEAKER than competitors… reported honestly as the actionable finding." But the report presents bare 0.000/0.002 cells with no note. A reader can't distinguish "engine can't do symbol lookup" from "v1 query template returns empty for bare-name lookups + noisy `src.`-prefixed FQNs" (the latter, per the team's own analysis). Add exactly that sentence to the report; do not change the numbers.

---

## 4. Report caveats section is materially incomplete
ICE_BENCH_REPORT-ice-final.md's "Caveats" is four generic paragraphs. Every one of the following is documented *somewhere* (runbook, status.jsonl, trackq.json, adapter docstrings) but **absent from the report that the final PR will headline**. Disclosed-in-a-side-file is not disclosed; the report must carry:

1. **Single corpus.** The masterplan promised small+medium+large+4-language corpora; this run is small-py (click @ `8a4ce84`, 27k LOC) only. Track-Q is Python-only by construction (H3 oracle).
2. **ns-graphify partial.** Query ops show "No valid runs" because the NS-ingest→queryable-graph_id async path is incomplete — say so, in the report.
3. **Unequal sample sizes.** ns-ice ran n=80 (symbol_lookup/neighbors) and n=150 (nl_locate) vs 199–200 elsewhere, reduced because embeddings-ON nl_locate is Gemini-bound (~8s/query). Masterplan required N≥200; the det leg meets it, the cloud leg doesn't. State the per-cell Ns (they're already in trackq.json `sample_sizes`).
4. **LSP spot-check skipped** (`lsp_agreement_pct: null`) — promised oracle-agreement number does not exist for this run.
5. **store_size bases are not comparable:** ns = whole bind-mounted Neo4j+Qdrant stack (2.2GB, includes DBMS baseline and everything else in the stack — the ns-ice vs ns-ice-det values differ by 0.001%, because it's the same store measured twice); CBM = per-project `size_bytes`; graphify = output dir. Either scope NS per-code_space or annotate the column.
6. **ns-ice vs ns-ice-det framing is entirely missing.** The report never says what ns-ice-det *is*. Add the paragraph: det = default config (`code_index_embeddings=false`, DET-1 in #187 — a default flipped mid-mission as a product directive, deviating from the flags-keep-prior-behavior rule), deterministic/API-token-free, and the peer-class argument (CBM ships a local Nomic embedder; graphify is deterministic AST). The framing itself is legitimate and even conservative (det's nl_locate is 9% and that's the headline config) — but only if the report actually explains it. Also fix the "BM25" label (§1, #187 row).
7. **Query latencies are single-pass** (N queries once), not median-of-3-reps; the boilerplate "median of 3 repetitions" applies to index/snapshot ops only.
8. **Raw-row `system_version` fields contain "TODO"** (`graphify-cli@TODO+ns-api@TODO`) — the pins exist in systems.lock.json; stamp them into the rows or note the lock file as the authority. Also note the NS service was rebuilt between the ns-ice leg (20:47Z) and ns-ice-det leg (23:09Z) (det-only locate fix in between).

---

## 5. Recommendation

**Do not open the final PR as-is.** Required before opening (est. half a day, mostly re-runs of cheap ops + report text):

1. §3.1 nl_locate leakage — re-run with a docstring-stripped corpus (preferred) or demote/caveat the 81% everywhere, and delete the "quantifies semantic value" claim.
2. §3.2 CBM index_cold — re-run with per-rep `delete_project`, or report rep0 as cold.
3. §3.3 graphify `affected`/`update` — benchmark or honestly footnote; correct CAPABILITIES.md.
4. §4 — regenerate the report with the run-specific caveats (items 1–8) and the det framing paragraph; add the §3.4 interpretation sentence for NS's structural 0.000.
5. Final PR body must carry: the DET-1 default-flip disclosure, the I3-reconcile known issue, links to this review, and the headline tables *from the corrected report*.

Everything else — the 11 PRs, the engine completion, the quiescence discipline, the N/A machinery, the against-self-interest reporting — is solid and should ship. The mission's honesty infrastructure is genuinely good; the failures above are the residue of a compressed final sprint, not spin. Fix them and the PR is defensible in front of any skeptical reader, including this one.

*— Fable*
