# results/

**These files are a historical record. They are not publishable numbers.**
Benchmark runs are currently paused (see `../README.md`), and nothing in this
directory should be quoted as a current Neuralscape result without the context
below.

## What is here

| Files | What they are |
|---|---|
| `accuracy-*.json`, `accuracy-battery-*.md` | Aggregate output of the memory-accuracy battery from early July 2026 runs. Each JSON records the Neuralscape commit it ran against in its `ns_commit` field, and each battery markdown states it in its header. Every one of those commits is older than current `dev`. The retrieval, ingestion, judging and suite-loader code has changed since, so these files do not describe today's behaviour. |
| `landed-store-verification.md` | A write-up of one investigation (whether scores were depressed by answers running before async writes landed). It describes a specific run condition, not a general result. |
| `cost-estimates.json` | Analytic token and cost estimates from `--phase estimate`. Model-based, not measured spend. |
| `sample-run.json`, `sample-compare.json`, `sample-stress.json` | Sample outputs of the latency runner, the A/B orchestrator and the stress test. They exist so the dashboard has something to render; their `label` and `git_commit` fields name the older commit they came from. Fixtures, not measurements to cite. |

## Known problems with specific files

- **The ConvoMem score from the July battery is void.** ConvoMem question ids
  collided across categories in the harness at the time, which corrupted
  resume and judging. The harness has since been fixed (ids are namespaced per
  category), but the committed ConvoMem results were produced before the fix and
  were not regenerated. This applies to both `accuracy-convomem-*.json` files and
  to the ConvoMem row in the battery markdown files.
- **BEAM has no valid retrieval recall.** The committed BEAM result carries no
  retrieval figure, and the battery tables show a dash for it. The harness now
  reports BEAM retrieval as incompatible (gold ids are user-level) rather than
  as a number; do not infer one.
- **DMR files** predate later fixes to how DMR turns and session times are
  ingested, so they were produced under a different ingestion than the current
  code would use.
- **`accuracy-battery-20260704T035754Z.md`** says "No paid runs yet". That was
  true when that file was written (all suites pending); it is not true of the
  directory as a whole, which later gained the runs above.
- Battery tables list competitors' self-reported figures. Those are on their own
  configurations and are context only, never a controlled comparison.

## Rules for this directory

- Do not edit or delete these files to make them look current. Add a note here
  instead.
- Raw per-question output (`results/raw/`, which contains conversation text) is
  gitignored and never committed.
- A new run's output should not be added until runs resume and the result has
  been reviewed for publication.
