#!/bin/sh
# Run every declared bench suite against the INSTALLED wheel, then police the
# result.  Executed from a suite root that holds copied tests plus scripts/ and
# ci/ (never the package source), so imports resolve from site-packages.  One
# list of suites for the host job and the container job.
#
#   ci_run_suite.sh <junit-out> <path-to-ci_wheel_smoke.py>
#
# Two pytest runs, merged into one JUnit report:
#   1. "copied": tests that import only absolute package names and are copied
#      next to this script (tests, icebench/tests, trackq, tokensave).
#   2. "installed": suites that live INSIDE a package and use relative imports
#      (icebench/report, the *_standalone adapters, trackb/*).  They cannot be
#      copied out of their package, so they run from the installed wheel via
#      --pyargs.  That is the isolation-honest choice: the source tree is not on
#      the path, and the wheel must carry both the code and its tests.
#      --rootdir keeps JUnit classnames dotted (trackb.mem0_locomo.tests.test_x).
#
# Deliberately NOT run:
#   - trackb/longmemeval_ref/tests/test_loader.py: reads the LongMemEval dataset,
#     which is downloaded at run time and never committed (needs network).
#   - the live-binary smokes in the *_standalone adapters skip themselves here
#     and are declared in ci/expected_skips.json.
set -eu
junit="$1"
smoke="$2"
parts="$junit.parts"
rm -rf "$parts"
mkdir -p "$parts"
purelib=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

run() {
  name="$1"
  shift
  python -m pytest -q -p no:cacheprovider --import-mode=importlib \
    -o asyncio_mode=auto \
    --strict-config --strict-markers \
    --junitxml="$parts/$name.xml" \
    "$@"
}

run copied \
  tests \
  icebench/tests \
  icebench/trackq/tests \
  icebench/tokensave/tests

run installed --rootdir "$purelib" --pyargs \
  icebench.report.test_generator \
  icebench.systems.cbm_standalone.test_adapter \
  icebench.systems.graphify_standalone.test_adapter \
  trackb.longmemeval_ref.tests.test_answer \
  trackb.longmemeval_ref.tests.test_grade \
  trackb.longmemeval_ref.tests.test_ingest \
  trackb.longmemeval_ref.tests.test_report \
  trackb.mem0_locomo.tests \
  trackb.mem0_tracka.tests

python - "$junit" "$parts" <<'PY'
import sys
from pathlib import Path
from xml.etree import ElementTree

out, parts = sys.argv[1], Path(sys.argv[2])
merged = ElementTree.Element("testsuites")
for part in sorted(parts.glob("*.xml")):
    for suite in ElementTree.parse(part).getroot().iter("testsuite"):
        merged.append(suite)
ElementTree.ElementTree(merged).write(out, encoding="utf-8", xml_declaration=True)
PY

python "$smoke" --assert-junit "$junit"
python scripts/ci_skip_gate.py --junit "$junit" --manifest ci/expected_skips.json
