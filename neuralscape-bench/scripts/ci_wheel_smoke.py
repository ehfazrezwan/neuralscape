"""CI checks for an isolated ``neuralscape-bench`` wheel installation."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
import sysconfig
from pathlib import Path
from xml.etree import ElementTree


# Every suite the CI job claims to run must actually appear in the JUnit report.
# A suite dropped from the pytest argument list would otherwise pass silently.
# Skips are policed separately by scripts/ci_skip_gate.py.
REQUIRED_SUITE_PREFIXES = (
    "tests.",
    "icebench.tests.",
    "icebench.trackq.tests.",
    "icebench.tokensave.tests.",
    "icebench.report.test_generator",
    "icebench.systems.cbm_standalone.test_adapter",
    "icebench.systems.graphify_standalone.test_adapter",
    "trackb.longmemeval_ref.tests.",
    "trackb.mem0_locomo.tests.",
    "trackb.mem0_tracka.tests.",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def smoke_installed_wheel() -> None:
    """Verify metadata, isolation, import provenance, and strict validation."""

    _require(
        sys.version_info[:2] == (3, 11),
        f"CI must exercise the declared Python 3.11 floor, got {sys.version}",
    )

    distribution = importlib.metadata.distribution("neuralscape-bench")
    requirements = distribution.requires or []
    normalized_requirements = [
        requirement.replace(" ", "").lower() for requirement in requirements
    ]
    _require(
        any(
            requirement.startswith("pydantic")
            and ">=2.0.0" in requirement
            and "<3.0.0" in requirement
            for requirement in normalized_requirements
        ),
        f"wheel metadata lacks the direct Pydantic v2 requirement: {requirements}",
    )

    pydantic_version = importlib.metadata.version("pydantic")
    _require(
        pydantic_version.split(".", 1)[0] == "2",
        f"wheel resolved an unsupported Pydantic version: {pydantic_version}",
    )
    _require(
        importlib.util.find_spec("contracts_common") is None,
        "product service contract module is visible in the wheel environment",
    )

    # The trackq tests skipif tree-sitter is missing; the wheel must carry it so
    # they run instead of skipping (ci_skip_gate.py enforces the same from JUnit).
    from icebench.trackq.oracle import TREE_SITTER_AVAILABLE

    _require(
        TREE_SITTER_AVAILABLE,
        "tree-sitter is not importable from the wheel environment",
    )

    import trackb

    trackb_path = Path(trackb.__file__).resolve()
    _require(
        Path(sysconfig.get_paths()["purelib"]).resolve() in trackb_path.parents,
        f"trackb was not imported from installed site-packages: {trackb_path}",
    )

    from pydantic import ValidationError
    from neuralscape_bench import run_manifest
    from neuralscape_bench.run_manifest import ConcurrencySetting

    module_path = Path(run_manifest.__file__).resolve()
    purelib_path = Path(sysconfig.get_paths()["purelib"]).resolve()
    try:
        module_path.relative_to(purelib_path)
    except ValueError as exc:
        raise AssertionError(
            f"run_manifest was not imported from installed site-packages: {module_path}"
        ) from exc

    distributed_module = Path(
        distribution.locate_file("neuralscape_bench/run_manifest.py")
    ).resolve()
    _require(
        module_path == distributed_module,
        f"imported module does not belong to installed wheel: {module_path}",
    )

    distributed_files = distribution.files or []
    index_entry = next(
        (
            entry
            for entry in distributed_files
            if str(entry) == "neuralscape_bench/static/index.html"
        ),
        None,
    )
    _require(index_entry is not None, "wheel metadata omits dashboard static/index.html")
    distributed_index = Path(distribution.locate_file(index_entry)).resolve()
    _require(
        distributed_index.is_file(),
        f"packaged dashboard index is missing: {distributed_index}",
    )
    index_text = distributed_index.read_text(encoding="utf-8")
    _require(
        "<title>Neuralscape Benchmark</title>" in index_text,
        "packaged dashboard index could not be loaded or has unexpected content",
    )

    from neuralscape_bench import dashboard

    _require(
        dashboard.STATIC_DIR.resolve() == distributed_index.parent,
        f"dashboard does not use wheel-packaged assets: {dashboard.STATIC_DIR}",
    )

    setting = ConcurrencySetting(
        schema_version="candidate-v1",
        scope="clean-wheel-driver",
        value=1,
    )
    _require(setting.value == 1, "valid strict counter was not preserved")

    try:
        ConcurrencySetting(
            schema_version="candidate-v1",
            scope="clean-wheel-driver",
            value=True,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("strict counter accepted a boolean")

    print("wheel smoke passed")
    print("python", sys.version.split()[0])
    print("pydantic", pydantic_version)
    print("module", module_path)
    print("dashboard-index", distributed_index)
    print("requires-dist", requirements)


def assert_junit(path: Path) -> None:
    """Fail on test failures, an empty report, or a required suite that did not run."""

    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    _require(cases, f"JUnit report contains no tests: {path}")

    failed = [
        case
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    ]
    _require(not failed, f"JUnit report contains {len(failed)} failed/error tests")

    classnames = {case.attrib.get("classname", "") for case in cases}
    missing = [
        prefix
        for prefix in REQUIRED_SUITE_PREFIXES
        if not any(name.startswith(prefix) for name in classnames)
    ]
    _require(not missing, f"JUnit report has no tests from required suites: {missing}")

    print(f"JUnit policy passed: {len(cases)} tests, all required suites present")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assert-junit",
        type=Path,
        help="validate a pytest JUnit report instead of running the wheel smoke",
    )
    args = parser.parse_args()

    if args.assert_junit is None:
        smoke_installed_wheel()
    else:
        assert_junit(args.assert_junit)


if __name__ == "__main__":
    main()
