"""CI checks for an isolated ``neuralscape-bench`` wheel installation."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
import sysconfig
from pathlib import Path
from xml.etree import ElementTree


EXPECTED_SKIP_NAME = "test_fetch_corpus_mock"
EXPECTED_SKIP_CLASS_SUFFIX = "icebench.tests.test_corpora"
EXPECTED_SKIP_REASON = "Skipping network fetch in unit tests"


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
    """Fail on test failures or any skip outside the one documented network test."""

    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    _require(cases, f"JUnit report contains no tests: {path}")

    failed = [
        case
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    ]
    _require(not failed, f"JUnit report contains {len(failed)} failed/error tests")

    skipped: list[tuple[str, str, str]] = []
    for case in cases:
        marker = case.find("skipped")
        if marker is None:
            continue
        skipped.append(
            (
                case.attrib.get("classname", ""),
                case.attrib.get("name", ""),
                marker.attrib.get("message", ""),
            )
        )

    _require(
        len(skipped) == 1,
        f"expected exactly one documented network skip, found {skipped}",
    )
    class_name, test_name, reason = skipped[0]
    _require(
        class_name.endswith(EXPECTED_SKIP_CLASS_SUFFIX)
        and test_name == EXPECTED_SKIP_NAME
        and EXPECTED_SKIP_REASON in reason,
        f"unexpected skipped test: {skipped[0]}",
    )

    print(f"JUnit policy passed: {len(cases)} tests, one documented network skip")


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
