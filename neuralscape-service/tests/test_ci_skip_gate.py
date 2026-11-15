"""Unexpected-skip gate for the service test suites.

A green pytest run can hide skipped tests (missing optional extras, absent
corpora, a silently excluded suite).  This module reads the junit report of a
suite run and compares its skips with the declared inventory in
``.github/ci/expected_skips.json`` (version 2: that file names the excluded
suites and points at ``expected_skips.d/``, which holds ONE JSON file per
declared skip so parallel PRs that add or remove skips never conflict).  A run is rejected when it has failures or
errors, ran nothing, skipped a test that is not declared, no longer skips a
declared test, skipped for a different reason, or ran a suite that the
inventory says is excluded.

CI invokes it as a script (it needs only the standard library)::

    python3 -I neuralscape-service/tests/test_ci_skip_gate.py \
        --junit junit.xml --manifest .github/ci/expected_skips.json --profile host

To (re)generate entry files from a junit report, e.g. for a slice that adds
many skips at once::

    python3 -I neuralscape-service/tests/test_ci_skip_gate.py \
        --junit junit.xml --profile container \
        --write-entries .github/ci/expected_skips.d --why "<one line>"

Generated entries are a starting point: review every file (test id, reason
substring, profiles, why) before committing.  The mode merges profiles into
existing files and never deletes any.

Exit status: 0 accepted, 1 rejected, 2 unusable input.  The functions below
are also exercised by the tests at the bottom, which run in the packaged test
image as well (they never read repository files outside ``tests/``).
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MANIFEST_VERSION = 2  # version 1 (inline "profiles") is still read for the unit tests below
KNOWN_PROFILES = ("host", "container")
ENTRY_KEYS = {"test", "reason_contains", "profiles", "why"}


class GateInputError(Exception):
    """The junit report or the manifest cannot be interpreted."""


def _check_excluded(obj):
    excluded = obj.get("excluded_suites")
    if not isinstance(excluded, list) or not all(isinstance(x, str) and x for x in excluded):
        raise GateInputError("excluded_suites must be a list of non-empty strings")
    return excluded


def entry_filename(test):
    """File name for a declared skip: the test id with ``::`` replaced by ``--``."""

    return re.sub(r"[^A-Za-z0-9._\-\[\]]", "_", test.replace("::", "--")) + ".json"


def load_entries_dir(directory, profile):
    """Return {test: reason_contains} for ``profile`` from a per-entry directory."""

    directory = Path(directory)
    if not directory.is_dir():
        raise GateInputError(f"entries directory missing: {directory}")
    expected, origin = {}, {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if path.suffix != ".json" or not path.is_file():
            raise GateInputError(f"{path.name}: only .json entry files belong in {directory.name}/")
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GateInputError(f"{path.name}: unreadable entry: {exc}") from exc
        if not isinstance(entry, dict):
            raise GateInputError(f"{path.name}: entry must be a JSON object")
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            raise GateInputError(f"{path.name}: unknown keys {sorted(unknown)}")
        test, reason, why = entry.get("test"), entry.get("reason_contains"), entry.get("why")
        for key, value in (("test", test), ("reason_contains", reason), ("why", why)):
            if not (isinstance(value, str) and value.strip()):
                raise GateInputError(f"{path.name}: '{key}' must be a non-empty string")
        profiles = entry.get("profiles")
        if not (isinstance(profiles, list) and profiles and all(isinstance(x, str) for x in profiles)):
            raise GateInputError(f"{path.name}: 'profiles' must be a non-empty list")
        bad = [x for x in profiles if x not in KNOWN_PROFILES]
        if bad:
            raise GateInputError(f"{path.name}: unknown profile(s) {bad}; known: {list(KNOWN_PROFILES)}")
        if len(set(profiles)) != len(profiles):
            raise GateInputError(f"{path.name}: duplicate profile in {profiles}")
        if profile not in profiles:
            continue
        if test in expected:
            raise GateInputError(f"duplicate skip entry {test!r} in {origin[test]} and {path.name}")
        expected[test] = reason
        origin[test] = path.name
    return expected, origin


def load_manifest(path_or_obj, profile):
    """Return (expected_skips, excluded_suites) for ``profile``."""

    obj, base = path_or_obj, None
    if not isinstance(obj, dict):
        base = Path(path_or_obj).resolve().parent
        try:
            obj = json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GateInputError(f"manifest unreadable: {exc}") from exc
    if not isinstance(obj, dict) or obj.get("version") not in (1, MANIFEST_VERSION):
        raise GateInputError(f"manifest version must be {MANIFEST_VERSION}")
    excluded = _check_excluded(obj)
    if profile not in KNOWN_PROFILES:
        raise GateInputError(f"profile {profile!r} is not declared")
    if obj["version"] == MANIFEST_VERSION:
        entries_dir = obj.get("entries_dir")
        if not (isinstance(entries_dir, str) and entries_dir):
            raise GateInputError("manifest needs an 'entries_dir'")
        if base is None:
            raise GateInputError("'entries_dir' needs a manifest file to resolve against")
        expected, _ = load_entries_dir(base / entries_dir, profile)
        return expected, excluded
    profiles = obj.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise GateInputError(f"profile {profile!r} is not declared")
    entries = profiles[profile].get("skips") if isinstance(profiles[profile], dict) else None
    if not isinstance(entries, list):
        raise GateInputError(f"profile {profile!r} needs a skips list")
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise GateInputError("skip entries must be objects")
        test, reason = entry.get("test"), entry.get("reason_contains")
        if not (isinstance(test, str) and test and isinstance(reason, str) and reason):
            raise GateInputError("skip entries need non-empty 'test' and 'reason_contains'")
        if test in expected:
            raise GateInputError(f"duplicate skip entry {test!r}")
        expected[test] = reason
    return expected, excluded


def load_junit(path_or_text):
    """Return a list of (test_id, classname, outcome, detail) from junit XML.

    ``outcome`` is ``passed``, ``skipped``, ``failed`` or ``error``; ``detail``
    is the skip type, message and element text (the reason) for skips.
    """

    try:
        if isinstance(path_or_text, str) and path_or_text.lstrip().startswith("<"):
            root = ET.fromstring(path_or_text)
        else:
            root = ET.parse(str(path_or_text)).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GateInputError(f"junit report unreadable: {exc}") from exc
    cases = []
    for case in root.iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        test_id = f"{classname}::{name}"
        if case.find("failure") is not None:
            cases.append((test_id, classname, "failed", ""))
        elif case.find("error") is not None:
            cases.append((test_id, classname, "error", ""))
        elif case.find("skipped") is not None:
            node = case.find("skipped")
            # The message of a collection skip is always "collection skipped"; the real reason is the element text.
            text = " ".join((node.text or "").split())
            cases.append((test_id, classname, "skipped", f"{node.get('type', '')} {node.get('message', '')} {text}".strip()))
        else:
            cases.append((test_id, classname, "passed", ""))
    return cases


def _module_paths(classname):
    parts = classname.split(".")
    return {"/".join(parts[:n]) + ".py" for n in range(1, len(parts) + 1)}


def check(cases, expected, excluded):
    """Return a list of violation strings (empty means the run is accepted)."""

    problems = []
    if not cases:
        problems.append("the report contains no test cases")
    for test_id, _, outcome, _ in cases:
        if outcome in ("failed", "error"):
            problems.append(f"{outcome}: {test_id}")
    seen = {}
    for test_id, classname, outcome, detail in cases:
        if outcome == "skipped":
            seen[test_id] = detail
        hit = _module_paths(classname) & set(excluded)
        if hit:
            problems.append(f"excluded suite ran: {sorted(hit)[0]} ({test_id})")
    for test_id, detail in sorted(seen.items()):
        if test_id not in expected:
            problems.append(f"unexpected skip: {test_id} [{detail.strip()}]")
        elif expected[test_id] not in detail:
            problems.append(f"skip reason changed: {test_id} [{detail.strip()}] lacks {expected[test_id]!r}")
    for test_id in sorted(expected):
        if test_id not in seen:
            problems.append(f"declared skip did not skip: {test_id}")
    return problems


def summarize(cases):
    counts = {"passed": 0, "skipped": 0, "failed": 0, "error": 0}
    for _, _, outcome, _ in cases:
        counts[outcome] += 1
    return counts


def reason_snippet(detail, width=48):
    """A stable leading substring of a skip's reason (type/collection prefixes dropped)."""

    text = " ".join(detail.split())
    if text.startswith("collection skipped"):
        text = text[len("collection skipped"):].strip()
    words = text.split(" ", 1)
    if words and words[0].startswith("pytest."):
        text = words[1] if len(words) > 1 else ""
    if not text:
        raise GateInputError("a skipped case has no reason text to pin")
    return text[:width].rstrip()


def write_entries(cases, directory, profile, why):
    """Create/extend entry files for every skipped case of a junit run.

    Existing files gain ``profile``; nothing is ever deleted or rewritten
    otherwise.  Returns the list of paths written.  The output is a draft for
    human review, not a verdict.
    """

    if profile not in KNOWN_PROFILES:
        raise GateInputError(f"profile {profile!r} is not declared")
    if not why.strip():
        raise GateInputError("--why must state why these skips are expected")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    skipped = {test_id: detail for test_id, _, outcome, detail in cases if outcome == "skipped"}
    names = {}
    for test_id in skipped:
        name = entry_filename(test_id)
        if name in names:
            raise GateInputError(f"{test_id!r} and {names[name]!r} map to the same file {name}")
        names[name] = test_id
    written = []
    for name, test_id in sorted(names.items()):
        path = directory / name
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("test") != test_id:
                raise GateInputError(f"{name} holds {entry.get('test')!r}, not {test_id!r}")
            if profile in entry.get("profiles", []):
                continue
            entry["profiles"] = sorted(set(entry.get("profiles", [])) | {profile})
        else:
            entry = {
                "test": test_id,
                "reason_contains": reason_snippet(skipped[test_id]),
                "profiles": [profile],
                "why": why,
            }
        path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--junit", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--write-entries", metavar="DIR", help="generate entry files from the junit skips instead of checking")
    parser.add_argument("--why", default="", help="one-line justification stored in generated entries")
    args = parser.parse_args(argv)
    if args.write_entries:
        try:
            written = write_entries(load_junit(args.junit), args.write_entries, args.profile, args.why)
        except GateInputError as exc:
            print(f"skip-gate: {exc}", file=sys.stderr)
            return 2
        print(f"skip-gate: wrote {len(written)} entr{'y' if len(written) == 1 else 'ies'} to {args.write_entries}; REVIEW each before committing")
        return 0
    if not args.manifest:
        parser.error("--manifest is required unless --write-entries is given")
    try:
        expected, excluded = load_manifest(args.manifest, args.profile)
        cases = load_junit(args.junit)
    except GateInputError as exc:
        print(f"skip-gate: {exc}", file=sys.stderr)
        return 2
    problems = check(cases, expected, excluded)
    counts = summarize(cases)
    print(
        f"skip-gate[{args.profile}]: {counts['passed']} passed, {counts['skipped']} skipped, "
        f"{counts['failed']} failed, {counts['error']} errors; "
        f"{len(expected)} skips declared, excluded suites: {', '.join(excluded) or 'none'}"
    )
    for problem in problems:
        print(f"skip-gate: {problem}", file=sys.stderr)
    return 1 if problems else 0


# --------------------------------------------------------------------------
# Tests (plain asserts; deliberately no pytest import so the CLI stays stdlib)
# --------------------------------------------------------------------------

MANIFEST = {
    "version": 1,
    "excluded_suites": ["tests/test_async_pipeline.py"],
    "profiles": {
        "host": {"skips": [{"test": "tests.test_a::test_optional", "reason_contains": "extra not installed"}]},
        "container": {"skips": []},
    },
}


def _report(*cases):
    body = []
    for classname, name, kind, message in cases:
        if kind == "skiptext":  # a collection skip: generic message, the reason is the element text
            reason, _, text = message.partition("|")
            body.append(
                f'<testcase classname="{classname}" name="{name}"><skipped message="{reason}">{text}</skipped></testcase>'
            )
        elif kind == "pass":
            body.append(f'<testcase classname="{classname}" name="{name}" time="0.1"/>')
        elif kind == "skip":
            body.append(
                f'<testcase classname="{classname}" name="{name}"><skipped type="pytest.skip" '
                f'message="{message}"/></testcase>'
            )
        elif kind == "xfail":
            body.append(
                f'<testcase classname="{classname}" name="{name}"><skipped type="pytest.xfail" '
                f'message="{message}"/></testcase>'
            )
        else:
            body.append(f'<testcase classname="{classname}" name="{name}"><{kind} message="x"/></testcase>')
    return '<?xml version="1.0"?><testsuites><testsuite name="pytest">' + "".join(body) + "</testsuite></testsuites>"


OK_REPORT = _report(
    ("tests.test_a", "test_one", "pass", ""),
    ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
)


def _problems(report, profile="host", manifest=MANIFEST):
    expected, excluded = load_manifest(manifest, profile)
    return check(load_junit(report), expected, excluded)


def test_exact_declared_skip_set_is_accepted():
    assert _problems(OK_REPORT) == []


def test_undeclared_skip_is_rejected():
    report = _report(
        ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
        ("tests.test_b", "test_new", "skip", "something else"),
    )
    assert _problems(report) == ["unexpected skip: tests.test_b::test_new [pytest.skip something else]"]


def test_declared_skip_that_runs_or_disappears_is_rejected():
    ran = _report(("tests.test_a", "test_optional", "pass", ""))
    gone = _report(("tests.test_a", "test_one", "pass", ""))
    for report in (ran, gone):
        assert _problems(report) == ["declared skip did not skip: tests.test_a::test_optional"]


def test_a_collection_skip_is_judged_by_its_element_text_not_its_generic_message():
    manifest = dict(MANIFEST, profiles={"host": {"skips": [
        {"test": "::tests.test_story", "reason_contains": "absent module(s): contracts_x"},
    ]}})
    absent = _report(("", "tests.test_story", "skiptext", "collection skipped|('f.py', 3, 'Skipped: absent module(s): contracts_x')"))
    broken = _report(("", "tests.test_story", "skiptext",
                      "collection skipped|('f.py', 3, \"Skipped: could not import 'contracts_x': No module named 'dep'\")"))
    assert _problems(absent, manifest=manifest) == []
    assert _problems(broken, manifest=manifest) != []  # a present-but-broken module is not an absent one


def test_skip_for_a_different_reason_is_rejected():
    report = _report(("tests.test_a", "test_optional", "skip", "cloud credentials missing"))
    problems = _problems(report)
    assert len(problems) == 1 and problems[0].startswith("skip reason changed: tests.test_a::test_optional")


def test_xfail_is_counted_as_a_skip_that_must_be_declared():
    report = _report(
        ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
        ("tests.test_a", "test_known_bad", "xfail", "reason"),
    )
    assert _problems(report) == ["unexpected skip: tests.test_a::test_known_bad [pytest.xfail reason]"]


def test_failures_and_errors_are_rejected_even_with_declared_skips():
    for kind in ("failure", "error"):
        report = _report(
            ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
            ("tests.test_a", "test_bad", kind, ""),
        )
        assert _problems(report) == [f"{'failed' if kind == 'failure' else 'error'}: tests.test_a::test_bad"]


def test_empty_report_is_rejected():
    problems = _problems('<testsuites><testsuite name="pytest"></testsuite></testsuites>', "container")
    assert problems == ["the report contains no test cases"]


def test_excluded_suite_that_ran_is_rejected_at_module_or_class_level():
    for classname in ("tests.test_async_pipeline", "tests.test_async_pipeline.TestPipeline"):
        report = _report(
            ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
            (classname, "test_x", "pass", ""),
        )
        problems = _problems(report)
        assert len(problems) == 1 and problems[0].startswith("excluded suite ran: tests/test_async_pipeline.py")


def test_similarly_named_module_is_not_mistaken_for_an_excluded_suite():
    report = _report(
        ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
        ("tests.test_async_pipeline_helpers", "test_x", "pass", ""),
    )
    assert _problems(report) == []


def test_profiles_are_independent():
    assert _problems(OK_REPORT, "container") == ["unexpected skip: tests.test_a::test_optional [pytest.skip optional extra not installed]"]


def test_unusable_inputs_raise_instead_of_passing():
    bad_manifests = [
        {},
        dict(MANIFEST, version=2),
        dict(MANIFEST, excluded_suites="tests/x.py"),
        dict(MANIFEST, profiles={}),
        dict(MANIFEST, profiles={"host": {}}),
        dict(MANIFEST, profiles={"host": {"skips": [{"test": "t", "reason_contains": ""}]}}),
        dict(MANIFEST, profiles={"host": {"skips": [{"test": "t", "reason_contains": "r"}] * 2}}),
    ]
    for manifest in bad_manifests:
        try:
            load_manifest(manifest, "host")
        except GateInputError:
            continue
        raise AssertionError(f"accepted unusable manifest {manifest!r}")
    for report in ("<testsuites><oops", "/nonexistent/junit.xml"):
        try:
            load_junit(report)
        except GateInputError:
            continue
        raise AssertionError(f"accepted unusable report {report!r}")


def test_cli_exit_status_distinguishes_accept_reject_and_unusable(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST), encoding="utf-8")
    good, bad = tmp_path / "good.xml", tmp_path / "bad.xml"
    good.write_text(OK_REPORT, encoding="utf-8")
    bad.write_text(_report(("tests.test_b", "test_new", "skip", "x")), encoding="utf-8")

    def run(junit, profile="host", manifest_path=manifest):
        return subprocess.run(
            [sys.executable, "-I", __file__, "--junit", str(junit), "--manifest", str(manifest_path), "--profile", profile],
            capture_output=True,
            text=True,
        )

    accepted = run(good)
    assert accepted.returncode == 0 and "1 passed, 1 skipped" in accepted.stdout
    rejected = run(bad)
    assert rejected.returncode == 1 and "unexpected skip" in rejected.stderr
    assert run(tmp_path / "missing.xml").returncode == 2
    assert run(good, profile="nope").returncode == 2


def _entry(test="tests.test_a::test_optional", reason="extra not installed", profiles=("host",), why="missing extra", **extra):
    return dict({"test": test, "reason_contains": reason, "profiles": list(profiles), "why": why}, **extra)


def _manifest_dir(tmp_path, entries):
    """Write a version-2 manifest plus entry files; ``entries`` maps file name -> object or raw text."""

    entries_dir = tmp_path / "expected_skips.d"
    entries_dir.mkdir()
    for name, entry in entries.items():
        (entries_dir / name).write_text(entry if isinstance(entry, str) else json.dumps(entry), encoding="utf-8")
    manifest = tmp_path / "expected_skips.json"
    manifest.write_text(
        json.dumps({"version": 2, "excluded_suites": ["tests/test_async_pipeline.py"], "entries_dir": "expected_skips.d"}),
        encoding="utf-8",
    )
    return manifest


def _load_error(manifest, profile="host"):
    try:
        load_manifest(manifest, profile)
    except GateInputError as exc:
        return str(exc)
    raise AssertionError("manifest was accepted")


def test_directory_manifest_loads_entries_per_profile(tmp_path):
    manifest = _manifest_dir(tmp_path, {
        "a.json": _entry(profiles=("host", "container")),
        "b.json": _entry(test="tests.test_b::test_x", reason="no corpus", profiles=("container",)),
    })
    assert load_manifest(manifest, "host")[0] == {"tests.test_a::test_optional": "extra not installed"}
    assert load_manifest(manifest, "container")[0] == {
        "tests.test_a::test_optional": "extra not installed",
        "tests.test_b::test_x": "no corpus",
    }


def test_directory_manifest_rejects_duplicates_naming_both_files(tmp_path):
    message = _load_error(_manifest_dir(tmp_path, {"one.json": _entry(), "two.json": _entry()}))
    assert "duplicate" in message and "one.json" in message and "two.json" in message


def test_directory_manifest_allows_the_same_test_in_disjoint_profiles(tmp_path):
    manifest = _manifest_dir(tmp_path, {"h.json": _entry(profiles=("host",)), "c.json": _entry(profiles=("container",))})
    assert list(load_manifest(manifest, "host")[0]) == ["tests.test_a::test_optional"]


def test_directory_manifest_rejects_unknown_profile(tmp_path):
    message = _load_error(_manifest_dir(tmp_path, {"p.json": _entry(profiles=("host", "laptop"))}))
    assert "p.json" in message and "laptop" in message


def test_directory_manifest_rejects_malformed_entries_naming_the_file(tmp_path):
    cases = {
        "broken.json": "{not json",
        "list.json": "[]",
        "nowhy.json": {k: v for k, v in _entry().items() if k != "why"},
        "typo.json": _entry(reasons="x"),
        "noprofiles.json": _entry(profiles=()),
    }
    for name, content in cases.items():
        message = _load_error(_manifest_dir(Path(tempfile.mkdtemp()), {name: content}))
        assert name in message, (name, message)


def test_directory_manifest_rejects_stray_files_and_a_missing_directory(tmp_path):
    assert "notes.txt" in _load_error(_manifest_dir(tmp_path, {"notes.txt": "hi"}))
    bare = tmp_path / "bare"
    bare.mkdir()
    manifest = bare / "expected_skips.json"
    manifest.write_text(json.dumps({"version": 2, "excluded_suites": [], "entries_dir": "gone"}), encoding="utf-8")
    assert "missing" in _load_error(manifest)


def test_empty_entries_directory_is_valid_only_while_nothing_skips(tmp_path):
    manifest = _manifest_dir(tmp_path, {})
    expected, excluded = load_manifest(manifest, "host")
    assert expected == {}
    clean = _report(("tests.test_a", "test_one", "pass", ""))
    assert check(load_junit(clean), expected, excluded) == []
    skipping = _report(("tests.test_a", "test_one", "skip", "boom"))
    assert check(load_junit(skipping), expected, excluded) == ["unexpected skip: tests.test_a::test_one [pytest.skip boom]"]


def test_removing_an_entry_file_is_all_it_takes_to_retire_a_skip(tmp_path):
    manifest = _manifest_dir(tmp_path, {"a.json": _entry()})
    (tmp_path / "expected_skips.d" / "a.json").unlink()
    expected, excluded = load_manifest(manifest, "host")
    assert check(load_junit(OK_REPORT), expected, excluded) == [
        "unexpected skip: tests.test_a::test_optional [pytest.skip optional extra not installed]"
    ]


def test_entry_file_names_follow_the_test_id():
    assert entry_filename("tests.test_a::test_x") == "tests.test_a--test_x.json"
    assert entry_filename("::tests.test_story") == "--tests.test_story.json"
    assert entry_filename("tests.t::test_p[a/b c]") == "tests.t--test_p[a_b_c].json"


def test_write_entries_generates_reviewable_files_and_merges_profiles(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(_report(
        ("tests.test_a", "test_one", "pass", ""),
        ("tests.test_a", "test_optional", "skip", "optional extra not installed on this host"),
        ("tests.test_b", "test_bad", "xfail", "strict reproduction of a pinned bug that is long enough to be cut"),
        ("", "tests.test_story", "skiptext", "collection skipped|('f.py', 3, 'Skipped: absent module(s): contracts_x')"),
    ), encoding="utf-8")
    out = tmp_path / "expected_skips.d"

    def run(profile, *extra):
        return subprocess.run(
            [sys.executable, "-I", __file__, "--junit", str(report), "--profile", profile, "--write-entries", str(out), *extra],
            capture_output=True,
            text=True,
        )

    assert run("host").returncode == 2  # --why is mandatory
    first = run("host", "--why", "pinned for the test")
    assert first.returncode == 0 and "wrote 3 entries" in first.stdout and "REVIEW" in first.stdout
    assert run("container", "--why", "ignored for existing files").returncode == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == ["--tests.test_story.json", "tests.test_a--test_optional.json", "tests.test_b--test_bad.json"]
    entry = json.loads((out / "tests.test_b--test_bad.json").read_text(encoding="utf-8"))
    assert entry["profiles"] == ["container", "host"] and entry["why"] == "pinned for the test"
    assert len(entry["reason_contains"]) <= 48 and entry["reason_contains"].startswith("strict reproduction")
    story = json.loads((out / "--tests.test_story.json").read_text(encoding="utf-8"))
    assert story["reason_contains"].startswith("('f.py', 3, 'Skipped: absent module(s)")
    # The generated directory, used as a manifest, accepts the run it was generated from.
    manifest = tmp_path / "expected_skips.json"
    manifest.write_text(json.dumps({"version": 2, "excluded_suites": [], "entries_dir": "expected_skips.d"}), encoding="utf-8")
    for profile in ("host", "container"):
        expected, excluded = load_manifest(manifest, profile)
        assert check(load_junit(str(report)), expected, excluded) == []


def test_repository_manifest_is_well_formed_when_present():
    path = Path(__file__).resolve().parents[2] / ".github" / "ci" / "expected_skips.json"
    if not path.is_file():  # the packaged test image carries only the service tree
        return
    for profile in ("host", "container"):
        load_manifest(path, profile)
    # One file per entry, named after its test id, so retiring a skip is `git rm` of exactly one file.
    for entry_path in (path.parent / "expected_skips.d").glob("*.json"):
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        assert entry_path.name == entry_filename(entry["test"]), entry_path.name


if __name__ == "__main__":
    sys.exit(main())
