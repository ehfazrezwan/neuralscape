"""Measured accuracy of the modality post-check on a fixed labelled sample.

40 (source sentence, distilled fact, flagged?) cases written by hand: ``flagged=True`` means the fact
must NOT be labelled explicit (an upgraded modal, or an obligation the source does not state);
``flagged=False`` means a faithful restatement that must stay explicit. Run with ``-s`` to see the counts:

    uv run pytest tests/test_ingest_modality_sample.py -s

False negative = an upgrade that was labelled explicit (the harm #444 is about).
False positive = a faithful fact needlessly relabelled inductive.
"""

from __future__ import annotations

from ingest.modality import ModalitySource, modality_upgraded

# (category, source, fact, flagged)
SAMPLE: list[tuple[str, str, str, bool]] = [
    # ── upgrades: soft modal in the source, strong modal in the fact ──
    ("upgrade", "Session leads should capture an observation first.", "Session leads must capture an observation first.", True),
    ("upgrade", "Each follow-up should have a named owner.", "Every follow-up must have a designated owner.", True),
    ("upgrade", "All participants should review the written summary.", "All participants are required to review the written summary.", True),
    ("upgrade", "Teams may rotate the pager weekly.", "Teams must rotate the pager weekly.", True),
    ("upgrade", "We recommend pinning dependency versions.", "Dependency versions are mandatory to pin.", True),
    ("upgrade", "Reviewers could check the audit trail.", "Reviewers shall check the audit trail.", True),
    ("upgrade", "It is a good idea to back up the database nightly.", "The database must be backed up nightly.", True),
    ("upgrade", "Ideally commits are small and focused.", "Commits are required to be small and focused.", True),
    ("upgrade", "Contributors might add tests for new endpoints.", "Contributors need to add tests for new endpoints.", True),
    ("upgrade", "Staff should wear badges on site.", "Staff have to wear badges on site.", True),
    ("upgrade", "Deploys should happen on weekdays.", "Deploys must happen on weekdays.", True),
    ("upgrade", "Owners should rotate API keys quarterly.", "Owners are obliged to rotate API keys quarterly.", True),
    # ── upgrades where a distant must exists elsewhere in the document (the old false-negative) ──
    ("distant-must", "The vault door must stay locked. Marmots should hibernate through winter.", "Marmots must hibernate through the winter.", True),
    ("distant-must", "Releases must pass CI. Release notes should list breaking changes.", "Release notes must list breaking changes.", True),
    ("distant-must", "Staff should review the summary; the build must pass CI.", "Staff must review the summary.", True),
    ("distant-must", "Everyone must lock the door!Staff should review the written summary before sign-off.", "Staff must review the written summary before sign-off.", True),
    # ── obligations the source does not state at all ──
    ("unstated", "We like astronomy. Cats are nice.", "Quasars must be catalogued nightly.", True),
    ("unstated", "The cafe opens at nine.", "Customers must be seated by the host.", True),
    # ── faithful restatements ──
    ("faithful", "Session leads should capture an observation first.", "Session leads should capture an observation first.", False),
    ("faithful", "Reviewers must check the audit trail before approving.", "Reviewers must check the audit trail before approving.", False),
    ("faithful", "The design review requires compatibility evidence before release.", "Compatibility evidence is required before release by the design review.", False),
    ("faithful", "Key observations need durable text.", "Key observations must be written down as durable text.", False),
    ("faithful", "Owners shall rotate API keys quarterly.", "Owners must rotate API keys quarterly.", False),
    ("faithful", "It is mandatory to log every deploy.", "Every deploy must be logged.", False),
    ("faithful", "Staff should wear badges; visitors must sign the register.", "Visitors must sign the register.", False),
    ("faithful", "A later session may revise a tentative finding.", "A later session may revise a tentative finding.", False),
    ("faithful", "The whiteboard is erased after the workshop.", "The whiteboard is erased after each workshop.", False),
    ("faithful", "Teams should rotate the pager weekly.", "Teams should rotate the pager weekly.", False),
    # ── negations (not obligations) ──
    ("negation", "Reviewers should check the audit trail. Optional: skipping is fine.", "Reviewers need not check the audit trail.", False),
    ("negation", "A second reviewer is not required for small changes.", "Small changes do not need a second reviewer.", False),
    ("negation", "There is no need to restart the service after a config change.", "Restarting the service after a config change is not required.", False),
    ("negation", "Reviewers need not check the audit trail for the budget.", "Reviewers must check the audit trail for the budget.", True),
    ("negation", "Reviewers must not approve their own requests.", "Reviewers must not approve their own requests.", False),
    ("negation", "Reviewers should not approve their own requests.", "Reviewers must not approve their own requests.", True),
    # ── faithful imperatives and normative statements ──
    ("imperative", "Employees are to submit expense reports monthly.", "Employees must submit expense reports monthly.", False),
    ("imperative", "Employees: wear badges at all times.", "Employees must wear badges at all times.", False),
    ("imperative", "Minimum length: 12 characters.", "Passwords must have a minimum length of 12 characters.", False),
    ("imperative", "Contractors got to complete the safety induction first.", "Contractors must complete the safety induction first.", False),
    ("imperative", "Passwords need at least 12 characters.", "Passwords must have at least 12 characters.", False),
    ("imperative", "Workshop transcript: Mina says that session leads should capture an observation before suggesting a follow-up.", "Session leads must capture an observation before suggesting a follow-up.", True),
]


# Added after the re-review: cases the first 40 lacked (soft modals inside limit/label clauses, typographic
# apostrophes). Reported separately so the original 40-pair numbers stay comparable.
EXTRA: list[tuple[str, str, str, bool]] = [
    ("soft-in-limit", "Passwords should be at least 12 characters.", "Passwords must be at least 12 characters.", True),
    ("soft-in-limit", "Services may have a maximum of three retries.", "Services must have a maximum of three retries.", True),
    ("soft-in-label", "Note: Users should register accounts.", "Users must register accounts.", True),
    ("soft-in-label", "Tip: teams may rotate the pager weekly.", "Teams must rotate the pager weekly.", True),
    ("soft-in-label", "Optional: staff could wear badges on site.", "Staff must wear badges on site.", True),
    ("curly", "Users should not share passwords.", "Users mustn\u2019t share passwords.", True),
    ("curly", "Users may register accounts.", "Users don\u2019t need to register accounts.", False),
    ("curly", "Users mustn\u2019t share passwords.", "Users must not share passwords.", False),
    ("limit-faithful", "Passwords have a minimum of twelve characters.", "Passwords must have a minimum of twelve characters.", False),
    ("label-faithful", "Employees: wear badges at all times.", "Employees must wear badges at all times.", False),
]


def test_sample_is_forty_cases():
    assert len(SAMPLE) == 40


def test_measured_false_positive_and_false_negative_counts(capsys):
    _measure(SAMPLE, capsys, "fixed 40-pair sample", pin_fp=["Every deploy must be logged.", "Small changes do not need a second reviewer."])


def test_measured_counts_on_the_added_cases(capsys):
    _measure(EXTRA, capsys, "added re-review cases", pin_fp=[])


def _measure(sample, capsys, title, *, pin_fp):
    fp, fn, rows = [], [], []
    for category, source, fact, flagged in sample:
        got = modality_upgraded(fact, ModalitySource(source))
        rows.append((category, flagged, got))
        if got and not flagged:
            fp.append((category, fact))
        if flagged and not got:
            fn.append((category, fact))
    n_up = sum(1 for *_, f in sample if f)
    n_ok = len(sample) - n_up
    with capsys.disabled():
        print(f"\nmodality_upgraded, {title} ({len(sample)} pairs): "
              f"false negatives {len(fn)}/{n_up} (upgrades labelled explicit), "
              f"false positives {len(fp)}/{n_ok} (faithful facts relabelled inductive)")
        for c, f in fn:
            print(f"  FN [{c}] {f}")
        for c, f in fp:
            print(f"  FP [{c}] {f}")
        by_cat: dict[str, list[int]] = {}
        for c, want, got in rows:
            by_cat.setdefault(c, [0, 0])[0 if want == got else 1] += 1
        print("  per category (correct, wrong): " + ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(by_cat.items())))
    # Regression bounds. No upgrade may slip through as explicit. The two known false positives are
    # heavy paraphrases (too little lexical overlap to compare, so they are "unknown" and conservatively
    # labelled inductive: documented limit, not tuned away on this sample).
    assert not fn, fn
    assert sorted(f for _, f in fp) == sorted(pin_fp), fp
