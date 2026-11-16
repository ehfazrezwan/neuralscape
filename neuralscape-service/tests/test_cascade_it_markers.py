"""The real-store cascade test selects its victim by a marker a real extractor keeps.

The first real run failed because the marker carried a hex suffix the LLM
dropped. These tests pin the selection against that exact extractor output.
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "integration_tests" / "cascade_markers.py"
_spec = importlib.util.spec_from_file_location("cascade_markers", _PATH)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

# Exact output of the real extractor in the failing gate run.
REAL_EXTRACTION = [
    ("a", "The user owns a dog named Quillfeather."),
    ("b", "The user has lived in Reykjavik since March."),
    ("c", "The user prefers smoked lapsang souchong tea."),
]


def test_marker_is_letters_only():
    assert cm.VICTIM_MARKER.isalpha()


def test_selects_exactly_one_victim_from_the_real_extractor_output():
    assert cm.select_victims(REAL_EXTRACTION) == ["a"]


def test_match_is_case_insensitive_and_whole_word():
    assert cm.mentions_victim("my dog QUILLFEATHER is loud")
    assert cm.mentions_victim("quillfeather")
    assert not cm.mentions_victim("Quillfeathers")      # a different word
    assert not cm.mentions_victim("")
    assert not cm.mentions_victim(None)


def test_old_hex_suffixed_marker_would_not_match_the_extractor_output():
    # why the previous selection failed: the extractor dropped the suffix
    assert "quillfeather005b22" not in REAL_EXTRACTION[0][1].lower()


def test_conversation_carries_the_marker_once_and_the_other_two_facts():
    text = " ".join(m["content"] for m in cm.conversation())
    assert text.count(cm.VICTIM_MARKER) == 1
    assert "Reykjavik" in text and "lapsang" in text
