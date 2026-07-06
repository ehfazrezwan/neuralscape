"""Test LoCoMo dataset loader."""

import json
from pathlib import Path

import pytest

from trackb.mem0_locomo.loader import load_locomo


@pytest.fixture
def sample_locomo_data():
    """Minimal locomo10.json structure for testing."""
    return [
        {
            "sample_id": "conv1",
            "conversation": {
                "speaker_a": "Alice",
                "session_1": [
                    {"speaker": "Alice", "text": "I got a new cat named Whiskers.", "dia_id": "D1:0"},
                    {"speaker": "Bob", "text": "That's great!", "dia_id": "D1:1"},
                ],
                "session_1_date_time": "2023-05-01",
                "session_2": [
                    {"speaker": "Alice", "text": "Whiskers broke my vase.", "dia_id": "D2:0"},
                ],
            },
            "qa": [
                {
                    "question": "What is the cat's name?",
                    "answer": "Whiskers",
                    "category": 4,
                    "evidence": ["D1:0"],
                },
                {
                    "question": "What happened on May 1st?",
                    "answer": "Alice got a cat named Whiskers",
                    "category": 2,
                    "evidence": ["D1:0", "D1:1"],
                },
                {
                    "question": "What is Alice's favorite color?",
                    "category": 5,
                    "adversarial_answer": "Not mentioned in the conversation",
                    "evidence": [],
                },
            ],
        }
    ]


def test_load_locomo_basic(tmp_path, sample_locomo_data):
    """Test loading basic dataset structure."""
    dataset_file = tmp_path / "locomo10.json"
    with open(dataset_file, "w") as f:
        json.dump(sample_locomo_data, f)

    data = load_locomo(dataset_file)

    assert data.suite == "mem0-locomo"
    assert len(data.conversations) == 1
    assert len(data.qa_items) == 3

    # Check conversation
    conv = data.conversations[0]
    assert conv.conv_id == "conv1"
    assert len(conv.sessions) == 2

    # Check sessions
    s1 = conv.sessions[0]
    assert s1.session_id == "1"
    assert s1.date == "2023-05-01"
    assert len(s1.turns) == 2
    assert s1.turns[0].role == "user"  # Alice is speaker_a
    assert s1.turns[1].role == "assistant"  # Bob

    # Check QA items
    qa = data.qa_items
    assert qa[0].qa_id == "conv1-qa0"
    assert qa[0].question == "What is the cat's name?"
    assert qa[0].gold_answer == "Whiskers"
    assert qa[0].qtype == "4-single-hop"
    assert qa[0].is_abstention is False

    # Adversarial question
    assert qa[2].qtype == "5-adversarial"
    assert qa[2].is_abstention is True
    assert qa[2].gold_answer == "Not mentioned in the conversation"


def test_load_locomo_evidence_parsing(tmp_path):
    """Test evidence field parsing (list or string repr)."""
    data = [
        {
            "sample_id": "c1",
            "conversation": {
                "speaker_a": "A",
                "session_1": [{"speaker": "A", "text": "hello", "dia_id": "D1:0"}],
            },
            "qa": [
                {"question": "Q1", "answer": "A1", "category": 1, "evidence": ["D1:0"]},
                {"question": "Q2", "answer": "A2", "category": 1, "evidence": "['D1:0']"},  # string repr
                {"question": "Q3", "answer": "A3", "category": 1, "evidence": "D1:0"},  # plain string
            ],
        }
    ]

    dataset_file = tmp_path / "test.json"
    with open(dataset_file, "w") as f:
        json.dump(data, f)

    result = load_locomo(dataset_file)

    # All should parse evidence correctly
    assert result.qa_items[0].evidence_turn_ids == ("D1:0",)
    assert result.qa_items[1].evidence_turn_ids == ("D1:0",)
    assert result.qa_items[2].evidence_turn_ids == ("D1:0",)


def test_load_locomo_session_id_extraction(tmp_path):
    """Test session ID extraction from evidence (D12:3 -> session '12')."""
    data = [
        {
            "sample_id": "c1",
            "conversation": {
                "speaker_a": "A",
                "session_12": [{"speaker": "A", "text": "hello", "dia_id": "D12:0"}],
            },
            "qa": [
                {"question": "Q", "answer": "A", "category": 1, "evidence": ["D12:0", "D12:1"]},
            ],
        }
    ]

    dataset_file = tmp_path / "test.json"
    with open(dataset_file, "w") as f:
        json.dump(data, f)

    result = load_locomo(dataset_file)

    # Should extract session "12" from "D12:0"
    assert result.qa_items[0].evidence_session_ids == ("12",)


def test_load_locomo_image_captions(tmp_path):
    """Test BLIP caption folding into turn text."""
    data = [
        {
            "sample_id": "c1",
            "conversation": {
                "speaker_a": "A",
                "session_1": [
                    {"speaker": "A", "text": "Check this out", "blip_caption": "a cute dog", "dia_id": "D1:0"},
                    {"speaker": "A", "text": "", "blip_caption": "sunset photo", "dia_id": "D1:1"},
                ],
            },
            "qa": [],
        }
    ]

    dataset_file = tmp_path / "test.json"
    with open(dataset_file, "w") as f:
        json.dump(data, f)

    result = load_locomo(dataset_file)

    turns = result.conversations[0].sessions[0].turns
    assert "[shares a photo: a cute dog]" in turns[0].content
    assert "Check this out" in turns[0].content
    assert turns[1].content == "A: [shares a photo: sunset photo]"


def test_load_locomo_category_labels(tmp_path):
    """Test category number to label mapping."""
    data = [
        {
            "sample_id": "c1",
            "conversation": {
                "speaker_a": "A",
                "session_1": [{"speaker": "A", "text": "hi", "dia_id": "D1:0"}],
            },
            "qa": [
                {"question": "Q1", "answer": "A1", "category": 1},
                {"question": "Q2", "answer": "A2", "category": 2},
                {"question": "Q3", "answer": "A3", "category": 3},
                {"question": "Q4", "answer": "A4", "category": 4},
                {"question": "Q5", "category": 5, "adversarial_answer": "Not mentioned"},
            ],
        }
    ]

    dataset_file = tmp_path / "test.json"
    with open(dataset_file, "w") as f:
        json.dump(data, f)

    result = load_locomo(dataset_file)

    qtypes = [qa.qtype for qa in result.qa_items]
    assert qtypes == [
        "1-multi-hop",
        "2-temporal",
        "3-open-domain",
        "4-single-hop",
        "5-adversarial",
    ]


def test_load_locomo_rejects_invalid_structure(tmp_path):
    """Test that loader rejects non-list or empty data."""
    # Not a list
    with open(tmp_path / "bad1.json", "w") as f:
        json.dump({"not": "a list"}, f)

    with pytest.raises(ValueError, match="Expected list of conversations"):
        load_locomo(tmp_path / "bad1.json")

    # Empty list
    with open(tmp_path / "bad2.json", "w") as f:
        json.dump([], f)

    with pytest.raises(ValueError, match="Expected at least 1 conversation"):
        load_locomo(tmp_path / "bad2.json")
