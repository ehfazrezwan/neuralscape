"""Unit tests for T1.1 retro fixes — require_speaker gating + parser hardening.

Covers:
- require_speaker=False → byte-identical to original 82c80ab prompt
- require_speaker=True → conversational prompt with correct section ordering
- Parser fixes: single-space speaker, malformed when-suffix, colon false-positives
- Legacy parser backward-compat (no speaker pollution on solo-coding memories)
"""

from __future__ import annotations

import pytest

from prompts import (
    CODING_ASSISTANT_EXTRACTION_PROMPT,
    ParsedFact,
    build_extraction_messages,
    parse_extraction_response,
    parse_extraction_response_rich,
)


# The original pre-T1.1 prompt at git ref 82c80ab (byte-exact reference)
ORIGINAL_PROMPT_82C80AB = """You are a memory extraction engine for an AI assistant. The user may be coding, doing research, running meetings, writing, or any other knowledge work — extract memories that fit the broad context, not just code.

Analyze the conversation below and extract distinct, factual memories about the user, their preferences, projects, and environment.

Each extracted fact MUST be prefixed with a category tag in square brackets. Use ONLY these categories:

- [preference] — Personal preferences: how the user likes to work, communicate, and consume information
- [personal_fact] — Personal details about the user: name, timezone, role, team, working hours
- [technical_skill] — Skills and proficiencies the user has, technical or otherwise
- [domain_knowledge] — Subject-matter knowledge the user has accumulated (industry, market, scientific, organizational)
- [tech_stack] — Tools, systems, or platforms used in this project
- [convention] — Norms and conventions adopted by this project (code style, communication, naming, process)
- [architecture] — Structural decisions about this project (system design, org structure, information architecture)
- [dependency] — External dependencies of this project (libraries, vendors, blocking teams, pinned versions)
- [decision] — Decisions made — with the why, not just the what
- [interaction] — Notable events: meetings, conversations, calls, demos
- [workflow] — Recurring multi-step processes (git flow, deployment, review, weekly rituals)
- [procedure] — Step-by-step how-tos for repeatable tasks
- [task_context] — Active work-in-progress: current goals, recent state, blockers — short-lived

Rules:
1. Extract ONLY factual, reusable information. Skip greetings, acknowledgments, and transient dialogue.
2. Each fact should be a standalone sentence that makes sense without the conversation context.
3. Be specific. "Uses Python" is too vague. "Uses Python 3.12 with FastAPI for backend services" is good.
4. Deduplicate — don't extract the same fact twice with different wording.
5. If a fact could belong to multiple categories, pick the most specific one.
6. For project-specific facts (tech_stack, convention, architecture, dependency), mention the project name if known.
7. NEVER extract raw tool operations, shell commands run, files edited/read/written, git operations, terminal output, or build/test execution logs — these are ephemeral actions, not reusable knowledge.
8. NEVER extract information only meaningful in the current session context (e.g., "currently running tests", "just fixed a bug in X file").

Respond with a JSON object:
{
    "facts": [
        "[category] Fact description here",
        "[category] Another fact here"
    ]
}

If no memorable facts can be extracted, return: {"facts": []}

CONVERSATION:
"""


class TestRequireSpeakerGating:
    """Verify require_speaker flag controls which prompt is used."""

    def test_require_speaker_false_is_byte_identical_to_original(self):
        """DEFAULT (False) must be byte-identical to pre-T1.1 prompt."""
        messages = [{"role": "user", "content": "test"}]
        result = build_extraction_messages(messages, require_speaker=False)
        prompt = result[0]["content"]
        # Extract just the prompt part (before the conversation)
        prompt_part = prompt.split("user: test")[0]
        assert prompt_part == ORIGINAL_PROMPT_82C80AB

    def test_constant_is_byte_identical_to_original(self):
        """The exported CODING_ASSISTANT_EXTRACTION_PROMPT constant must be the original."""
        assert CODING_ASSISTANT_EXTRACTION_PROMPT == ORIGINAL_PROMPT_82C80AB

    def test_require_speaker_true_has_attribution_section(self):
        """require_speaker=True must use the conversational variant."""
        messages = [{"role": "user", "content": "test"}]
        result = build_extraction_messages(messages, require_speaker=True)
        prompt = result[0]["content"]
        assert "SPEAKER ATTRIBUTION" in prompt
        assert "ALL participants" in prompt
        assert "family, friends, possessions, pets" in prompt  # broadened personal_fact

    def test_require_speaker_true_coding_hygiene_before_specificity(self):
        """Hygiene rules must come BEFORE the SPECIFICITY section (LLM recency bias)."""
        messages = [{"role": "user", "content": "test"}]
        result = build_extraction_messages(messages, require_speaker=True)
        prompt = result[0]["content"]
        hygiene1_pos = prompt.find("NEVER extract raw tool operations")
        hygiene2_pos = prompt.find("NEVER extract information only meaningful in the current session context")
        specificity_pos = prompt.find("SPECIFICITY")
        assert hygiene1_pos > 0, "Hygiene rule 1 must be present"
        assert hygiene2_pos > 0, "Hygiene rule 2 must be present"
        assert specificity_pos > 0, "SPECIFICITY section must be present"
        assert hygiene1_pos < specificity_pos, "Hygiene rule 1 must come before SPECIFICITY"
        assert hygiene2_pos < specificity_pos, "Hygiene rule 2 must come before SPECIFICITY"

    def test_default_is_false(self):
        """When require_speaker is omitted, default is False (original prompt)."""
        messages = [{"role": "user", "content": "test"}]
        result = build_extraction_messages(messages)
        prompt = result[0]["content"]
        prompt_part = prompt.split("user: test")[0]
        assert prompt_part == ORIGINAL_PROMPT_82C80AB


class TestParserFixes:
    """Parser hardening — single-space, malformed when, colon false-positives."""

    def test_single_space_speaker_parsed_correctly(self):
        """'Ana: fact' (exactly one space) must parse as speaker."""
        response = '{"facts": ["[personal_fact] Ana: owns a dog"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker == "Ana"
        assert pf.content == "owns a dog"

    def test_malformed_when_empty_value(self):
        """(when: ) with empty value → occurred_at=None, content clean."""
        response = '{"facts": ["[interaction] user: attended meeting (when: )"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker == "user"
        assert pf.content == "attended meeting"
        assert pf.occurred_at is None
        assert "(when:" not in pf.content

    def test_malformed_when_missing_closing_paren(self):
        """(when: value without closing paren → not matched, stays in content."""
        response = '{"facts": ["[interaction] user: met (when: yesterday"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker == "user"
        # The malformed suffix isn't matched by the regex, so it stays in content
        assert "when: yesterday" in pf.content or "(when: yesterday" in pf.content

    def test_colon_false_positive_ratio(self):
        """'3:1' has no space after colon → not a speaker."""
        response = '{"facts": ["[preference] uses a 3:1 water to rice ratio"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker is None
        assert "3:1" in pf.content

    def test_colon_false_positive_time(self):
        """'10:00 AM' has no space after colon → not a speaker."""
        response = '{"facts": ["[interaction] meeting at 10:00 AM"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker is None
        assert "10:00 AM" in pf.content

    def test_colon_false_positive_url(self):
        """'https://example.com' → not a speaker."""
        response = '{"facts": ["[tech_stack] uses https://api.example.com endpoint"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker is None
        assert "https://api.example.com" in pf.content


class TestLegacyParserBackwardCompat:
    """Legacy parser must NOT pollute solo-coding memories with speaker prefixes."""

    def test_no_speaker_prefix_clean_fact(self):
        """Input without speaker → output without speaker."""
        response = '{"facts": ["[preference] Prefers tabs over spaces"]}'
        [(category, content)] = parse_extraction_response(response)
        assert category == "preference"
        assert content == "Prefers tabs over spaces"
        assert not content.startswith("user:")

    def test_speaker_folded_when_present(self):
        """When speaker IS present (conversational mode), it folds into content."""
        response = '{"facts": ["[preference] user: Prefers tabs over spaces"]}'
        [(category, content)] = parse_extraction_response(response)
        assert category == "preference"
        assert content == "user: Prefers tabs over spaces"

    def test_no_bogus_speaker_from_false_positive(self):
        """A sentence fragment shouldn't be mis-parsed as speaker (corrected test)."""
        # The existing test had "uses microservices: ..." asserting it WAS parsed
        # as speaker — that was a false-positive lock-in. The correct behavior:
        # "uses microservices" is a verb phrase, not a name/role, so it should
        # NOT be parsed as speaker (but the current permissive regex allows it).
        # At minimum, ensure the legacy parser doesn't ADD a bogus prefix when
        # the input had none.
        response = '{"facts": ["[architecture] Chose microservices over monolith"]}'
        [(category, content)] = parse_extraction_response(response)
        assert category == "architecture"
        assert content == "Chose microservices over monolith"
        # The fact must not gain a speaker prefix that wasn't in the input
        assert not content.startswith("Chose:")


class TestMultipleColonsCorrected:
    """Fix the false-positive test that locked in wrong behavior."""

    def test_sentence_fragment_no_longer_parsed_as_speaker(self):
        """'uses microservices: ...' is a verb phrase, NOT a speaker (F-2).

        The char-class regex still matches it, but the F-2 plausibility gate
        rejects it (not a role word / labelled speaker / proper name), so the
        whole remainder stays as content — junk never reaches metadata.speaker.
        Previously this test locked in the permissive behavior
        (speaker == "uses microservices"); F-2 corrects it.
        """
        response = '{"facts": ["[architecture] uses microservices: API: Gateway: Auth"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker is None
        # No speaker split → the full remainder is preserved as content.
        assert pf.content == "uses microservices: API: Gateway: Auth"
        # The legacy parser (flag OFF) is unchanged: content stays byte-identical.
        from prompts import parse_extraction_response
        [(cat, content)] = parse_extraction_response(response)
        assert cat == "architecture"
        assert content == "uses microservices: API: Gateway: Auth"

    def test_noun_phrase_not_parsed_as_speaker(self):
        """'Naming convention: ...' (mixed-case noun phrase) is not a speaker (F-2)."""
        response = '{"facts": ["[convention] Naming convention: use snake_case"]}'
        [pf] = parse_extraction_response_rich(response)
        assert pf.speaker is None
        assert pf.content == "Naming convention: use snake_case"

    def test_plausible_speakers_still_parse(self):
        """F-2 must NOT drop real speakers: role words, labelled, proper names."""
        cases = {
            '{"facts": ["[preference] Ana: prefers tabs"]}': ("Ana", "prefers tabs"),
            '{"facts": ["[preference] user: prefers tabs"]}': ("user", "prefers tabs"),
            '{"facts": ["[preference] assistant: suggests spaces"]}': ("assistant", "suggests spaces"),
            '{"facts": ["[preference] Speaker 1: likes dark mode"]}': ("Speaker 1", "likes dark mode"),
            '{"facts": ["[personal_fact] Dr. Smith: runs marathons"]}': ("Dr. Smith", "runs marathons"),
            '{"facts": ["[personal_fact] Bob Jones: lives in Berlin"]}': ("Bob Jones", "lives in Berlin"),
        }
        for response, (exp_speaker, exp_content) in cases.items():
            [pf] = parse_extraction_response_rich(response)
            assert pf.speaker == exp_speaker, f"{response} → {pf.speaker!r}"
            assert pf.content == exp_content

    def test_is_plausible_speaker_unit(self):
        """Direct unit coverage of the F-2 plausibility gate."""
        from prompts import _is_plausible_speaker
        for good in ("user", "assistant", "Ana", "Bob Jones", "Dr. Smith", "Speaker 1", "team", "Person 2"):
            assert _is_plausible_speaker(good), good
        for bad in ("uses microservices", "Naming convention", "chose to refactor the whole thing",
                    "", "   ", "x" * 41, "runs the pipeline nightly"):
            assert not _is_plausible_speaker(bad), bad
