"""Tiny synthetic fixtures mirroring each suite's REAL on-disk schema.

Shapes were verified against the actual downloaded datasets (including the
quirks: LoCoMo's str-encoded evidence lists and string categories, DMR's
bare {"text"} turns, BEAM's turn markers). Content is synthetic — no
dataset text is committed to the repo.
"""

LOCOMO_SAMPLE = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Ana",
            "speaker_b": "Ben",
            "session_1_date_time": "1:00 pm on 5 May, 2023",
            "session_1": [
                {"speaker": "Ana", "dia_id": "D1:1", "text": "I adopted a beagle named Kiwi yesterday."},
                {"speaker": "Ben", "dia_id": "D1:2", "text": "Congrats! I just started pottery classes."},
                {"speaker": "Ana", "dia_id": "D1:3", "text": "",
                 "img_url": "http://x/1.jpg", "blip_caption": "a small dog on a sofa"},
            ],
            "session_2_date_time": "4:10 pm on 20 May, 2023",
            "session_2": [
                {"speaker": "Ana", "dia_id": "D2:1", "text": "Kiwi chewed my running shoes today."},
                {"speaker": "Ben", "dia_id": "D2:2", "text": "Ha! My first vase came out crooked."},
            ],
        },
        "qa": [
            {"question": "What is the name of Ana's dog?", "answer": "Kiwi",
             "evidence": ["D1:1"], "category": 4},
            {"question": "When did Ana adopt her dog?", "answer": "4 May 2023",
             "evidence": "['D1:1']", "category": "2"},
            {"question": "What instrument does Ben play?",
             "adversarial_answer": "Not mentioned in the conversation",
             "evidence": [], "category": 5},
        ],
    },
]

LONGMEMEVAL_SAMPLE = [
    {
        "question_id": "q_001",
        "question_type": "single-session-user",
        "question": "What breed is my dog?",
        "answer": "Beagle",
        "question_date": "2023/05/30 (Tue) 10:00",
        "haystack_dates": ["2023/05/05 (Fri) 13:00", "2023/05/20 (Sat) 16:10"],
        "haystack_session_ids": ["sess_a", "sess_b"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I adopted a beagle named Kiwi.", "has_answer": True},
                {"role": "assistant", "content": "Congratulations on the adoption!"},
            ],
            [
                {"role": "user", "content": "What's a good pasta recipe?"},
                {"role": "assistant", "content": "Try cacio e pepe."},
            ],
        ],
        "answer_session_ids": ["sess_a"],
    },
    {
        "question_id": "q_002_abs",
        "question_type": "multi-session",
        "question": "What color is my car?",
        "answer": "The user never mentioned a car.",
        "question_date": "2023/06/01 (Thu) 09:00",
        "haystack_dates": ["2023/05/05 (Fri) 13:00"],
        "haystack_session_ids": ["sess_c"],
        "haystack_sessions": [
            [{"role": "user", "content": "I like cycling to work."}],
        ],
        "answer_session_ids": [],
    },
]

DMR_SAMPLE = [
    {
        "metadata": {"initial_data_id": "valid_9", "session_id": 4},
        "previous_dialogs": [
            {
                "time_back": "3 days",
                "dialog": [
                    {"text": "Hi! I train falcons for a living."},
                    {"text": "Wow, that is a rare job. I bake sourdough."},
                ],
            },
        ],
        "dialog": [
            {"text": "My favorite falcon is named Ziggy.", "id": "Speaker 1"},
            {"text": "Ziggy is a great name for a bird.", "id": "Speaker 2"},
        ],
        "self_instruct": {
            "B": "What do I do for a living?",
            "A": "You train falcons.",
        },
    },
]

BEAM_QUERIES = [
    {
        "id": "1_knowledge_update_0",
        "query": "Which city am I moving to?",
        "gold_answers": ["Lisbon"],
        "gold_ids": ["doc-2"],
        "user_id": "1",
        "meta": {"question_category": "knowledge_update", "conversation_id": "1",
                 "rubric": ["Mentions Lisbon as the final destination"]},
    },
    {
        "id": "1_abstention_0",
        "query": "What did I say about my sister's wedding?",
        "gold_answers": ["No information about a wedding."],
        "gold_ids": ["doc-1"],
        "user_id": "1",
        "meta": {"question_category": "abstention", "conversation_id": "1",
                 "why_unanswerable": "No wedding is mentioned."},
    },
]

BEAM_DOCUMENTS = [
    {
        "id": "doc-1",
        "user_id": "1",
        "timestamp": "2024-03-15",
        "content": "[March-15-2024 | Turn 0] User: I am planning a move to Porto.\n"
                   "[Turn 1] Assistant: Porto is lovely in spring.",
    },
    {
        "id": "doc-2",
        "user_id": "1",
        "timestamp": "2024-04-02",
        "content": "[April-02-2024 | Turn 0] User: Change of plans, I am moving to Lisbon instead.\n"
                   "[Turn 1] Assistant: Noted — Lisbon it is.",
    },
]

CONVOMEM_FILE = {
    "evidence_items": [
        {
            "question": "What color do I use for hot leads?",
            "answer": "Green",
            "category": "Professional Life",
            "personId": "abc-123",
            "message_evidences": [{"speaker": "User", "text": "I mark hot leads in green."}],
            "conversations": [
                {
                    "id": "conv-a",
                    "containsEvidence": True,
                    "messages": [
                        {"speaker": "User", "text": "I mark hot leads in green."},
                        {"speaker": "Assistant", "text": "Green for hot leads, got it."},
                    ],
                },
                {
                    "id": "conv-b",
                    "containsEvidence": False,
                    "messages": [
                        {"speaker": "User", "text": "Draft an email to the vendor."},
                        {"speaker": "Assistant", "text": "Here is a draft."},
                    ],
                },
            ],
        },
    ],
}

MEMBENCH_FILE = {
    "movie": [
        {
            "tid": 0,
            "message_list": [
                [
                    {"mid": 0, "user": "I loved The Godfather.", "assistant": "A classic drama.",
                     "time": "morning", "place": "home"},
                    {"mid": 1, "user": "Also loved 12 Angry Men.", "assistant": "Another great drama.",
                     "time": "noon", "place": "office"},
                ],
            ],
            "QA": {
                "qid": 0,
                "question": "What kind of movies do I prefer?",
                "answer": "Drama",
                "choices": {"A": "Musical", "B": "Drama", "C": "Horror", "D": "Sci-fi"},
                "ground_truth": "B",
                "target_step_id": [[0, 0], [1, 0]],
            },
        },
    ],
}
