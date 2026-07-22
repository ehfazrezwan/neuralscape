"""Track B: mem0 LoCoMo evaluation harness with Neuralscape backend.

Implements mem0's published LoCoMo evaluation methodology faithfully:
- Ingest multi-session conversations into NS (per-speaker turns)
- Answer each QA by retrieving top-k memories + LLM generation
- Judge answers with an LLM judge (gemini-3.1-flash-lite @ temp=0)
- Report category-wise + overall accuracy (LoCoMo 5 categories)

This is Track B: competitors' methodology, NS as backend. Keep results
clearly separated from Track A (our NSBench harness).
"""

__version__ = "0.1.0"
