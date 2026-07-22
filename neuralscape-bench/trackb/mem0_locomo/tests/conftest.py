"""Test fixtures for trackb.mem0_locomo tests.

Ensures trackb modules can be imported in tests.
"""

import sys
from pathlib import Path

# Add neuralscape-bench to path so we can import neuralscape_bench
bench_root = Path(__file__).parent.parent.parent.parent
if str(bench_root) not in sys.path:
    sys.path.insert(0, str(bench_root))
