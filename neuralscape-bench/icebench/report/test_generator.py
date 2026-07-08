"""
Tests for report generator.

Verifies honesty rules:
- N/A ≠ 0 (never fabricate numbers)
- DNF is a recorded result (not a blank)
- Medians reported with min/max
- HTML is self-contained
"""

import json
import tempfile
from pathlib import Path

import pytest

from ..schema import ResultRow, write_row
from .generator import (
    ReportConfig,
    generate_report,
    Stats,
    _nearest_rank_percentile,
)


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def synthetic_results_jsonl(temp_dir):
    """Create synthetic results JSONL with DNF and valid rows."""
    results_file = temp_dir / "results.jsonl"

    # System A: ns-ice (all OK)
    for rep in range(3):
        write_row(
            results_file,
            ResultRow(
                schema="icebench-v1",
                kind="index",
                system="ns-ice",
                system_version="1.0.0",
                corpus="small",
                repo_sha="abc123",
                op="index_cold",
                rep=rep,
                seed=42,
                wall_s=10.0 + rep,
                peak_rss_mb=100.0 + rep * 5,
                cpu_s=9.0 + rep,
                bytes=1000000,
                ok=True,
                dnf=False,
            ),
        )

    # System A: query with varying latency
    for rep in range(3):
        write_row(
            results_file,
            ResultRow(
                schema="icebench-v1",
                kind="query",
                system="ns-ice",
                system_version="1.0.0",
                corpus="small",
                repo_sha="abc123",
                op="symbol_lookup",
                rep=rep,
                seed=42,
                latency_ms=50.0 + rep * 10,
                ok=True,
                dnf=False,
            ),
        )

    # System B: cbm (DNF on one operation)
    write_row(
        results_file,
        ResultRow(
            schema="icebench-v1",
            kind="index",
            system="cbm",
            system_version="2.0.0",
            corpus="small",
            repo_sha="abc123",
            op="index_cold",
            rep=0,
            seed=42,
            wall_s=None,
            peak_rss_mb=None,
            cpu_s=None,
            ok=False,
            dnf=True,
            dnf_reason="OOM (>32GB)",
        ),
    )

    # System B: successful query
    for rep in range(3):
        write_row(
            results_file,
            ResultRow(
                schema="icebench-v1",
                kind="query",
                system="cbm",
                system_version="2.0.0",
                corpus="small",
                repo_sha="abc123",
                op="symbol_lookup",
                rep=rep,
                seed=42,
                latency_ms=100.0 + rep * 20,
                ok=True,
                dnf=False,
            ),
        )

    return results_file


@pytest.fixture
def synthetic_score_report(temp_dir):
    """Create synthetic Track-Q score report with N/A and DNF."""
    score_file = temp_dir / "score_report.json"

    score_data = {
        "ns-ice": {
            "small": {
                "nl_locate": {
                    "hits_at_1": 0.85,
                    "hits_at_5": 0.92,
                    "hits_at_10": 0.95,
                    "mrr": 0.88,
                    "hit_rate": None,  # N/A for this op
                    "precision": None,
                    "recall": None,
                    "dnf": False,
                    "dnf_reason": None,
                },
                "symbol_lookup": {
                    "hits_at_1": None,  # N/A (not a locate op)
                    "hits_at_5": None,
                    "hits_at_10": None,
                    "mrr": None,
                    "hit_rate": 0.90,
                    "precision": 0.87,
                    "recall": 0.83,
                    "dnf": False,
                    "dnf_reason": None,
                },
            }
        },
        "cbm": {
            "small": {
                "nl_locate": {
                    "hits_at_1": None,
                    "hits_at_5": None,
                    "hits_at_10": None,
                    "mrr": None,
                    "hit_rate": None,
                    "precision": None,
                    "recall": None,
                    "dnf": True,
                    "dnf_reason": "timeout (>300s)",
                },
            }
        },
    }

    with open(score_file, "w") as f:
        json.dump(score_data, f)

    return score_file


@pytest.fixture
def synthetic_capabilities():
    """Create synthetic capability matrix."""
    return {
        "ns-ice": {
            "symbol_lookup": "supported",
            "nl_locate": "supported",
            "imports_of": "supported",
            "callers_of": "N/A (graph not implemented)",
        },
        "cbm": {
            "symbol_lookup": "supported",
            "nl_locate": "supported",
            "imports_of": "N/A (not supported)",
            "callers_of": "N/A (not supported)",
        },
    }


def test_generate_report_markdown(
    temp_dir, synthetic_results_jsonl, synthetic_score_report, synthetic_capabilities
):
    """Test Markdown report generation with honesty rules."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"

    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        score_report_json=synthetic_score_report,
        capabilities_matrix=synthetic_capabilities,
        markdown_output=md_output,
        html_output=html_output,
        oracle_agreement_pct=92.5,
    )

    generate_report(config)

    # Verify Markdown was created
    assert md_output.exists()
    md_content = md_output.read_text()

    # Check header
    assert "# ICEBench Report" in md_content
    assert "ICEBench-v1" in md_content

    # Check capability matrix includes N/A cells
    assert "Capability Matrix" in md_content
    assert "N/A (graph not implemented)" in md_content
    assert "N/A (not supported)" in md_content

    # Check DNF is rendered as DNF (not 0 or blank)
    assert "DNF Log" in md_content
    assert "OOM (>32GB)" in md_content

    # Check Track-P table has median with min/max
    assert "10.00 (10.00-12.00)" in md_content or "11.00 (10.00-12.00)" in md_content

    # Check Track-Q has N/A for missing metrics (not 0)
    assert "Track Q: Accuracy" in md_content
    # N/A should appear where metrics are None
    # The nl_locate row should have N/A for hit_rate/precision/recall

    # Check caveats section
    assert "Caveats & Methodology Notes" in md_content
    assert "shared oracle bias" in md_content.lower()
    assert "92.5%" in md_content  # Oracle agreement


def test_generate_report_html_self_contained(
    temp_dir, synthetic_results_jsonl, synthetic_score_report, synthetic_capabilities
):
    """Test HTML report is self-contained (no external URLs)."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"

    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        score_report_json=synthetic_score_report,
        capabilities_matrix=synthetic_capabilities,
        markdown_output=md_output,
        html_output=html_output,
    )

    generate_report(config)

    # Verify HTML was created
    assert html_output.exists()
    html_content = html_output.read_text()

    # Check HTML structure
    assert "<!DOCTYPE html>" in html_content
    assert "<title>ICEBench Report</title>" in html_content

    # Check no external URLs (self-contained)
    # Should not have http:// or https:// links to external resources
    # (except possibly in content, but not in <script src> or <link href>)
    assert '<script src="http' not in html_content
    assert '<link href="http' not in html_content

    # Check inline styles
    assert "<style>" in html_content
    assert "color-scheme: dark" in html_content

    # Check DNF is styled with class
    assert 'class="dnf"' in html_content

    # Check N/A is styled with class
    assert 'class="na"' in html_content


def test_dnf_rendered_as_dnf_not_zero(
    temp_dir, synthetic_results_jsonl, synthetic_score_report, synthetic_capabilities
):
    """Test DNF is rendered as 'DNF', not 0 or blank."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"

    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        score_report_json=synthetic_score_report,
        capabilities_matrix=synthetic_capabilities,
        markdown_output=md_output,
        html_output=html_output,
    )

    generate_report(config)

    md_content = md_output.read_text()
    html_content = html_output.read_text()

    # Markdown: DNF should appear in tables, not 0.
    # Track-P table cols: Operation, Wall, Peak RSS, CPU, Latency, Bytes, Notes.
    assert "| index_cold | DNF | DNF | DNF | DNF | DNF |" in md_content
    assert "| cbm |" in md_content  # System with DNF (in DNF Log)

    # HTML: DNF should have dnf class (reason is HTML-escaped).
    assert '<td class="dnf">DNF</td>' in html_content
    assert "OOM (&gt;32GB)" in html_content


def test_na_rendered_as_na_not_zero(
    temp_dir, synthetic_results_jsonl, synthetic_score_report, synthetic_capabilities
):
    """Test N/A is rendered as 'N/A', not 0 or fabricated number."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"

    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        score_report_json=synthetic_score_report,
        capabilities_matrix=synthetic_capabilities,
        markdown_output=md_output,
        html_output=html_output,
    )

    generate_report(config)

    md_content = md_output.read_text()
    html_content = html_output.read_text()

    # Markdown: N/A should appear for unsupported capabilities
    assert "N/A (graph not implemented)" in md_content
    assert "N/A (not supported)" in md_content

    # HTML: N/A should have na class
    assert '<span class="na">N/A</span>' in html_content or 'class="na"' in html_content


def test_median_min_max_in_track_p(temp_dir, synthetic_results_jsonl):
    """Track-P wall time renders median with min/max range."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"

    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        markdown_output=md_output,
        html_output=html_output,
    )

    generate_report(config)

    md_content = md_output.read_text()

    # With reps 0,1,2 having wall_s of 10,11,12:
    # median = 11, min = 10, max = 12. Format: median (min-max)
    assert "11.00 (10.00-12.00)" in md_content


def test_nearest_rank_percentile_known_vector():
    """p50/p95/p99 use the nearest-rank definition (index = ceil(p*n)-1)."""
    # A known 10-element vector [1..10]. Nearest-rank:
    #   p50 -> ceil(0.50*10)-1 = 4 -> value 5
    #   p95 -> ceil(0.95*10)-1 = 9 -> value 10
    #   p99 -> ceil(0.99*10)-1 = 9 -> value 10
    vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert _nearest_rank_percentile(vec, 0.50) == 5.0
    assert _nearest_rank_percentile(vec, 0.95) == 10.0
    assert _nearest_rank_percentile(vec, 0.99) == 10.0

    # The off-by-one case from the review: n=4, p=0.5.
    # Old code did int(4*0.5)=2 -> value 30 (WRONG).
    # Nearest-rank: ceil(2.0)-1 = 1 -> value 20 (CORRECT).
    quad = [10.0, 20.0, 30.0, 40.0]
    assert _nearest_rank_percentile(quad, 0.50) == 20.0

    # p=0 returns the smallest element; p=1 returns the largest.
    assert _nearest_rank_percentile(quad, 0.0) == 10.0
    assert _nearest_rank_percentile(quad, 1.0) == 40.0

    # Unsorted input is not required — helper assumes sorted; Stats sorts first.
    stats = Stats.from_values([40.0, 10.0, 30.0, 20.0])
    assert stats.p50 == 20.0
    assert stats.min_val == 10.0
    assert stats.max_val == 40.0


def test_track_p_latency_percentiles_rendered(temp_dir):
    """Latency p50/p95/p99 appear (nearest-rank) in the Track-P table."""
    results_file = temp_dir / "results.jsonl"
    # 10 query reps with latencies 1..10 ms for a known percentile check.
    for rep, lat in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        write_row(
            results_file,
            ResultRow(
                schema="icebench-v1",
                kind="query",
                system="ns-ice",
                system_version="1.0.0",
                corpus="small",
                repo_sha="abc123",
                op="symbol_lookup",
                rep=rep,
                seed=42,
                latency_ms=float(lat),
                ok=True,
                dnf=False,
            ),
        )

    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"
    config = ReportConfig(
        results_jsonl=results_file,
        markdown_output=md_output,
        html_output=html_output,
    )
    generate_report(config)

    md_content = md_output.read_text()
    # p50=5, p95=10, p99=10 -> "5.00/10.00/10.00"
    assert "5.00/10.00/10.00" in md_content


def test_no_data_graceful(temp_dir):
    """Test report generation with no data (empty JSONL)."""
    results_file = temp_dir / "empty.jsonl"
    results_file.touch()

    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"

    config = ReportConfig(
        results_jsonl=results_file,
        markdown_output=md_output,
        html_output=html_output,
    )

    generate_report(config)

    md_content = md_output.read_text()
    html_content = html_output.read_text()

    # Should not crash, should indicate no data
    assert "ICEBench Report" in md_content
    assert "ICEBench Report" in html_content


def test_html_with_chart_js_inline(temp_dir, synthetic_results_jsonl):
    """Test HTML with Chart.js inlined (self-contained)."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"

    # Create a fake Chart.js file
    chart_js_file = temp_dir / "chart.js"
    chart_js_file.write_text("// Fake Chart.js content\nconsole.log('Chart.js loaded');")

    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        markdown_output=md_output,
        html_output=html_output,
        chart_js_path=chart_js_file,
    )

    generate_report(config)

    html_content = html_output.read_text()

    # Chart.js should be inlined
    assert "// Fake Chart.js content" in html_content
    assert "console.log('Chart.js loaded');" in html_content

    # No external script src
    assert '<script src="http' not in html_content


def test_tuple_key_grouping_with_underscore_corpus(temp_dir):
    """Corpus names containing '_' must not be mis-grouped (tuple-safe keys)."""
    results_file = temp_dir / "results.jsonl"
    # Corpus name AND op name both contain underscores.
    for rep in range(3):
        write_row(
            results_file,
            ResultRow(
                schema="icebench-v1",
                kind="index",
                system="ns-ice",
                system_version="1.0.0",
                corpus="big_repo_v2",
                repo_sha="abc123",
                op="index_cold",
                rep=rep,
                seed=42,
                wall_s=5.0 + rep,
                peak_rss_mb=50.0,
                cpu_s=4.0,
                ok=True,
                dnf=False,
            ),
        )

    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"
    config = ReportConfig(
        results_jsonl=results_file,
        markdown_output=md_output,
        html_output=html_output,
    )
    generate_report(config)

    md_content = md_output.read_text()
    html_content = html_output.read_text()

    # The corpus heading must be intact (not split at the first underscore).
    assert "Corpus: big_repo_v2" in md_content
    assert "big_repo_v2" in html_content
    # The op must appear as a single cell "index_cold" with real stats.
    assert "| index_cold |" in md_content
    assert "6.00 (5.00-7.00)" in md_content  # median 6, min 5, max 7


def test_track_q_dnf_in_notes_not_none(
    temp_dir, synthetic_results_jsonl, synthetic_score_report
):
    """Track-Q DNF reason goes in the Notes column and is never literal 'None'."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"
    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        score_report_json=synthetic_score_report,
        markdown_output=md_output,
        html_output=html_output,
    )
    generate_report(config)

    md_content = md_output.read_text()
    html_content = html_output.read_text()

    # The Track-Q table header must include a Notes column.
    assert (
        "| Operation | Hits@1 | Hits@5 | Hits@10 | MRR | Hit Rate | Precision | Recall | Notes |"
        in md_content
    )
    # The Track-Q DNF row (cbm/nl_locate) puts its reason in Notes, 8 DNF cells + notes.
    assert "| nl_locate | DNF | DNF | DNF | DNF | DNF | DNF | DNF | timeout (>300s) |" in md_content
    # A non-DNF row must have a trailing (empty) Notes cell — consistent col count.
    # ns-ice nl_locate: hits present, hit_rate/precision/recall N/A, trailing Notes.
    assert "| nl_locate | 0.850 | 0.920 | 0.950 | 0.880 | N/A | N/A | N/A |  |" in md_content
    # Never render the literal "None" as a reason.
    assert "None" not in md_content

    # HTML: DNF reason string appears (not "None"); Notes header present.
    assert "timeout (&gt;300s)" in html_content or "timeout (>300s)" in html_content
    assert "<th>Notes</th>" in html_content


def test_dnf_log_includes_track_q(
    temp_dir, synthetic_results_jsonl, synthetic_score_report
):
    """The DNF Log must include Track-Q DNFs, not just Track-P (completeness)."""
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"
    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        score_report_json=synthetic_score_report,
        markdown_output=md_output,
        html_output=html_output,
    )
    generate_report(config)

    md_content = md_output.read_text()
    html_content = html_output.read_text()

    # Track-P DNF (ns raw row): OOM on cbm/index_cold.
    assert "OOM (>32GB)" in md_content
    # Track-Q DNF (from score_report): cbm/nl_locate timeout — must appear in the log.
    assert "timeout (>300s)" in md_content
    # Both tracks tagged in the log.
    assert "| P | cbm | small | index_cold |" in md_content
    assert "| Q | cbm | small | nl_locate |" in md_content

    # HTML DNF log includes both.
    assert "OOM (&gt;32GB)" in html_content or "OOM (>32GB)" in html_content
    assert "timeout (&gt;300s)" in html_content or "timeout (>300s)" in html_content


def test_html_escapes_dnf_reason_markup(temp_dir):
    """A dnf_reason containing HTML markup must be escaped (no XSS/broken markup)."""
    results_file = temp_dir / "results.jsonl"
    write_row(
        results_file,
        ResultRow(
            schema="icebench-v1",
            kind="index",
            system="cbm",
            system_version="2.0.0",
            corpus="small",
            repo_sha="abc123",
            op="index_cold",
            rep=0,
            seed=42,
            ok=False,
            dnf=True,
            dnf_reason="<script>alert('xss')</script> & crash",
        ),
    )

    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"
    config = ReportConfig(
        results_jsonl=results_file,
        markdown_output=md_output,
        html_output=html_output,
    )
    generate_report(config)

    html_content = html_output.read_text()

    # The raw <script> tag must NOT appear unescaped.
    assert "<script>alert('xss')</script>" not in html_content
    # It must be HTML-escaped.
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_content
    assert "&amp; crash" in html_content


def test_capability_matrix_escaped_and_na(temp_dir, synthetic_results_jsonl):
    """Capability statuses are escaped; non-'supported' cells get the na class."""
    caps = {
        "ns-ice": {"op_a": "supported", "op_b": "N/A (<not> ready)"},
    }
    md_output = temp_dir / "report.md"
    html_output = temp_dir / "report.html"
    config = ReportConfig(
        results_jsonl=synthetic_results_jsonl,
        capabilities_matrix=caps,
        markdown_output=md_output,
        html_output=html_output,
    )
    generate_report(config)

    html_content = html_output.read_text()
    # Escaped angle brackets in the N/A reason.
    assert "N/A (&lt;not&gt; ready)" in html_content
    assert 'class="na">N/A (&lt;not&gt; ready)' in html_content
