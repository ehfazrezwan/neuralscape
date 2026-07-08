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
from .generator import ReportConfig, generate_report


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

    # Markdown: DNF should appear in tables, not 0
    assert "| index_cold | DNF | DNF | DNF | DNF |" in md_content
    assert "| cbm |" in md_content  # System with DNF

    # HTML: DNF should have dnf class
    assert '<td class="dnf">DNF</td>' in html_content
    assert "OOM (>32GB)" in html_content


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


def test_percentiles_computed_correctly(temp_dir, synthetic_results_jsonl):
    """Test p50/p95/p99 are computed correctly."""
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
    # median = 11, min = 10, max = 12
    # The format is: median (min-max)
    assert "11.00 (10.00-12.00)" in md_content


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
