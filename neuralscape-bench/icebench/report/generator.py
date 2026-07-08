"""
Report generator — Markdown + self-contained HTML.

Consumes:
- Raw results JSONL (icebench-v1 schema via icebench.schema.read_rows)
- ScoreReport JSON dict (from H3's score_results, shape described below)
- systems.lock.json + corpora.lock.json (for methodology tags)
- Optional capability matrix input

ScoreReport shape (per system×corpus×op):
{
  "system_name": {
    "corpus_name": {
      "op_name": {
        "hits_at_1": 0.85,
        "hits_at_5": 0.92,
        "hits_at_10": 0.95,
        "mrr": 0.88,
        "hit_rate": 0.90,
        "precision": 0.87,
        "recall": 0.83,
        "dnf": false,
        "dnf_reason": null
      }
    }
  }
}
"""

from pathlib import Path
from typing import Any
import json
from dataclasses import dataclass
from collections import defaultdict
import statistics

from ..schema import read_rows, ResultRow


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    results_jsonl: Path
    score_report_json: Path | None = None
    systems_lock_json: Path | None = None
    corpora_lock_json: Path | None = None
    capabilities_matrix: dict[str, dict[str, str]] | None = None
    markdown_output: Path = Path("/data/ice/reports/ICE_BENCH_REPORT.md")
    html_output: Path = Path("/data/ice/reports/ice_bench_report.html")
    chart_js_path: Path | None = None
    quiescence_statement: str = "Machine in single-user mode, no background services."
    oracle_agreement_pct: float | None = None


@dataclass
class Stats:
    """Statistics for a metric."""

    median: float
    min_val: float
    max_val: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: list[float]) -> "Stats":
        """Compute stats from a list of values."""
        if not values:
            raise ValueError("Cannot compute stats from empty list")

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return cls(
            median=statistics.median(sorted_vals),
            min_val=min(sorted_vals),
            max_val=max(sorted_vals),
            p50=sorted_vals[int(n * 0.50)] if n > 0 else 0.0,
            p95=sorted_vals[int(n * 0.95)] if n > 1 else sorted_vals[-1],
            p99=sorted_vals[int(n * 0.99)] if n > 1 else sorted_vals[-1],
        )


class ReportGenerator:
    """Generates Markdown and HTML reports from benchmark results."""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.rows: list[ResultRow] = []
        self.score_report: dict[str, Any] = {}
        self.systems_meta: dict[str, Any] = {}
        self.corpora_meta: dict[str, Any] = {}
        self.capabilities: dict[str, dict[str, str]] = config.capabilities_matrix or {}

    def load_data(self) -> None:
        """Load all input data."""
        # Load results JSONL
        self.rows = list(read_rows(self.config.results_jsonl))

        # Load score report if provided
        if self.config.score_report_json and self.config.score_report_json.exists():
            with open(self.config.score_report_json) as f:
                self.score_report = json.load(f)

        # Load systems metadata if provided
        if self.config.systems_lock_json and self.config.systems_lock_json.exists():
            with open(self.config.systems_lock_json) as f:
                self.systems_meta = json.load(f)

        # Load corpora metadata if provided
        if self.config.corpora_lock_json and self.config.corpora_lock_json.exists():
            with open(self.config.corpora_lock_json) as f:
                self.corpora_meta = json.load(f)

    def generate(self) -> None:
        """Generate both Markdown and HTML reports."""
        self.load_data()

        # Generate Markdown
        md_content = self._generate_markdown()
        self.config.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        self.config.markdown_output.write_text(md_content)

        # Generate HTML
        html_content = self._generate_html()
        self.config.html_output.parent.mkdir(parents=True, exist_ok=True)
        self.config.html_output.write_text(html_content)

    def _generate_markdown(self) -> str:
        """Generate the Markdown report."""
        sections = [
            self._md_header(),
            self._md_methodology(),
            self._md_capabilities(),
            self._md_track_p_tables(),
            self._md_track_q_tables(),
            self._md_dnf_log(),
            self._md_caveats(),
        ]
        return "\n\n".join(sections)

    def _md_header(self) -> str:
        """Generate report header."""
        return """# ICEBench Report

**Harness:** ICEBench-v1
**Date:** {date}
**Machine:** {machine}

This report presents performance and accuracy measurements for coding-agentic memory layers.
No established benchmark exists for this domain (design stance: coding-agentic memory is a
novel capability requiring novel evaluation).

""".format(
            date=self.rows[0].ts.split("T")[0] if self.rows else "unknown",
            machine=self.rows[0].machine if self.rows else "unknown",
        )

    def _md_methodology(self) -> str:
        """Generate methodology section."""
        if not self.rows:
            return "## Methodology\n\n*No data available.*"

        sample_row = self.rows[0]
        return f"""## Methodology

**Harness:** ICEBench-v1
**Machine:** {sample_row.machine}
**Repo SHA:** {sample_row.repo_sha}
**Seed:** {sample_row.seed}
**Quiescence:** {self.config.quiescence_statement}

Each measurement is the median of 3 repetitions with min/max reported.
"""

    def _md_capabilities(self) -> str:
        """Generate capability matrix."""
        if not self.capabilities:
            return "## Capability Matrix\n\n*Not provided.*"

        lines = ["## Capability Matrix", ""]
        lines.append("System capabilities by operation class:")
        lines.append("")

        # Extract all unique operations
        all_ops = set()
        for ops in self.capabilities.values():
            all_ops.update(ops.keys())
        all_ops = sorted(all_ops)

        # Header
        systems = sorted(self.capabilities.keys())
        lines.append("| Operation | " + " | ".join(systems) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(systems)) + " |")

        # Rows
        for op in all_ops:
            row = [op]
            for system in systems:
                status = self.capabilities.get(system, {}).get(op, "N/A")
                row.append(status)
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _md_track_p_tables(self) -> str:
        """Generate Track-P performance tables."""
        lines = ["## Track P: Performance", ""]

        # Group by system and corpus
        grouped = defaultdict(lambda: defaultdict(list))
        for row in self.rows:
            if row.kind in ["index", "query", "snapshot", "store"]:
                grouped[row.system][f"{row.corpus}_{row.op}"].append(row)

        for system in sorted(grouped.keys()):
            lines.append(f"### {system}")
            lines.append("")

            corpora = defaultdict(lambda: defaultdict(list))
            for key, rows_list in grouped[system].items():
                if "_" in key:
                    corpus, op = key.split("_", 1)
                    corpora[corpus][op] = rows_list

            for corpus in sorted(corpora.keys()):
                lines.append(f"#### Corpus: {corpus}")
                lines.append("")
                lines.append(
                    "| Operation | Wall (s) | Peak RSS (MB) | CPU (s) | Bytes | Notes |"
                )
                lines.append("| --- | --- | --- | --- | --- | --- |")

                for op in sorted(corpora[corpus].keys()):
                    rows_list = corpora[corpus][op]
                    if not rows_list:
                        continue

                    # Check for DNF
                    dnf_rows = [r for r in rows_list if r.dnf]
                    if dnf_rows:
                        reason = dnf_rows[0].dnf_reason or "unknown"
                        lines.append(
                            f"| {op} | DNF | DNF | DNF | DNF | {reason} |"
                        )
                        continue

                    # Compute stats
                    ok_rows = [r for r in rows_list if r.ok and not r.dnf]
                    if not ok_rows:
                        lines.append(f"| {op} | N/A | N/A | N/A | N/A | No valid runs |")
                        continue

                    wall_vals = [r.wall_s for r in ok_rows if r.wall_s is not None]
                    rss_vals = [
                        r.peak_rss_mb for r in ok_rows if r.peak_rss_mb is not None
                    ]
                    cpu_vals = [r.cpu_s for r in ok_rows if r.cpu_s is not None]
                    bytes_vals = [r.bytes for r in ok_rows if r.bytes is not None]

                    def fmt_stat(vals: list[float]) -> str:
                        if not vals:
                            return "N/A"
                        stats = Stats.from_values(vals)
                        return (
                            f"{stats.median:.2f} ({stats.min_val:.2f}-{stats.max_val:.2f})"
                        )

                    def fmt_bytes(vals: list[int]) -> str:
                        if not vals:
                            return "N/A"
                        med = statistics.median(vals)
                        return f"{int(med)}"

                    lines.append(
                        f"| {op} | {fmt_stat(wall_vals)} | {fmt_stat(rss_vals)} | "
                        f"{fmt_stat(cpu_vals)} | {fmt_bytes(bytes_vals) if bytes_vals else 'N/A'} | - |"
                    )

                lines.append("")

        return "\n".join(lines)

    def _md_track_q_tables(self) -> str:
        """Generate Track-Q accuracy tables."""
        if not self.score_report:
            return "## Track Q: Accuracy\n\n*Not available (H3 not merged).*"

        lines = ["## Track Q: Accuracy", ""]

        for system in sorted(self.score_report.keys()):
            lines.append(f"### {system}")
            lines.append("")

            for corpus in sorted(self.score_report[system].keys()):
                lines.append(f"#### Corpus: {corpus}")
                lines.append("")
                lines.append(
                    "| Operation | Hits@1 | Hits@5 | Hits@10 | MRR | Hit Rate | Precision | Recall |"
                )
                lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

                for op in sorted(self.score_report[system][corpus].keys()):
                    metrics = self.score_report[system][corpus][op]

                    # Check for DNF
                    if metrics.get("dnf"):
                        reason = metrics.get("dnf_reason", "unknown")
                        lines.append(
                            f"| {op} | DNF | DNF | DNF | DNF | DNF | DNF | DNF | {reason} |"
                        )
                        continue

                    def fmt(key: str) -> str:
                        val = metrics.get(key)
                        if val is None:
                            return "N/A"
                        return f"{val:.3f}"

                    lines.append(
                        f"| {op} | {fmt('hits_at_1')} | {fmt('hits_at_5')} | {fmt('hits_at_10')} | "
                        f"{fmt('mrr')} | {fmt('hit_rate')} | {fmt('precision')} | {fmt('recall')} |"
                    )

                lines.append("")

        return "\n".join(lines)

    def _md_dnf_log(self) -> str:
        """Generate DNF/crash log."""
        dnf_rows = [r for r in self.rows if r.dnf]
        if not dnf_rows:
            return "## DNF Log\n\n*No DNF events recorded.*"

        lines = ["## DNF Log", ""]
        lines.append(
            "DNF (Did Not Finish) events are first-class results indicating stability issues:"
        )
        lines.append("")
        lines.append("| System | Corpus | Operation | Rep | Reason |")
        lines.append("| --- | --- | --- | --- | --- |")

        for row in dnf_rows:
            lines.append(
                f"| {row.system} | {row.corpus} | {row.op} | {row.rep} | {row.dnf_reason or 'unknown'} |"
            )

        return "\n".join(lines)

    def _md_caveats(self) -> str:
        """Generate caveats section."""
        lines = ["## Caveats & Methodology Notes", ""]

        # Sample sizes
        lines.append(f"**Sample sizes:** Each measurement is the median of 3 repetitions.")
        lines.append("")

        # Shared oracle bias
        if self.config.oracle_agreement_pct is not None:
            lines.append(
                f"**Shared oracle bias:** All systems share the same ground-truth oracle "
                f"(tree-sitter structural QA). Oracle agreement with LSP spot-checks: "
                f"{self.config.oracle_agreement_pct:.1f}%."
            )
        else:
            lines.append(
                "**Shared oracle bias:** All systems share the same ground-truth oracle "
                "(tree-sitter structural QA)."
            )
        lines.append("")

        # NS-authored harness bias
        lines.append(
            "**NS-authored harness bias:** This harness is authored by the Neuralscape team. "
            "Mitigations: adapters are forbidden from encoding system-specific intelligence; "
            "per-operation N/A honesty (no fabricated numbers for unsupported operations)."
        )
        lines.append("")

        # Novel domain
        lines.append(
            "**Novel domain:** No established benchmark exists for coding-agentic memory layers. "
            "This is an initial design stance for evaluating a novel capability."
        )
        lines.append("")

        return "\n".join(lines)

    def _generate_html(self) -> str:
        """Generate self-contained HTML dashboard."""
        # Load Chart.js
        chart_js = ""
        if self.config.chart_js_path and self.config.chart_js_path.exists():
            chart_js = self.config.chart_js_path.read_text()

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ICEBench Report</title>
  <style>
{self._html_styles()}
  </style>
</head>
<body>
  <header>
    <h1>ICEBench Report</h1>
    <p>Performance and accuracy measurements for coding-agentic memory layers</p>
  </header>
  <main>
{self._html_content()}
  </main>
  <script>
{chart_js}
  </script>
  <script>
{self._html_charts_script()}
  </script>
</body>
</html>
"""
        return html

    def _html_styles(self) -> str:
        """Generate HTML styles (design language from static/index.html)."""
        return """    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0d1117; color: #c9d1d9; }
    header { padding: 16px 24px; border-bottom: 1px solid #21262d; }
    header h1 { margin: 0; font-size: 18px; }
    header p { margin: 4px 0 0; color: #8b949e; font-size: 12px; }
    main { padding: 24px; max-width: 1400px; margin: 0 auto; }
    h2 { font-size: 16px; margin: 32px 0 16px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
    h3 { font-size: 14px; margin: 24px 0 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 16px 0; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #21262d; }
    th { background: #161b22; font-weight: 600; }
    .card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; margin: 16px 0; }
    .dnf { color: #f85149; }
    .na { color: #8b949e; font-style: italic; }
    .chart-container { margin: 24px 0; }
    canvas { max-width: 100%; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .note { font-size: 11px; color: #8b949e; margin: 8px 0; }"""

    def _html_content(self) -> str:
        """Generate HTML content sections."""
        sections = [
            self._html_methodology(),
            self._html_capabilities(),
            self._html_track_p_tables(),
            self._html_track_q_tables(),
            self._html_dnf_log(),
            self._html_caveats(),
        ]
        return "\n".join(sections)

    def _html_methodology(self) -> str:
        """Generate methodology card."""
        if not self.rows:
            return '<div class="card"><h2>Methodology</h2><p class="na">No data available.</p></div>'

        sample_row = self.rows[0]
        return f"""    <div class="card">
      <h2>Methodology</h2>
      <p><strong>Harness:</strong> ICEBench-v1</p>
      <p><strong>Machine:</strong> {sample_row.machine}</p>
      <p><strong>Repo SHA:</strong> {sample_row.repo_sha}</p>
      <p><strong>Seed:</strong> {sample_row.seed}</p>
      <p><strong>Quiescence:</strong> {self.config.quiescence_statement}</p>
      <p class="note">Each measurement is the median of 3 repetitions with min/max reported.</p>
    </div>"""

    def _html_capabilities(self) -> str:
        """Generate capability matrix table."""
        if not self.capabilities:
            return '<div class="card"><h2>Capability Matrix</h2><p class="na">Not provided.</p></div>'

        # Extract all unique operations
        all_ops = set()
        for ops in self.capabilities.values():
            all_ops.update(ops.keys())
        all_ops = sorted(all_ops)

        systems = sorted(self.capabilities.keys())
        header_cells = "".join(f"<th>{s}</th>" for s in systems)

        rows = []
        for op in all_ops:
            cells = [f"<td>{op}</td>"]
            for system in systems:
                status = self.capabilities.get(system, {}).get(op, "N/A")
                if status == "N/A" or "not supported" in status.lower():
                    cells.append(f'<td class="na">{status}</td>')
                elif status.lower() == "supported":
                    cells.append(f"<td>{status}</td>")
                else:
                    cells.append(f'<td class="na">{status}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")

        return f"""    <div class="card">
      <h2>Capability Matrix</h2>
      <table>
        <thead>
          <tr><th>Operation</th>{header_cells}</tr>
        </thead>
        <tbody>
{"".join(rows)}
        </tbody>
      </table>
    </div>"""

    def _html_track_p_tables(self) -> str:
        """Generate Track-P performance tables."""
        lines = ['    <div class="card">', '      <h2>Track P: Performance</h2>']

        # Group by system and corpus
        grouped = defaultdict(lambda: defaultdict(list))
        for row in self.rows:
            if row.kind in ["index", "query", "snapshot", "store"]:
                grouped[row.system][f"{row.corpus}_{row.op}"].append(row)

        for system in sorted(grouped.keys()):
            lines.append(f"      <h3>{system}</h3>")

            corpora = defaultdict(lambda: defaultdict(list))
            for key, rows_list in grouped[system].items():
                if "_" in key:
                    corpus, op = key.split("_", 1)
                    corpora[corpus][op] = rows_list

            for corpus in sorted(corpora.keys()):
                lines.append(f"      <h4>Corpus: {corpus}</h4>")
                lines.append("      <table>")
                lines.append("        <thead>")
                lines.append(
                    "          <tr><th>Operation</th><th>Wall (s)</th><th>Peak RSS (MB)</th>"
                    "<th>CPU (s)</th><th>Bytes</th><th>Notes</th></tr>"
                )
                lines.append("        </thead>")
                lines.append("        <tbody>")

                for op in sorted(corpora[corpus].keys()):
                    rows_list = corpora[corpus][op]
                    if not rows_list:
                        continue

                    # Check for DNF
                    dnf_rows = [r for r in rows_list if r.dnf]
                    if dnf_rows:
                        reason = dnf_rows[0].dnf_reason or "unknown"
                        lines.append(
                            f'          <tr><td>{op}</td><td class="dnf">DNF</td>'
                            f'<td class="dnf">DNF</td><td class="dnf">DNF</td>'
                            f'<td class="dnf">DNF</td><td>{reason}</td></tr>'
                        )
                        continue

                    # Compute stats
                    ok_rows = [r for r in rows_list if r.ok and not r.dnf]
                    if not ok_rows:
                        lines.append(
                            f'          <tr><td>{op}</td><td class="na">N/A</td>'
                            f'<td class="na">N/A</td><td class="na">N/A</td>'
                            f'<td class="na">N/A</td><td>No valid runs</td></tr>'
                        )
                        continue

                    wall_vals = [r.wall_s for r in ok_rows if r.wall_s is not None]
                    rss_vals = [
                        r.peak_rss_mb for r in ok_rows if r.peak_rss_mb is not None
                    ]
                    cpu_vals = [r.cpu_s for r in ok_rows if r.cpu_s is not None]
                    bytes_vals = [r.bytes for r in ok_rows if r.bytes is not None]

                    def fmt_stat(vals: list[float]) -> str:
                        if not vals:
                            return '<span class="na">N/A</span>'
                        stats = Stats.from_values(vals)
                        return (
                            f"{stats.median:.2f} ({stats.min_val:.2f}-{stats.max_val:.2f})"
                        )

                    def fmt_bytes(vals: list[int]) -> str:
                        if not vals:
                            return '<span class="na">N/A</span>'
                        med = statistics.median(vals)
                        return f"{int(med)}"

                    lines.append(
                        f"          <tr><td>{op}</td><td>{fmt_stat(wall_vals)}</td>"
                        f"<td>{fmt_stat(rss_vals)}</td><td>{fmt_stat(cpu_vals)}</td>"
                        f"<td>{fmt_bytes(bytes_vals) if bytes_vals else '<span class=\"na\">N/A</span>'}</td>"
                        f"<td>-</td></tr>"
                    )

                lines.append("        </tbody>")
                lines.append("      </table>")

        lines.append("    </div>")
        return "\n".join(lines)

    def _html_track_q_tables(self) -> str:
        """Generate Track-Q accuracy tables."""
        if not self.score_report:
            return '    <div class="card"><h2>Track Q: Accuracy</h2><p class="na">Not available (H3 not merged).</p></div>'

        lines = ['    <div class="card">', '      <h2>Track Q: Accuracy</h2>']

        for system in sorted(self.score_report.keys()):
            lines.append(f"      <h3>{system}</h3>")

            for corpus in sorted(self.score_report[system].keys()):
                lines.append(f"      <h4>Corpus: {corpus}</h4>")
                lines.append("      <table>")
                lines.append("        <thead>")
                lines.append(
                    "          <tr><th>Operation</th><th>Hits@1</th><th>Hits@5</th>"
                    "<th>Hits@10</th><th>MRR</th><th>Hit Rate</th><th>Precision</th><th>Recall</th></tr>"
                )
                lines.append("        </thead>")
                lines.append("        <tbody>")

                for op in sorted(self.score_report[system][corpus].keys()):
                    metrics = self.score_report[system][corpus][op]

                    # Check for DNF
                    if metrics.get("dnf"):
                        reason = metrics.get("dnf_reason", "unknown")
                        lines.append(
                            f'          <tr><td>{op}</td><td class="dnf">DNF</td>'
                            f'<td class="dnf">DNF</td><td class="dnf">DNF</td>'
                            f'<td class="dnf">DNF</td><td class="dnf">DNF</td>'
                            f'<td class="dnf">DNF</td><td class="dnf">DNF</td></tr>'
                        )
                        continue

                    def fmt(key: str) -> str:
                        val = metrics.get(key)
                        if val is None:
                            return '<span class="na">N/A</span>'
                        return f"{val:.3f}"

                    lines.append(
                        f"          <tr><td>{op}</td><td>{fmt('hits_at_1')}</td>"
                        f"<td>{fmt('hits_at_5')}</td><td>{fmt('hits_at_10')}</td>"
                        f"<td>{fmt('mrr')}</td><td>{fmt('hit_rate')}</td>"
                        f"<td>{fmt('precision')}</td><td>{fmt('recall')}</td></tr>"
                    )

                lines.append("        </tbody>")
                lines.append("      </table>")

        lines.append("    </div>")
        return "\n".join(lines)

    def _html_dnf_log(self) -> str:
        """Generate DNF/crash log."""
        dnf_rows = [r for r in self.rows if r.dnf]
        if not dnf_rows:
            return '    <div class="card"><h2>DNF Log</h2><p class="na">No DNF events recorded.</p></div>'

        lines = ['    <div class="card">', '      <h2>DNF Log</h2>']
        lines.append(
            '      <p>DNF (Did Not Finish) events are first-class results indicating stability issues:</p>'
        )
        lines.append("      <table>")
        lines.append("        <thead>")
        lines.append(
            "          <tr><th>System</th><th>Corpus</th><th>Operation</th><th>Rep</th><th>Reason</th></tr>"
        )
        lines.append("        </thead>")
        lines.append("        <tbody>")

        for row in dnf_rows:
            lines.append(
                f"          <tr><td>{row.system}</td><td>{row.corpus}</td>"
                f"<td>{row.op}</td><td>{row.rep}</td>"
                f'<td class="dnf">{row.dnf_reason or "unknown"}</td></tr>'
            )

        lines.append("        </tbody>")
        lines.append("      </table>")
        lines.append("    </div>")
        return "\n".join(lines)

    def _html_caveats(self) -> str:
        """Generate caveats section."""
        oracle_note = ""
        if self.config.oracle_agreement_pct is not None:
            oracle_note = f" Oracle agreement with LSP spot-checks: {self.config.oracle_agreement_pct:.1f}%."

        return f"""    <div class="card">
      <h2>Caveats &amp; Methodology Notes</h2>
      <p><strong>Sample sizes:</strong> Each measurement is the median of 3 repetitions.</p>
      <p><strong>Shared oracle bias:</strong> All systems share the same ground-truth oracle (tree-sitter structural QA).{oracle_note}</p>
      <p><strong>NS-authored harness bias:</strong> This harness is authored by the Neuralscape team.
         Mitigations: adapters are forbidden from encoding system-specific intelligence;
         per-operation N/A honesty (no fabricated numbers for unsupported operations).</p>
      <p><strong>Novel domain:</strong> No established benchmark exists for coding-agentic memory layers.
         This is an initial design stance for evaluating a novel capability.</p>
    </div>"""

    def _html_charts_script(self) -> str:
        """Generate JavaScript for charts (honest axes, start at 0)."""
        # For now, return empty script. Charts can be added later.
        # This would use the inlined Chart.js to create visualizations.
        return """// Charts would be rendered here using Chart.js
// Example: latency bars, index-time bars, etc.
// All axes start at 0 (honest axes, no truncation)
console.log('ICEBench report loaded');"""


def generate_report(config: ReportConfig) -> None:
    """
    Generate ICEBench report (Markdown + HTML).

    Args:
        config: Report configuration.
    """
    generator = ReportGenerator(config)
    generator.generate()
