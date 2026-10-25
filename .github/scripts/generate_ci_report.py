import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def flatten(mapping: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten(value, name))
        else:
            rows.append((name, value))
    return rows


def cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        value = f"{value:.5f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def run_url() -> str | None:
    server = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if not all((server, repository, run_id)):
        return None
    return f"{server}/{repository}/actions/runs/{run_id}"


def build_report(
    metrics: dict[str, Any] | None,
    params: dict[str, Any],
    results: list[dict[str, Any]] | None,
) -> str:
    lines = ["# Legal RAG CI Evaluation Report", ""]
    metadata = {
        "Commit": os.getenv("GITHUB_SHA", "local"),
        "Ref": os.getenv("GITHUB_REF_NAME", "local"),
        "Event": os.getenv("GITHUB_EVENT_NAME", "local"),
        "Workflow run": run_url() or "local",
        "Type checking": os.getenv("TYPECHECK_RESULT", "local"),
        "Tests": os.getenv("TEST_RESULT", "local"),
        "Evaluation": os.getenv("EVALUATION_RESULT", "local"),
        "Faithfulness gate": os.getenv("GATE_RESULT", "local"),
    }
    lines.extend(["## Run", "", "| Field | Value |", "| --- | --- |"])
    lines.extend(f"| {cell(key)} | {cell(value)} |" for key, value in metadata.items())

    lines.extend(["", "## Evaluation gate", ""])
    if metrics and "ragas_faithfulness" in metrics:
        score = float(metrics["ragas_faithfulness"])
        status = "✅ Passed" if score >= 0.75 else "❌ Failed"
        lines.append(f"{status}: RAGAS faithfulness `{score:.5f}` (required: `>= 0.75`).")
    else:
        lines.append("⚠️ Evaluation metrics were not produced.")

    lines.extend(["", "## Configuration", "", "| Parameter | Value |", "| --- | --- |"])
    lines.extend(f"| `{cell(key)}` | `{cell(value)}` |" for key, value in flatten(params))

    lines.extend(["", "## Metrics", "", "| Metric | Value |", "| --- | --- |"])
    if metrics:
        lines.extend(f"| `{cell(key)}` | {cell(value)} |" for key, value in sorted(metrics.items()))
    else:
        lines.append("| — | Metrics unavailable |")

    lines.extend(
        [
            "",
            "## Golden-question results",
            "",
            "| ID | Question | Expected article | Answered | Cited articles | Faithfulness |",
            "| --- | --- | ---: | --- | --- | ---: |",
        ]
    )
    if results:
        for result in results:
            question = str(result.get("question", ""))
            if len(question) > 100:
                question = f"{question[:97]}..."
            citations = ", ".join(str(value) for value in result.get("cited_article_numbers", [])) or "—"
            lines.append(
                "| {id} | {question} | {expected} | {answered} | {citations} | {faithfulness} |".format(
                    id=cell(result.get("id")),
                    question=cell(question),
                    expected=cell(result.get("expected_article_number")),
                    answered="Yes" if result.get("answered") else "No",
                    citations=cell(citations),
                    faithfulness=cell(result.get("ragas_faithfulness")),
                )
            )
    else:
        lines.append("| — | Detailed results unavailable | — | — | — | — |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the GitHub Actions evaluation report.")
    parser.add_argument("--metrics", type=Path, default=Path("data/output/ragas_metrics.json"))
    parser.add_argument("--results", type=Path, default=Path("data/output/ragas_results.json"))
    parser.add_argument("--params", type=Path, default=Path("dvc/params.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/ci-evaluation.md"))
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    results = load_json(args.results)
    params = load_yaml(args.params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(metrics, params, results), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
