#!/usr/bin/env python3
"""
Validate that pipeline.py produced a coherent, replayable set of artifacts.

Checks performed:
  - All required artifacts exist on disk.
  - Every JSON file is parseable.
  - Clause extraction happened before any LLM call (by timestamp comparison).
  - Every extracted clause has a Stage 1 (risk-scoring) score.
  - Risk categories and severities match the framework only.
  - Each pre-override critical clause has its own Stage 2 LLM call record.
  - Operator overrides were saved AND applied to final_severities.
  - The negotiation brief reflects post-override severities.
  - llm_calls.jsonl has at least one stage_1 and one stage_3 record.

Exit code: 0 if everything passes, 1 otherwise.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent

REQUIRED_ARTIFACTS = [
    "contract.txt",
    "risk_framework.json",
    "extracted_clauses.json",
    "risk_analysis.json",
    "operator_overrides.json",
    "negotiation_brief.md",
    "redlined_contract.md",
    "clause_cross_references.json",
    "signature_risk_score.json",
    "llm_calls.jsonl",
]

JSON_ARTIFACTS = [a for a in REQUIRED_ARTIFACTS if a.endswith(".json")]


def _load_json(path: Path):
    """Read and parse a JSON file. Returns parsed object or raises."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path):
    """Read and parse a JSON-Lines file, returning a list of records."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} line {i}: {e}") from e
    return out


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp; tolerate a trailing 'Z'."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def main() -> int:
    """Run all validation checks. Returns 0 on success, 1 on any failure."""
    errors: list[str] = []
    warnings: list[str] = []

    # ----- 1. Artifact existence -----
    for name in REQUIRED_ARTIFACTS:
        if not (ROOT / name).exists():
            errors.append(f"missing artifact: {name}")
    if errors:
        return _report(errors, warnings)

    # ----- 2. JSON validity -----
    parsed: dict = {}
    for name in JSON_ARTIFACTS:
        try:
            parsed[name] = _load_json(ROOT / name)
        except json.JSONDecodeError as e:
            errors.append(f"invalid JSON in {name}: {e}")
    if errors:
        return _report(errors, warnings)

    try:
        llm_calls = _load_jsonl(ROOT / "llm_calls.jsonl")
    except ValueError as e:
        errors.append(str(e))
        return _report(errors, warnings)

    framework = parsed["risk_framework.json"]["risk_framework"]
    valid_categories = set(framework["categories"])
    valid_severities = set(framework["severity_levels"].keys())

    extracted = parsed["extracted_clauses.json"]["clauses"]
    extracted_numbers = {c["clause_number"] for c in extracted}

    risk_analysis = parsed["risk_analysis.json"]
    stage_1_scores = risk_analysis.get("stage_1_risk_scoring", [])
    stage_2_deep = risk_analysis.get("stage_2_deep_analysis", [])
    final_severities = risk_analysis.get("final_severities", [])

    overrides_doc = parsed["operator_overrides.json"]
    overrides = {int(k): v for k, v in overrides_doc.get("overrides", {}).items()}

    # ----- 3. Clause extraction happened before any LLM call -----
    if llm_calls:
        first_llm_ts = min(_parse_iso(r["timestamp"]) for r in llm_calls)
        # If extracted_clauses.json is newer than the earliest LLM call, the
        # pipeline replayed extraction after calling the LLM — illegal order.
        clauses_mtime = datetime.fromtimestamp(
            (ROOT / "extracted_clauses.json").stat().st_mtime,
            tz=first_llm_ts.tzinfo,
        )
        if clauses_mtime > first_llm_ts:
            warnings.append(
                "extracted_clauses.json mtime is newer than the earliest LLM "
                "call — replay order may be inverted."
            )
    else:
        errors.append("llm_calls.jsonl is empty — no LLM calls were logged")

    # ----- 4. Every extracted clause has a Stage 1 risk score -----
    scored_numbers = {r["clause_number"] for r in stage_1_scores}
    missing = extracted_numbers - scored_numbers
    if missing:
        errors.append(f"clauses without Stage 1 risk score: {sorted(missing)}")
    extra = scored_numbers - extracted_numbers
    if extra:
        errors.append(f"Stage 1 scored clauses not in extraction: {sorted(extra)}")

    # ----- 5. Risk categories and severities match the framework -----
    for r in stage_1_scores:
        if r.get("risk_category") not in valid_categories:
            errors.append(
                f"clause {r.get('clause_number')}: invalid risk_category "
                f"{r.get('risk_category')!r}"
            )
        if r.get("severity") not in valid_severities:
            errors.append(
                f"clause {r.get('clause_number')}: invalid severity "
                f"{r.get('severity')!r}"
            )

    # ----- 6. Each pre-override critical clause has its own Stage 2 LLM record -----
    pre_override_critical = [
        r["clause_number"] for r in stage_1_scores if r.get("severity") == "critical"
    ]
    stage_2_records = [r for r in llm_calls if r.get("stage") == "stage_2_deep_analysis"]
    stage_2_clause_nums = {r.get("clause_number") for r in stage_2_records}
    for cn in pre_override_critical:
        if cn not in stage_2_clause_nums:
            errors.append(
                f"critical clause {cn} has no stage_2_deep_analysis LLM call record"
            )
    # And matching deep-analysis output entries.
    deep_clause_nums = {d.get("clause_number") for d in stage_2_deep}
    for cn in pre_override_critical:
        if cn not in deep_clause_nums:
            errors.append(
                f"critical clause {cn} missing from stage_2_deep_analysis output"
            )

    # ----- 7. Operator overrides saved AND applied -----
    if not isinstance(final_severities, list) or not final_severities:
        errors.append("risk_analysis.json: final_severities is empty")
    else:
        # Every override key must appear in final_severities with that value.
        fs_by_num = {fs["clause_number"]: fs for fs in final_severities}
        for cn, sev in overrides.items():
            fs = fs_by_num.get(cn)
            if fs is None:
                errors.append(f"override for clause {cn} not present in final_severities")
            elif fs.get("final_severity") != sev:
                errors.append(
                    f"override for clause {cn}: expected final_severity={sev!r}, "
                    f"got {fs.get('final_severity')!r}"
                )
            elif not fs.get("was_overridden"):
                errors.append(
                    f"override for clause {cn}: was_overridden flag not set"
                )
        # Every clause must appear in final_severities with a valid severity.
        for cn in extracted_numbers:
            fs = fs_by_num.get(cn)
            if fs is None:
                errors.append(f"clause {cn} missing from final_severities")
            elif fs.get("final_severity") not in valid_severities:
                errors.append(
                    f"clause {cn}: invalid final_severity "
                    f"{fs.get('final_severity')!r}"
                )

    # ----- 8. Negotiation brief reflects post-override severities -----
    brief = (ROOT / "negotiation_brief.md").read_text(encoding="utf-8")
    brief_sections = {
        "critical": "Red Lines",
        "high": "Priority Negotiations",
        "medium": "Acceptable With Modification",
        "low": "Standard / Accept",
    }
    for required in brief_sections.values():
        if required not in brief:
            errors.append(f"negotiation_brief.md missing section header: '{required}'")
    if "Opening Position" not in brief:
        errors.append("negotiation_brief.md missing 'Opening Position' section")
    if "NOT LEGAL ADVICE" not in brief.upper():
        errors.append("negotiation_brief.md missing legal-advice disclaimer")

    # Spot-check: each final-severity clause should appear in its severity
    # section of the brief, not in the section of its original severity (if
    # the two differ because of an override).
    fs_by_num = {fs["clause_number"]: fs for fs in final_severities}
    for cn, sev in overrides.items():
        fs = fs_by_num.get(cn) or {}
        original = fs.get("original_severity")
        if original and original != sev:
            target_section = brief_sections[sev]
            other_section = brief_sections[original]
            # Find the slice of the brief that belongs to the target section
            # vs. the original-severity section, then check which contains
            # this clause number. Heuristic but reliable for our format.
            target_slice = _section_slice(brief, target_section)
            other_slice = _section_slice(brief, other_section)
            num_pat = re.compile(rf"\b{cn}\b")
            in_target = bool(num_pat.search(target_slice))
            in_other = bool(num_pat.search(other_slice))
            if in_other and not in_target:
                errors.append(
                    f"override clause {cn}: appears under '{other_section}' "
                    f"in brief but final severity is {sev!r} ({target_section})"
                )

    # ----- 9. llm_calls.jsonl has required stage records -----
    seen_stages = {r.get("stage") for r in llm_calls}
    for required_stage in ("stage_1_risk_scoring", "stage_3_negotiation_brief"):
        if required_stage not in seen_stages:
            errors.append(f"llm_calls.jsonl missing required stage record: {required_stage}")
    # If there were any pre-override critical clauses, stage_2 must appear.
    if pre_override_critical and "stage_2_deep_analysis" not in seen_stages:
        errors.append("llm_calls.jsonl missing stage_2_deep_analysis records")

    # Each LLM record must have all required fields.
    required_fields = (
        "stage", "clause_number", "timestamp", "provider", "model",
        "prompt_hash", "input_artifacts", "output_artifact",
    )
    for i, rec in enumerate(llm_calls):
        for field in required_fields:
            if field not in rec:
                errors.append(f"llm_calls.jsonl record {i}: missing field '{field}'")

    return _report(errors, warnings)


def _section_slice(brief: str, header_text: str) -> str:
    """Return the portion of the markdown brief belonging to the given section.

    Looks for a markdown heading line containing header_text, then returns the
    text up to (but not including) the next heading.
    """
    # Find a line starting with one or more '#' that contains header_text.
    pattern = re.compile(rf"^#+\s+.*{re.escape(header_text)}.*$", re.MULTILINE)
    m = pattern.search(brief)
    if not m:
        return ""
    next_heading = re.compile(r"^#+\s+", re.MULTILINE)
    n = next_heading.search(brief, m.end())
    return brief[m.end() : n.start()] if n else brief[m.end() :]


def _report(errors: list[str], warnings: list[str]) -> int:
    """Print a human-readable summary of validation results and return exit code."""
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  - {e}")
        print(f"\n{len(errors)} validation error(s).")
        return 1
    print("OK: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
