#!/usr/bin/env python3
"""
Contract risk-analysis pipeline.

State machine (advances on success of each stage):
    INIT
    -> INPUTS_LOADED
    -> CLAUSES_EXTRACTED
    -> CLAUSES_RISK_SCORED
    -> CRITICAL_CLAUSES_ANALYSED
    -> OPERATOR_REVIEW_COMPLETE
    -> NEGOTIATION_BRIEF_GENERATED
    -> VALIDATION_COMPLETE
    -> RESULTS_FINALISED

Outputs are AI-GENERATED ANALYSIS, NOT LEGAL ADVICE.
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
# .env may contain whitespace around the key; strip defensively.
_GROQ_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
client = Groq(api_key=_GROQ_KEY)
MODEL = "llama-3.3-70b-versatile"

ROOT = Path(__file__).parent

# Inputs
CONTRACT_PATH = ROOT / "contract.txt"
FRAMEWORK_PATH = ROOT / "risk_framework.json"

# Artifacts
EXTRACTED_CLAUSES_PATH = ROOT / "extracted_clauses.json"
RISK_ANALYSIS_PATH = ROOT / "risk_analysis.json"
OPERATOR_OVERRIDES_PATH = ROOT / "operator_overrides.json"
NEGOTIATION_BRIEF_PATH = ROOT / "negotiation_brief.md"
REDLINED_CONTRACT_PATH = ROOT / "redlined_contract.md"
CROSS_REFERENCES_PATH = ROOT / "clause_cross_references.json"
SIGNATURE_SCORE_PATH = ROOT / "signature_risk_score.json"
LLM_LOG_PATH = ROOT / "llm_calls.jsonl"

# Disclaimers
DISCLAIMER_MD = (
    "> **AI-GENERATED ANALYSIS — NOT LEGAL ADVICE.** "
    "Produced by an automated pipeline. Review by qualified counsel before any "
    "legal or commercial decision."
)
DISCLAIMER_JSON = (
    "AI-GENERATED ANALYSIS. NOT LEGAL ADVICE. Review by qualified counsel "
    "before any legal or commercial decision."
)

VALID_SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_POINTS = {"critical": 25, "high": 12, "medium": 5, "low": 1}

# State machine — advanced via _set_state().
_STATE_ORDER = [
    "INIT",
    "INPUTS_LOADED",
    "CLAUSES_EXTRACTED",
    "CLAUSES_RISK_SCORED",
    "CRITICAL_CLAUSES_ANALYSED",
    "OPERATOR_REVIEW_COMPLETE",
    "NEGOTIATION_BRIEF_GENERATED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]
_state = {"current": "INIT"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _set_state(new_state: str) -> None:
    """Advance the pipeline state machine and print the transition.

    Raises RuntimeError if the transition is not the next legal step.
    """
    if new_state not in _STATE_ORDER:
        raise RuntimeError(f"Unknown state: {new_state}")
    expected_idx = _STATE_ORDER.index(_state["current"]) + 1
    if _STATE_ORDER[expected_idx] != new_state:
        raise RuntimeError(
            f"Illegal state transition {_state['current']} -> {new_state}"
        )
    _state["current"] = new_state
    print(f"[STATE] -> {new_state}")


def _banner(stage_num: int, name: str) -> None:
    """Print a clear banner showing which stage is starting."""
    print(f"\n========== STAGE {stage_num}: {name} ==========")


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    """Return SHA-256 hex digest of a string (used for prompt_hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing ```...``` markdown code fence if present."""
    t = text.strip()
    if t.startswith("```"):
        # drop first line (``` or ```json)
        nl = t.find("\n")
        t = t[nl + 1 :] if nl >= 0 else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _parse_json(text: str) -> Any:
    """Parse a JSON string, tolerating markdown code fences from the model."""
    return json.loads(_strip_code_fence(text))


# ---------------------------------------------------------------------------
# LLM wrapper + logging
# ---------------------------------------------------------------------------


def _log_llm_call(
    stage: str,
    clause_number: Any,
    prompt: str,
    input_artifacts: list,
    output_artifact: str,
) -> None:
    """Append one JSON-line record describing an LLM call to llm_calls.jsonl."""
    record = {
        "stage": stage,
        "clause_number": clause_number,
        "timestamp": _now_iso(),
        "provider": "groq",
        "model": MODEL,
        "prompt_hash": _sha256(prompt),
        "input_artifacts": list(input_artifacts),
        "output_artifact": output_artifact,
    }
    with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _call_llm(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> str:
    """Send a single chat completion request to Groq and return the response text.

    Uses temperature=0 by default for reproducibility. Set json_mode=True to
    request a JSON object response.
    """
    kwargs: dict = {
        "model": MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# STAGE 1 — Load inputs
# ---------------------------------------------------------------------------


def stage_1_load_inputs() -> tuple[str, dict]:
    """Read contract.txt and risk_framework.json from disk.

    Returns (contract_text, framework_dict). Raises FileNotFoundError if either
    input is missing — pipeline cannot proceed without both.
    """
    _banner(1, "Load inputs")
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {CONTRACT_PATH.name}")
    if not FRAMEWORK_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {FRAMEWORK_PATH.name}")

    contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
    framework = json.loads(FRAMEWORK_PATH.read_text(encoding="utf-8"))

    # Sanity-check the framework shape so we fail fast.
    rf = framework.get("risk_framework") or {}
    if not isinstance(rf.get("categories"), list) or not rf["categories"]:
        raise ValueError("risk_framework.json: missing or empty 'categories'")
    if not isinstance(rf.get("severity_levels"), dict) or not rf["severity_levels"]:
        raise ValueError("risk_framework.json: missing or empty 'severity_levels'")

    print(f"  contract.txt: {len(contract_text)} chars")
    print(f"  risk_framework.json: {len(rf['categories'])} categories, "
          f"{len(rf['severity_levels'])} severity levels")

    _set_state("INPUTS_LOADED")
    return contract_text, framework


# ---------------------------------------------------------------------------
# STAGE 2 — Deterministic clause extraction (NO LLM)
# ---------------------------------------------------------------------------

# Lines like "1. SERVICES" or "10. INDEMNIFICATION" mark the start of a clause.
_CLAUSE_HEADER_RE = re.compile(r"^(\d+)\.\s+([A-Z][A-Z0-9 ,/&'\-]+)\s*$", re.MULTILINE)


def stage_2_extract_clauses(contract_text: str) -> list[dict]:
    """Parse the contract by numbered, all-caps section headers.

    Deterministic — no LLM involved. Each clause record contains:
        clause_number, clause_title, clause_text, word_count
    """
    _banner(2, "Extract clauses (deterministic)")
    matches = list(_CLAUSE_HEADER_RE.finditer(contract_text))
    if not matches:
        raise ValueError(
            "No numbered section headers found in contract.txt — "
            "expected lines like '1. SERVICES'."
        )

    clauses: list[dict] = []
    for i, m in enumerate(matches):
        number = int(m.group(1))
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(contract_text)
        body = contract_text[body_start:body_end].strip()
        clauses.append(
            {
                "clause_number": number,
                "clause_title": title,
                "clause_text": body,
                "word_count": len(body.split()),
            }
        )

    payload = {
        "disclaimer": DISCLAIMER_JSON,
        "extraction_method": "deterministic_regex",
        "clauses": clauses,
    }
    EXTRACTED_CLAUSES_PATH.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"  Extracted {len(clauses)} clauses -> {EXTRACTED_CLAUSES_PATH.name}")

    _set_state("CLAUSES_EXTRACTED")
    return clauses


# ---------------------------------------------------------------------------
# STAGE 3 — Risk scoring (ONE LLM call for all clauses)
# ---------------------------------------------------------------------------


def stage_3_score_risks(clauses: list[dict], framework: dict) -> list[dict]:
    """Score every clause against the risk framework using ONE LLM call.

    Output per clause: clause_number, risk_category, severity,
    one_sentence_risk_summary, is_non_standard. Categories and severities are
    constrained to the framework — any out-of-vocabulary value is rejected.
    """
    _banner(3, "Risk scoring (1 LLM call)")
    rf = framework["risk_framework"]
    categories = rf["categories"]
    severity_levels = rf["severity_levels"]

    system = (
        "You are a contract-risk analyst. Output ONLY a single valid JSON object. "
        "Use only the categories and severities provided. Do not invent values."
    )
    user = (
        "Analyse each clause below and assign one risk category and one severity.\n\n"
        f"ALLOWED CATEGORIES (use only these exact strings):\n{json.dumps(categories)}\n\n"
        "ALLOWED SEVERITIES (use only these exact keys; definitions provided):\n"
        f"{json.dumps(severity_levels, indent=2)}\n\n"
        "CLAUSES:\n"
        f"{json.dumps(clauses, indent=2)}\n\n"
        "Return JSON with this exact shape:\n"
        '{"clauses": [\n'
        '  {"clause_number": <int>, "risk_category": "<one of allowed categories>",\n'
        '   "severity": "<one of allowed severities>",\n'
        '   "one_sentence_risk_summary": "<string>",\n'
        '   "is_non_standard": <true|false>}\n'
        "]}"
    )
    prompt_for_hash = system + "\n---\n" + user

    try:
        raw = _call_llm(system, user, json_mode=True)
        parsed = _parse_json(raw)
    except Exception as e:
        raise RuntimeError(f"Stage 3 LLM call failed: {e}") from e

    risks = parsed.get("clauses") if isinstance(parsed, dict) else None
    if not isinstance(risks, list):
        raise RuntimeError("Stage 3: LLM did not return a 'clauses' list")

    # Validate each entry against the framework.
    valid_categories = set(categories)
    valid_severities = set(severity_levels.keys())
    extracted_numbers = {c["clause_number"] for c in clauses}
    seen_numbers: set[int] = set()

    cleaned: list[dict] = []
    for r in risks:
        n = r.get("clause_number")
        if n not in extracted_numbers:
            raise RuntimeError(f"Stage 3: unknown clause_number {n!r}")
        if r.get("risk_category") not in valid_categories:
            raise RuntimeError(
                f"Stage 3: clause {n} has invalid category "
                f"{r.get('risk_category')!r}"
            )
        if r.get("severity") not in valid_severities:
            raise RuntimeError(
                f"Stage 3: clause {n} has invalid severity {r.get('severity')!r}"
            )
        cleaned.append(
            {
                "clause_number": n,
                "risk_category": r["risk_category"],
                "severity": r["severity"],
                "one_sentence_risk_summary": str(
                    r.get("one_sentence_risk_summary", "")
                ),
                "is_non_standard": bool(r.get("is_non_standard", False)),
            }
        )
        seen_numbers.add(n)

    missing = extracted_numbers - seen_numbers
    if missing:
        raise RuntimeError(
            f"Stage 3: LLM did not score clauses {sorted(missing)}"
        )

    # Persist Stage 1 (risk-scoring) results into risk_analysis.json.
    risk_analysis = {
        "disclaimer": DISCLAIMER_JSON,
        "stage_1_risk_scoring": cleaned,
        "stage_2_deep_analysis": [],
        "final_severities": [],
    }
    RISK_ANALYSIS_PATH.write_text(
        json.dumps(risk_analysis, indent=2), encoding="utf-8"
    )

    _log_llm_call(
        stage="stage_1_risk_scoring",
        clause_number=None,
        prompt=prompt_for_hash,
        input_artifacts=[EXTRACTED_CLAUSES_PATH.name, FRAMEWORK_PATH.name],
        output_artifact=RISK_ANALYSIS_PATH.name,
    )

    print(f"  Scored {len(cleaned)} clauses -> {RISK_ANALYSIS_PATH.name}")
    for r in cleaned:
        print(f"    Clause {r['clause_number']}: {r['severity']:<8} | {r['risk_category']}")

    _set_state("CLAUSES_RISK_SCORED")
    return cleaned


# ---------------------------------------------------------------------------
# STAGE 4 — Deep analysis (ONE LLM call PER critical clause, never batched)
# ---------------------------------------------------------------------------


def stage_4_deep_analysis(
    clauses: list[dict], risks: list[dict], framework: dict
) -> list[dict]:
    """Run deep analysis on every clause that Stage 3 marked critical.

    Each critical clause gets its OWN LLM call (no batching). Output per clause:
    clause_number, harm_mechanism, precedent_framing, redline_suggestions (3),
    market_standard_comparison, basis. The results are appended to
    risk_analysis.json without disturbing Stage 1 data.
    """
    _banner(4, "Deep analysis (1 LLM call per critical clause)")
    severity_levels = framework["risk_framework"]["severity_levels"]
    by_number = {c["clause_number"]: c for c in clauses}

    critical = [r for r in risks if r["severity"] == "critical"]
    if not critical:
        print("  No critical clauses — skipping deep analysis.")
        _set_state("CRITICAL_CLAUSES_ANALYSED")
        return []

    deep_results: list[dict] = []
    for r in critical:
        n = r["clause_number"]
        clause = by_number[n]
        system = (
            "You are a contract-risk analyst. Output ONLY a single valid JSON "
            "object describing the deep analysis of one clause."
        )
        user = (
            f"Clause number: {n}\n"
            f"Clause title: {clause['clause_title']}\n"
            f"Risk category: {r['risk_category']}\n"
            f"Severity: critical\n"
            f"Severity definition: {severity_levels['critical']}\n"
            f"One-sentence risk summary: {r['one_sentence_risk_summary']}\n\n"
            f"Clause text:\n\"\"\"\n{clause['clause_text']}\n\"\"\"\n\n"
            "Return JSON with this exact shape:\n"
            "{\n"
            f'  "clause_number": {n},\n'
            '  "harm_mechanism": "<how this clause causes harm in practice>",\n'
            '  "precedent_framing": "<how to frame this in negotiation, citing market norms>",\n'
            '  "redline_suggestions": ["<edit 1>", "<edit 2>", "<edit 3>"],\n'
            '  "market_standard_comparison": "<what market-standard wording looks like>",\n'
            '  "basis": "<the reasoning / evidence for the above>"\n'
            "}"
        )
        prompt_for_hash = system + "\n---\n" + user

        try:
            raw = _call_llm(system, user, json_mode=True)
            parsed = _parse_json(raw)
        except Exception as e:
            raise RuntimeError(f"Stage 4 LLM call failed for clause {n}: {e}") from e

        if not isinstance(parsed, dict) or parsed.get("clause_number") != n:
            raise RuntimeError(
                f"Stage 4: response for clause {n} missing or mismatched clause_number"
            )
        red = parsed.get("redline_suggestions")
        if not isinstance(red, list) or len(red) < 1:
            raise RuntimeError(
                f"Stage 4: clause {n} response missing redline_suggestions list"
            )
        # Pad/trim to exactly three suggestions for downstream consistency.
        red = [str(x) for x in red][:3]
        while len(red) < 3:
            red.append("")
        parsed["redline_suggestions"] = red

        deep_results.append(parsed)

        _log_llm_call(
            stage="stage_2_deep_analysis",
            clause_number=n,
            prompt=prompt_for_hash,
            input_artifacts=[
                EXTRACTED_CLAUSES_PATH.name,
                RISK_ANALYSIS_PATH.name,
                FRAMEWORK_PATH.name,
            ],
            output_artifact=RISK_ANALYSIS_PATH.name,
        )
        print(f"  Deep-analysed clause {n} ({clause['clause_title']})")

    # Append (preserving Stage 1).
    current = json.loads(RISK_ANALYSIS_PATH.read_text(encoding="utf-8"))
    current["stage_2_deep_analysis"] = deep_results
    RISK_ANALYSIS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")

    _set_state("CRITICAL_CLAUSES_ANALYSED")
    return deep_results


# ---------------------------------------------------------------------------
# STAGE 5 — Operator review checkpoint
# ---------------------------------------------------------------------------


def stage_5_operator_review(
    clauses: list[dict], risks: list[dict]
) -> dict[int, str]:
    """Print all clause risk scores and let the operator override severities.

    Repeats input() until the operator presses Enter. Only severities in
    VALID_SEVERITIES are accepted; clause numbers must exist. Overrides are
    saved to operator_overrides.json and applied to risk_analysis.json's
    final_severities. Returns the override map (clause_number -> severity).
    """
    _banner(5, "Operator review")
    by_number = {c["clause_number"]: c for c in clauses}

    print("\n  --- Clause risk scores (pre-override) ---")
    for r in risks:
        c = by_number.get(r["clause_number"], {})
        title = c.get("clause_title", "?")
        print(
            f"    Clause {r['clause_number']:>2}: {r['severity'].upper():<8} "
            f"| {r['risk_category']:<24} | {title}"
        )
        print(f"               -> {r['one_sentence_risk_summary']}")

    print(
        "\nAre there any clauses whose severity you want to override before "
        "generating the negotiation brief? Enter clause number and new severity, "
        "or press Enter to continue."
    )
    print(f"  Format: <clause_number> <severity>   (severities: {', '.join(VALID_SEVERITIES)})")

    valid_numbers = {r["clause_number"] for r in risks}
    overrides: dict[int, str] = {}

    while True:
        try:
            line = input("override> ").strip()
        except EOFError:
            # Non-interactive run — accept zero overrides and move on.
            print("  (no TTY — continuing with zero overrides)")
            break
        if not line:
            break
        parts = line.split()
        if len(parts) != 2:
            print("  Format: <clause_number> <severity>")
            continue
        try:
            num = int(parts[0])
        except ValueError:
            print("  Clause number must be an integer.")
            continue
        sev = parts[1].lower()
        if num not in valid_numbers:
            print(f"  Clause {num} not found.")
            continue
        if sev not in VALID_SEVERITIES:
            print(f"  Severity must be one of: {', '.join(VALID_SEVERITIES)}")
            continue
        overrides[num] = sev
        print(f"  Recorded override: clause {num} -> {sev}")

    # Compute final severities (post-override) for downstream stages.
    final_severities = []
    for r in risks:
        n = r["clause_number"]
        original = r["severity"]
        final = overrides.get(n, original)
        final_severities.append(
            {
                "clause_number": n,
                "original_severity": original,
                "final_severity": final,
                "was_overridden": n in overrides,
            }
        )

    OPERATOR_OVERRIDES_PATH.write_text(
        json.dumps(
            {
                "disclaimer": DISCLAIMER_JSON,
                "applied_at": _now_iso(),
                # str keys for JSON portability
                "overrides": {str(k): v for k, v in overrides.items()},
                "final_severities": final_severities,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Update risk_analysis.json with final severities (preserving prior stages).
    current = json.loads(RISK_ANALYSIS_PATH.read_text(encoding="utf-8"))
    current["final_severities"] = final_severities
    RISK_ANALYSIS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")

    print(f"  {len(overrides)} override(s) saved -> {OPERATOR_OVERRIDES_PATH.name}")
    _set_state("OPERATOR_REVIEW_COMPLETE")
    return overrides


# ---------------------------------------------------------------------------
# STAGE 6 — Negotiation brief (ONE LLM call)
# ---------------------------------------------------------------------------


def stage_6_negotiation_brief(
    clauses: list[dict],
    risks: list[dict],
    deep: list[dict],
    final_severities: list[dict],
) -> str:
    """Generate negotiation_brief.md via ONE LLM call.

    Sections required: Red Lines (critical), Priority Negotiations (high),
    Acceptable With Modification (medium), Standard / Accept (low),
    Opening Position (2-3 sentences). All severities used here are post-override.
    """
    _banner(6, "Negotiation brief (1 LLM call)")
    by_number = {c["clause_number"]: c for c in clauses}
    final_by_number = {fs["clause_number"]: fs for fs in final_severities}
    deep_by_number = {d["clause_number"]: d for d in deep}

    bundle = []
    for r in risks:
        n = r["clause_number"]
        bundle.append(
            {
                "clause_number": n,
                "clause_title": by_number[n]["clause_title"],
                "clause_text": by_number[n]["clause_text"],
                "risk_category": r["risk_category"],
                "original_severity": r["severity"],
                "final_severity": final_by_number[n]["final_severity"],
                "was_overridden": final_by_number[n]["was_overridden"],
                "one_sentence_risk_summary": r["one_sentence_risk_summary"],
                "is_non_standard": r["is_non_standard"],
                "deep_analysis": deep_by_number.get(n),
            }
        )

    system = (
        "You are a senior contract negotiator. Produce a markdown briefing for "
        "the operator. Prioritise clauses by FINAL severity (post-override). "
        "Be concise, specific, and actionable. Do NOT give legal advice."
    )
    user = (
        "Generate a negotiation briefing as GitHub-flavored markdown with EXACTLY "
        "these top-level sections, in this order:\n"
        "  ## Red Lines (critical)\n"
        "  ## Priority Negotiations (high)\n"
        "  ## Acceptable With Modification (medium)\n"
        "  ## Standard / Accept (low)\n"
        "  ## Opening Position\n\n"
        "Rules:\n"
        "- Group every clause under exactly ONE of the four severity sections, "
        "based on its `final_severity`.\n"
        "- Inside each section, list clauses as bullet points: clause number, "
        "title, one-line risk, and the strongest 1-2 talking points or asks. "
        "If `was_overridden` is true, note '(operator override)' on that bullet.\n"
        "- 'Opening Position' is 2-3 sentences framing the overall negotiation stance.\n"
        "- If a section has no clauses, write '_None._'\n\n"
        "INPUT CLAUSES (use FINAL severity for grouping):\n"
        f"{json.dumps(bundle, indent=2)}"
    )
    prompt_for_hash = system + "\n---\n" + user

    try:
        body = _call_llm(system, user, json_mode=False)
    except Exception as e:
        raise RuntimeError(f"Stage 6 LLM call failed: {e}") from e

    body = _strip_code_fence(body).strip()

    final_md = (
        f"# Negotiation Briefing\n\n"
        f"{DISCLAIMER_MD}\n\n"
        f"_Generated: {_now_iso()}_\n\n"
        f"---\n\n"
        f"{body}\n"
    )
    NEGOTIATION_BRIEF_PATH.write_text(final_md, encoding="utf-8")

    _log_llm_call(
        stage="stage_3_negotiation_brief",
        clause_number=None,
        prompt=prompt_for_hash,
        input_artifacts=[
            RISK_ANALYSIS_PATH.name,
            OPERATOR_OVERRIDES_PATH.name,
            EXTRACTED_CLAUSES_PATH.name,
        ],
        output_artifact=NEGOTIATION_BRIEF_PATH.name,
    )
    print(f"  Negotiation brief saved -> {NEGOTIATION_BRIEF_PATH.name}")
    _set_state("NEGOTIATION_BRIEF_GENERATED")
    return final_md


# ---------------------------------------------------------------------------
# STAGE 7 — Redlined contract
# ---------------------------------------------------------------------------


def stage_7_redline(
    contract_text: str,
    clauses: list[dict],
    deep: list[dict],
    final_severities: list[dict],
) -> str:
    """Build redlined_contract.md.

    - For critical clauses, replace the body with the first redline_suggestion
      from Stage 4 (deterministic — no extra LLM call).
    - For high clauses (when deep analysis is absent), do ONE LLM call to
      synthesise redline text using the clause text alone.
    - All replacement text is wrapped in **bold**.
    """
    _banner(7, "Redlined contract")
    deep_by_number = {d["clause_number"]: d for d in deep}
    final_by_number = {fs["clause_number"]: fs["final_severity"] for fs in final_severities}

    out_lines: list[str] = []
    out_lines.append("# Redlined Contract\n")
    out_lines.append(DISCLAIMER_MD + "\n")
    out_lines.append(f"_Generated: {_now_iso()}_\n")
    out_lines.append("\n---\n\n")

    # Walk the original contract and replace clause bodies where appropriate.
    by_number = {c["clause_number"]: c for c in clauses}

    # Header (everything before clause 1).
    first_clause = clauses[0]
    header_end = contract_text.find(first_clause["clause_text"])
    if header_end > 0:
        # Use the regex to find the first header line position instead, more robust.
        first_match = next(_CLAUSE_HEADER_RE.finditer(contract_text))
        out_lines.append(contract_text[: first_match.start()])

    for c in clauses:
        n = c["clause_number"]
        sev = final_by_number.get(n, "low")
        title_line = f"{n}. {c['clause_title']}\n"
        out_lines.append(title_line)

        if sev == "critical" and n in deep_by_number:
            redline = deep_by_number[n]["redline_suggestions"][0] or c["clause_text"]
            out_lines.append(
                f"**[REDLINED — critical] {redline}**\n\n"
            )
        elif sev == "high":
            replacement = _redline_one_clause(c, sev)
            out_lines.append(f"**[REDLINED — high] {replacement}**\n\n")
        else:
            out_lines.append(c["clause_text"] + "\n\n")

    REDLINED_CONTRACT_PATH.write_text("".join(out_lines), encoding="utf-8")
    print(f"  Redlined contract saved -> {REDLINED_CONTRACT_PATH.name}")
    return REDLINED_CONTRACT_PATH.read_text(encoding="utf-8")


def _redline_one_clause(clause: dict, severity: str) -> str:
    """Produce redline text for a single clause via a focused LLM call.

    Only used for high-severity clauses that did not receive Stage 4 deep
    analysis. Returns a single string of replacement clause text.
    """
    system = (
        "You are a contract redline editor. Rewrite the clause to be balanced "
        "and market-standard while preserving its lawful intent. Output ONLY "
        "JSON: {\"redline\": \"<rewritten clause text>\"}."
    )
    user = (
        f"Clause number: {clause['clause_number']}\n"
        f"Clause title: {clause['clause_title']}\n"
        f"Severity: {severity}\n\n"
        f"Original clause text:\n\"\"\"\n{clause['clause_text']}\n\"\"\""
    )
    prompt_for_hash = system + "\n---\n" + user
    try:
        raw = _call_llm(system, user, json_mode=True)
        parsed = _parse_json(raw)
        text = str(parsed.get("redline", "")).strip()
    except Exception:
        # Graceful degradation: if the LLM call fails, fall back to original text
        # with a flag so the operator can see that redlining was incomplete.
        text = (
            f"[REDLINE GENERATION FAILED — please draft manually] "
            f"{clause['clause_text']}"
        )
    _log_llm_call(
        stage="stage_optional_redline",
        clause_number=clause["clause_number"],
        prompt=prompt_for_hash,
        input_artifacts=[EXTRACTED_CLAUSES_PATH.name, RISK_ANALYSIS_PATH.name],
        output_artifact=REDLINED_CONTRACT_PATH.name,
    )
    return text or clause["clause_text"]


# ---------------------------------------------------------------------------
# STAGE 8 — Clause cross references
# ---------------------------------------------------------------------------


def stage_8_cross_references(
    clauses: list[dict], risks: list[dict], final_severities: list[dict]
) -> list[dict]:
    """Identify pairs of clauses that compound risk when read together.

    ONE LLM call. Output: clause_a, clause_b, combined_risk_description,
    combined_severity (in VALID_SEVERITIES). Saved to clause_cross_references.json.
    """
    _banner(8, "Clause cross references (1 LLM call)")
    final_by_number = {fs["clause_number"]: fs["final_severity"] for fs in final_severities}
    bundle = []
    for r in risks:
        n = r["clause_number"]
        bundle.append(
            {
                "clause_number": n,
                "risk_category": r["risk_category"],
                "final_severity": final_by_number.get(n, r["severity"]),
                "one_sentence_risk_summary": r["one_sentence_risk_summary"],
            }
        )

    system = (
        "You are a contract-risk analyst. Identify pairs of clauses whose "
        "combined effect creates risk greater than the sum of their parts. "
        "Output ONLY a JSON object."
    )
    user = (
        "Identify clause pairs that COMPOUND risk (e.g. broad data licence + "
        "weak termination, or low liability cap + indemnity). Be selective — "
        "only include genuine compounding interactions, not arbitrary pairs.\n\n"
        f"CLAUSES:\n{json.dumps(bundle, indent=2)}\n\n"
        "Return JSON shaped:\n"
        '{"cross_references": [\n'
        '  {"clause_a": <int>, "clause_b": <int>,\n'
        '   "combined_risk_description": "<string>",\n'
        f'   "combined_severity": "<one of {list(VALID_SEVERITIES)}>"}}\n'
        "]}\n"
        "If there are no genuine interactions, return an empty list."
    )
    prompt_for_hash = system + "\n---\n" + user

    try:
        raw = _call_llm(system, user, json_mode=True)
        parsed = _parse_json(raw)
        refs = parsed.get("cross_references", []) if isinstance(parsed, dict) else []
    except Exception as e:
        # Graceful degradation: empty list rather than failing the whole pipeline.
        print(f"  WARNING: cross-reference call failed ({e}); writing empty list.")
        refs = []

    valid_numbers = {r["clause_number"] for r in risks}
    cleaned: list[dict] = []
    for ref in refs:
        a, b = ref.get("clause_a"), ref.get("clause_b")
        sev = ref.get("combined_severity")
        if a not in valid_numbers or b not in valid_numbers or a == b:
            continue
        if sev not in VALID_SEVERITIES:
            continue
        cleaned.append(
            {
                "clause_a": int(a),
                "clause_b": int(b),
                "combined_risk_description": str(ref.get("combined_risk_description", "")),
                "combined_severity": sev,
            }
        )

    CROSS_REFERENCES_PATH.write_text(
        json.dumps(
            {
                "disclaimer": DISCLAIMER_JSON,
                "cross_references": cleaned,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log_llm_call(
        stage="stage_optional_cross_references",
        clause_number=None,
        prompt=prompt_for_hash,
        input_artifacts=[RISK_ANALYSIS_PATH.name],
        output_artifact=CROSS_REFERENCES_PATH.name,
    )
    print(f"  {len(cleaned)} cross-reference(s) -> {CROSS_REFERENCES_PATH.name}")
    return cleaned


# ---------------------------------------------------------------------------
# STAGE 9 — Signature risk score (DETERMINISTIC, no LLM)
# ---------------------------------------------------------------------------


def stage_9_signature_score(final_severities: list[dict]) -> dict:
    """Compute a deterministic 0-100 risk score from final severities.

    Formula: critical=25, high=12, medium=5, low=1; final = min(100, sum).
    """
    _banner(9, "Signature risk score (deterministic)")
    distribution = {s: 0 for s in VALID_SEVERITIES}
    for fs in final_severities:
        sev = fs["final_severity"]
        distribution[sev] = distribution.get(sev, 0) + 1

    raw_total = sum(distribution[s] * SEVERITY_POINTS[s] for s in VALID_SEVERITIES)
    final_score = min(100, raw_total)

    payload = {
        "disclaimer": DISCLAIMER_JSON,
        "formula": {
            "points_per_severity": SEVERITY_POINTS,
            "rule": "score = min(100, sum(count[severity] * points[severity]))",
        },
        "severity_distribution": distribution,
        "raw_total_before_cap": raw_total,
        "final_score": final_score,
        "justification": (
            f"Sum of severities * weights = {raw_total}; "
            f"capped at 100 -> {final_score}. "
            f"Counts: " + ", ".join(f"{k}={distribution[k]}" for k in VALID_SEVERITIES)
        ),
    }
    SIGNATURE_SCORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Final signature risk score: {final_score}/100 -> {SIGNATURE_SCORE_PATH.name}")
    return payload


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full pipeline. Returns 0 on success, 1 on failure."""
    print(f"[STATE] {_state['current']}")
    print(f"[NOTE] All outputs are AI-GENERATED ANALYSIS, NOT LEGAL ADVICE.")

    if not _GROQ_KEY:
        print(
            "ERROR: GROQ_API_KEY is not set. Add it to .env (no leading spaces) "
            "and re-run.",
            file=sys.stderr,
        )
        return 1

    try:
        contract_text, framework = stage_1_load_inputs()
        clauses = stage_2_extract_clauses(contract_text)
        risks = stage_3_score_risks(clauses, framework)
        deep = stage_4_deep_analysis(clauses, risks, framework)
        stage_5_operator_review(clauses, risks)

        # Reload final severities (Stage 5 wrote them to disk).
        risk_analysis = json.loads(RISK_ANALYSIS_PATH.read_text(encoding="utf-8"))
        final_severities = risk_analysis["final_severities"]

        stage_6_negotiation_brief(clauses, risks, deep, final_severities)
        stage_7_redline(contract_text, clauses, deep, final_severities)
        _set_state("VALIDATION_COMPLETE")
        stage_8_cross_references(clauses, risks, final_severities)
        stage_9_signature_score(final_severities)
        _set_state("RESULTS_FINALISED")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nERROR in state {_state['current']}: {e}", file=sys.stderr)
        return 1

    print("\nPipeline complete. Artifacts:")
    for p in (
        EXTRACTED_CLAUSES_PATH,
        RISK_ANALYSIS_PATH,
        OPERATOR_OVERRIDES_PATH,
        NEGOTIATION_BRIEF_PATH,
        REDLINED_CONTRACT_PATH,
        CROSS_REFERENCES_PATH,
        SIGNATURE_SCORE_PATH,
        LLM_LOG_PATH,
    ):
        print(f"  - {p.name}")
    print("\nReminder: AI-generated analysis. NOT legal advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
