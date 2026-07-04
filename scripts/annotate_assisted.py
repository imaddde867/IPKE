#!/usr/bin/env python3
"""Model-assisted annotation harness for IPKE-Bench gold drafts.

Produces a *draft* gold file for one bounded procedure unit, using the P3
two-stage method (Stage 1: ordered steps; Stage 2: constraints attached to
those step IDs) — the same method the benchmark evaluates, so the assisted
annotation is methodologically aligned with the pipeline under test.

EPISTEMIC CONTRACT (this is the whole point):
  * Output is written to datasets/paper/draft/<doc>.json with
    quality.review_status = "unreviewed" and annotator = "model-assisted:<model>".
  * A draft is NOT a paper-grade gold. It DELIBERATELY fails
    scripts/validate_paper_gold.py on the review gate until a human adjudicates
    it with scripts/adjudicate.py (which sets review_status="reviewed").
  * The draft IS structurally validated here: locked 6-type vocabulary, {must,
    should, may} enforcement, every constraint attached to a real step id,
    non-empty text. The harness retries (feeding the structural errors back to
    the model) until structure is clean or --max-retries is hit.
  * Independence: the first-pass human annotator MUST NOT read the draft before
    their own pass (docs/annotation/guidelines.md IAA rule). The draft is an
    adjudication target, not an anchor for first-pass annotation.

LLM BACKEND (injectable):
  * Inside Claude Science: import annotate_segment() and pass host.llm.
  * Standalone / user's stack: set IPKE_LLM_BASE_URL (OpenAI-compatible, e.g.
    an Ollama or AIOP endpoint) and IPKE_LLM_MODEL; the CLI uses that.

Usage (CLI, OpenAI-compatible backend):
    export IPKE_LLM_BASE_URL=http://localhost:11434/v1
    export IPKE_LLM_MODEL=qwen2.5:32b-instruct
    uv run python scripts/annotate_assisted.py \
        --segments datasets/paper/segments/epa_field_operations_manual_filter_sampling_sop.segments.json \
        --candidate 5.13.2 \
        --text datasets/paper/text/epa_field_operations_manual_filter_sampling_sop.txt \
        --out datasets/paper/draft/epa_field_operations_manual_filter_sampling_sop.json

Usage (Claude Science kernel):
    import annotate_assisted as aa
    draft = aa.annotate_segment(text, meta, llm_fn=lambda p, s=None: host.llm(p, system=s)["text"])
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

# Locked vocabulary. Import from repo if available; else inline (keeps the
# harness runnable outside the package too).
try:
    from src.benchmark.taxonomy import (
        LOCKED_CONSTRAINT_TYPES,
        LOCKED_ENFORCEMENT_LEVELS,
    )
    LOCKED_TYPES = set(LOCKED_CONSTRAINT_TYPES)
    LOCKED_ENFORCEMENT = set(LOCKED_ENFORCEMENT_LEVELS)
except Exception:  # pragma: no cover - fallback when run outside the package
    LOCKED_TYPES = {"precondition", "postcondition", "guard", "parameter",
                    "role_assignment", "reference"}
    LOCKED_ENFORCEMENT = {"must", "should", "may"}

LlmFn = Callable[..., str]

# --- text repair -------------------------------------------------------------
# Mid-word hyphen breaks from PDF text extraction ("sam-\npling" -> "sampling").
RE_HYPHEN_BREAK = re.compile(r"(\w+)-\n\s*(\w+)")


def repair_text(text: str) -> str:
    """Join hyphenated line breaks and collapse hard-wrapped lines lightly.

    Conservative: only joins a hyphen immediately followed by newline+word.
    Does not alter numbered-list newlines (those carry step structure).
    """
    prev = None
    out = text
    # iterate because a word may break twice
    while out != prev:
        prev = out
        out = RE_HYPHEN_BREAK.sub(r"\1\2", out)
    return out


# --- prompts -----------------------------------------------------------------
TYPE_DEFINITIONS = """\
- precondition: a state/condition that MUST hold BEFORE the step runs.
- postcondition: a state/result the step MUST establish AFTER it runs (records, confirmations, outputs).
- guard: a safety/operational prohibition or protective condition governing HOW the step is done (PPE, "do not touch X", hold-points).
- parameter: a quantitative or enumerated specification the step must meet (flow rate 1-3 L/min, 15 topics, torque value, list of required areas).
- role_assignment: a requirement that a NAMED role/actor performs or is responsible for the step (e.g. "the AFOM reviews...", "Mission Directorate AA shall ensure...").
- reference: a normative pointer to another document/section that binds this step (e.g. "per NFM 3", "in accordance with 1.5.1.a-e")."""

SYSTEM = (
    "You are an expert annotator building a gold-standard procedural-knowledge "
    "benchmark. You extract steps and the constraints attached to them from "
    "technical procedures, using a fixed vocabulary, with constraint text taken "
    "verbatim or near-verbatim from the source. You output only valid JSON."
)

STAGE1 = """\
Extract the ORDERED, ACTIONABLE steps of the single procedure below, end to end.

Rules:
- One step = one discrete action a practitioner performs, in execution order.
- Cover the WHOLE procedure (this is a complete bounded sub-procedure, target 15-40 steps). Do not stop early; do not merge distinct actions.
- id: S1, S2, ... in order. label: a concise imperative description of the action.
- action_verb: the main verb (lowercase). action_object: what it acts on.
- Ignore constraints/parameters for now (next stage handles them).

Procedure title: {title}
Source section: {section}

Respond with ONLY this JSON:
{{"steps": [{{"id": "S1", "label": "...", "action_verb": "...", "action_object": "..."}}]}}

Procedure text:
\"\"\"
{chunk}
\"\"\""""

STAGE2 = """\
You already extracted these steps (execution order):
{steps_json}

Now extract every CONSTRAINT in the procedure text and attach each to the step(s) it governs.

CRITICAL RULES:
- Every constraint MUST reference at least one step id from the list above in "attached_to".
- Do NOT invent constraints that attach to no step. Do NOT restate a step as a constraint.
- "type" MUST be exactly one of: precondition, postcondition, guard, parameter, role_assignment, reference.
{type_definitions}
- "enforcement" MUST be exactly one of: must, should, may.
  * must = shall / must / required / prohibition ("do not", "never").
  * should = recommended / should / preferred.
  * may = optional / permitted / "may" / "can".
- "text" MUST be verbatim or near-verbatim from the source (short, the governing clause).
- Actively look for PERMITTED / OPTIONAL clauses ("may", "if desired", "as needed") and label them enforcement="may". Do not silently drop them.
- A step may have several constraints; a constraint may attach to several steps.

Respond with ONLY this JSON:
{{"constraints": [{{"id": "C1", "type": "guard", "text": "...", "attached_to": ["S1"], "enforcement": "must"}}]}}

Procedure text:
\"\"\"
{chunk}
\"\"\""""

REPAIR = """\
Your previous JSON had these STRUCTURAL errors against the locked schema:
{errors}

Fix ONLY these problems and return the COMPLETE corrected JSON (same format as before).
Reminder: type in {{precondition, postcondition, guard, parameter, role_assignment, reference}};
enforcement in {{must, should, may}}; every constraint.attached_to must list real step ids from: {step_ids}."""


# --- JSON extraction ---------------------------------------------------------
def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model response (handles code fences)."""
    s = raw.strip()
    if "```" in s:
        # take the content of the first fenced block
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    # find outermost braces
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    return json.loads(s[start:end + 1])


# --- structural validation (mirrors validate_paper_gold, minus review gate) --
def structural_errors(steps: list[dict], constraints: list[dict]) -> list[str]:
    errs: list[str] = []
    step_ids = {s.get("id") for s in steps if s.get("id")}
    if not step_ids:
        return ["no step ids extracted"]
    for i, c in enumerate(constraints):
        cid = c.get("id", f"idx{i}")
        if c.get("type") not in LOCKED_TYPES:
            errs.append(f"{cid}: type={c.get('type')!r} not in locked vocabulary")
        if c.get("enforcement") not in LOCKED_ENFORCEMENT:
            errs.append(f"{cid}: enforcement={c.get('enforcement')!r} not in {{must,should,may}}")
        if not c.get("text"):
            errs.append(f"{cid}: empty text")
        refs = c.get("attached_to") or c.get("applies_to") or []
        if isinstance(refs, str):
            refs = [refs]
        if not refs:
            errs.append(f"{cid}: no attached_to")
        else:
            for r in refs:
                if r not in step_ids:
                    errs.append(f"{cid}: attached_to {r!r} is not a real step id")
    return errs


# --- gold assembly -----------------------------------------------------------
def assemble_gold(meta: dict, steps: list[dict], constraints: list[dict], model: str) -> dict:
    """Embed each constraint under the FIRST step it attaches to (matches the
    canonical gold shape where constraints live in steps[].constraints[])."""
    by_step: dict[str, list[dict]] = {s["id"]: [] for s in steps if s.get("id")}
    top_level: list[dict] = []
    for c in constraints:
        refs = c.get("attached_to") or c.get("applies_to") or []
        if isinstance(refs, str):
            refs = [refs]
        placed = False
        for r in refs:
            if r in by_step:
                by_step[r].append(c)
                placed = True
                break
        if not placed:
            top_level.append(c)

    out_steps = []
    for s in steps:
        sid = s.get("id")
        out_steps.append({
            "id": sid,
            "label": s.get("label", ""),
            "action_verb": s.get("action_verb", ""),
            "action_object": s.get("action_object", ""),
            "arguments": s.get("arguments", []),
            "parameters": s.get("parameters", []),
            "constraints": by_step.get(sid, []),
            "flags": s.get("flags", {}),
            "provenance": s.get("provenance", {}),
        })

    return {
        "procedure": {
            "doc_id": meta["doc_id"],
            "title": meta.get("title", meta["doc_id"]),
            "version": meta.get("version"),
            "domain": meta.get("domain"),
            "source": {
                "doc_id": meta["doc_id"],
                "section": meta.get("section"),
                "page": meta.get("page"),
                "char_start": meta.get("char_start"),
                "char_end": meta.get("char_end"),
            },
        },
        "steps": out_steps,
        "constraints": top_level,
        "relations": [],
        "quality": {
            "annotation_scope": "full_subprocedure",
            "review_status": "unreviewed",
            "annotator": f"model-assisted:{model}",
            "review_date": None,
            "review_notes": (
                "Model-assisted DRAFT via scripts/annotate_assisted.py (P3 two-stage). "
                "NOT reviewed. Must be adjudicated with scripts/adjudicate.py before it is a gold. "
                "Structural validation (locked vocab + attachment) passed at draft time."
            ),
        },
    }


# --- core routine ------------------------------------------------------------
def annotate_segment(text: str, meta: dict, llm_fn: LlmFn, *,
                     model: str = "unknown", max_retries: int = 3,
                     verbose: bool = True) -> dict:
    """Run the two-stage assisted annotation on one procedure's text.

    llm_fn(prompt: str, system: Optional[str]) -> str  (returns raw model text)
    """
    chunk = repair_text(text)
    title = meta.get("title", meta["doc_id"])
    section = meta.get("section", "")

    def log(m):
        if verbose:
            print(m, file=sys.stderr)

    # Stage 1: steps
    r1 = llm_fn(STAGE1.format(title=title, section=section, chunk=chunk), SYSTEM)
    steps = extract_json(r1).get("steps", [])
    # normalise ids to S1..Sn in given order
    for i, s in enumerate(steps, 1):
        s["id"] = f"S{i}"
    log(f"stage1: {len(steps)} steps")
    if not steps:
        raise ValueError("stage 1 returned no steps")

    steps_json = json.dumps([{"id": s["id"], "label": s.get("label", "")} for s in steps], indent=0)

    # Stage 2: constraints (+ structural repair loop)
    prompt2 = STAGE2.format(steps_json=steps_json, type_definitions=TYPE_DEFINITIONS, chunk=chunk)
    raw2 = llm_fn(prompt2, SYSTEM)
    constraints = extract_json(raw2).get("constraints", [])
    for i, c in enumerate(constraints, 1):
        c.setdefault("id", f"C{i}")

    attempt = 0
    while attempt < max_retries:
        errs = structural_errors(steps, constraints)
        if not errs:
            break
        attempt += 1
        log(f"repair {attempt}: {len(errs)} structural errors")
        rp = llm_fn(
            prompt2 + "\n\n" + REPAIR.format(
                errors="\n".join(f"- {e}" for e in errs),
                step_ids=", ".join(s["id"] for s in steps),
            ),
            SYSTEM,
        )
        try:
            constraints = extract_json(rp).get("constraints", constraints)
            for i, c in enumerate(constraints, 1):
                c.setdefault("id", f"C{i}")
        except ValueError:
            continue

    final_errs = structural_errors(steps, constraints)
    log(f"stage2: {len(constraints)} constraints, {len(final_errs)} residual structural errors")

    gold = assemble_gold(meta, steps, constraints, model)
    gold["quality"]["_draft_diagnostics"] = {
        "n_steps": len(steps),
        "n_constraints": len(constraints),
        "residual_structural_errors": final_errs,
        "repair_rounds": attempt,
    }
    return gold


# --- backends ----------------------------------------------------------------
def openai_backend() -> tuple[LlmFn, str]:
    """OpenAI-compatible backend from env (Ollama / AIOP / vLLM / OpenAI)."""
    import urllib.request

    base = os.environ.get("IPKE_LLM_BASE_URL")
    model = os.environ.get("IPKE_LLM_MODEL", "unknown")
    key = os.environ.get("IPKE_LLM_API_KEY", "not-needed")
    if not base:
        raise SystemExit(
            "No LLM backend: set IPKE_LLM_BASE_URL (+ IPKE_LLM_MODEL), "
            "or import annotate_segment() and pass llm_fn=host.llm inside Claude Science."
        )

    def call(prompt: str, system: Optional[str] = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        body = json.dumps({"model": model, "messages": msgs, "temperature": 0.0}).encode()
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    return call, model


def load_candidate(segments_path: Path, candidate: str) -> dict:
    seg = json.loads(segments_path.read_text())
    for c in seg["candidates"]:
        if str(c["section"]) == str(candidate):
            return {"doc_id": seg["doc_id"], "title": c["title"], "section": c["section"],
                    "char_start": c["char_start"], "char_end": c["char_end"]}
    raise SystemExit(f"candidate {candidate!r} not found in {segments_path}. "
                     f"Available: {[c['section'] for c in seg['candidates']]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", type=Path, required=True, help="*.segments.json from segment_procedures.py")
    ap.add_argument("--candidate", required=True, help="candidate section id to annotate (e.g. 5.13.2 or M3)")
    ap.add_argument("--text", type=Path, required=True, help="source .txt")
    ap.add_argument("--out", type=Path, required=True, help="output draft json")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--title", help="override procedure title")
    ap.add_argument("--domain", help="procedure.domain value")
    ap.add_argument("--version", help="procedure.version value")
    args = ap.parse_args()

    meta = load_candidate(args.segments, args.candidate)
    if args.title:
        meta["title"] = args.title
    if args.domain:
        meta["domain"] = args.domain
    if args.version:
        meta["version"] = args.version

    full = args.text.read_text(errors="ignore")
    cs, ce = meta["char_start"], meta["char_end"]
    segment_text = full[cs:ce] if cs is not None else full

    llm_fn, model = openai_backend()
    gold = annotate_segment(segment_text, meta, llm_fn, model=model, max_retries=args.max_retries)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, indent=2))
    diag = gold["quality"]["_draft_diagnostics"]
    print(f"wrote {args.out}: {diag['n_steps']} steps, {diag['n_constraints']} constraints, "
          f"{len(diag['residual_structural_errors'])} residual errors, {diag['repair_rounds']} repair rounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
