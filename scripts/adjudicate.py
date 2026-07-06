#!/usr/bin/env python3
"""
adjudicate.py -- turn a model-assisted DRAFT into a reviewed gold.

This is the human-in-the-loop gate of the IPKE-Bench annotation pipeline. A
draft produced by scripts/annotate_assisted.py carries
quality.review_status="unreviewed" and deliberately fails the paper validator's
review gate. Adjudication is the ONLY way a draft becomes a gold: a human makes
an explicit accept / edit / reject decision on every step and every constraint,
the decisions are logged verbatim into quality.review_notes for audit, and only
then are review_status="reviewed", annotator and review_date stamped.

Two modes:

  single   Review the model draft element by element (the common case). Each
           step and constraint is shown with its source; the adjudicator
           accepts it, edits any field, or rejects (drops) it, and may add
           elements the model missed.

  diff     Reconcile a model draft against an INDEPENDENT human first-pass
           annotation of the same procedure (used for the >=30% double-annotated
           subset -- see scripts/compute_iaa.py and the independent-annotator
           workflow). Elements are aligned; agreements, model-only and
           human-only elements are surfaced, and the adjudicator resolves each.
           The human first-pass is authored WITHOUT reading the draft, so this
           is genuine reconciliation, not anchored editing.

Reproducibility: the decision engine (apply_decisions) is pure and takes a
decisions list, so an adjudication can be replayed non-interactively from a
saved decisions file (--decisions) and unit-tested. Interactive prompting only
builds that decisions list.

Usage:
  # interactive single-draft review -> writes datasets/paper/gold/<doc>.json
  python scripts/adjudicate.py review datasets/paper/draft/<doc>.json \
      --annotator imad --out-dir datasets/paper/gold

  # interactive diff reconciliation of draft vs independent human pass
  python scripts/adjudicate.py diff datasets/paper/draft/<doc>.json \
      datasets/paper/human_pass/<doc>.json --annotator imad

  # non-interactive replay of a saved decision log (reproducible)
  python scripts/adjudicate.py replay datasets/paper/draft/<doc>.json \
      --decisions decisions/<doc>.json --annotator imad
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

# Reuse the exact locked vocabulary + structural validation from the harness so
# adjudicator and drafter can never drift apart.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from annotate_assisted import (  # type: ignore
        LOCKED_TYPES,
        LOCKED_ENFORCEMENT,
        structural_errors,
    )
except Exception:  # pragma: no cover - fallback if import path differs
    LOCKED_TYPES = {
        "precondition", "postcondition", "guard",
        "parameter", "role_assignment", "reference",
    }
    LOCKED_ENFORCEMENT = {"must", "should", "may"}

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
                errs.append(f"{cid}: enforcement={c.get('enforcement')!r} not allowed")
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
                        errs.append(f"{cid}: attached_to {r!r} not a real step id")
        return errs


# --- constraint flattening / rebuilding -------------------------------------
def flatten_constraints(gold: dict) -> list[dict]:
    """Return every constraint (step-embedded + top-level) as a flat list, each
    tagged with the step id(s) it attaches to. Order: by step, then top-level."""
    out: list[dict] = []
    for s in gold.get("steps", []):
        for c in s.get("constraints", []) or []:
            c = dict(c)
            c.setdefault("attached_to", [s["id"]])
            out.append(c)
    for c in gold.get("constraints", []) or []:
        out.append(dict(c))
    return out


def _norm_refs(c: dict) -> list[str]:
    refs = c.get("attached_to") or c.get("applies_to") or []
    return [refs] if isinstance(refs, str) else list(refs)


# --- pure decision engine ----------------------------------------------------
# A decision is a dict:
#   {"kind": "step"|"constraint", "id": <element id>,
#    "action": "accept"|"edit"|"reject"|"add",
#    "edits": {<field>: <value>, ...}   # for edit/add
#    "note": <free text>}               # optional, appended to log
def apply_decisions(draft: dict, decisions: list[dict], *,
                    annotator: str, review_date: str,
                    scope: str | None = None) -> tuple[dict, list[str]]:
    """Apply an ordered decision list to a draft and return (reviewed_gold, log).

    Pure and deterministic: same draft + decisions -> same gold. Rejected
    elements are dropped; edited elements have the named fields overwritten;
    added elements are appended. Steps keep their embedded constraints; a
    constraint whose attachment no longer resolves is moved to top-level so the
    validator can still see it (and the log flags it)."""
    log: list[str] = []
    step_decisions = {d["id"]: d for d in decisions if d.get("kind") == "step"}
    con_decisions = {d["id"]: d for d in decisions if d.get("kind") == "constraint"}

    # 1. steps
    kept_steps: list[dict] = []
    for s in draft.get("steps", []):
        d = step_decisions.get(s["id"])
        act = d["action"] if d else "accept"
        if act == "reject":
            log.append(f"step:{s['id']} REJECT" + (f" -- {d.get('note','')}" if d and d.get("note") else ""))
            continue
        s2 = dict(s)
        if act == "edit" and d:
            for k, v in (d.get("edits") or {}).items():
                s2[k] = v
            log.append(f"step:{s['id']} EDIT {sorted((d.get('edits') or {}).keys())}"
                       + (f" -- {d.get('note','')}" if d.get("note") else ""))
        else:
            log.append(f"step:{s['id']} ACCEPT")
        kept_steps.append(s2)

    # added steps
    for d in decisions:
        if d.get("kind") == "step" and d.get("action") == "add":
            new = dict(d.get("edits") or {})
            new.setdefault("constraints", [])
            kept_steps.append(new)
            log.append(f"step:{new.get('id','?')} ADD -- {d.get('note','')}")

    kept_step_ids = {s["id"] for s in kept_steps}

    # 2. constraints (flatten, decide, then re-place)
    flat = flatten_constraints(draft)
    resolved: list[dict] = []
    for c in flat:
        d = con_decisions.get(c.get("id"))
        act = d["action"] if d else "accept"
        if act == "reject":
            log.append(f"constraint:{c.get('id')} REJECT" + (f" -- {d.get('note','')}" if d and d.get("note") else ""))
            continue
        c2 = dict(c)
        if act == "edit" and d:
            for k, v in (d.get("edits") or {}).items():
                c2[k] = v
            log.append(f"constraint:{c.get('id')} EDIT {sorted((d.get('edits') or {}).keys())}"
                       + (f" -- {d.get('note','')}" if d.get("note") else ""))
        else:
            log.append(f"constraint:{c.get('id')} ACCEPT")
        resolved.append(c2)

    for d in decisions:
        if d.get("kind") == "constraint" and d.get("action") == "add":
            resolved.append(dict(d.get("edits") or {}))
            log.append(f"constraint:{(d.get('edits') or {}).get('id','?')} ADD -- {d.get('note','')}")

    # 3. re-embed constraints under their (surviving) first attached step
    by_step: dict[str, list[dict]] = {sid: [] for sid in kept_step_ids}
    top_level: list[dict] = []
    for c in resolved:
        placed = False
        for r in _norm_refs(c):
            if r in by_step:
                by_step[r].append(c)
                placed = True
                break
        if not placed:
            top_level.append(c)
            if _norm_refs(c):
                log.append(f"constraint:{c.get('id')} attachment {_norm_refs(c)} no longer resolves -> top-level")

    out_steps = []
    for s in kept_steps:
        s = dict(s)
        s["constraints"] = by_step.get(s["id"], [])
        out_steps.append(s)

    gold = dict(draft)
    gold["steps"] = out_steps
    gold["constraints"] = top_level

    # 4. stamp quality + fold decision log into review_notes
    q = dict(gold.get("quality", {}))
    q.pop("_draft_diagnostics", None)  # drafting artifact never reaches gold
    if scope:
        q["annotation_scope"] = scope
    q["review_status"] = "reviewed"
    q["annotator"] = annotator
    q["review_date"] = review_date
    prior = q.get("review_notes", "")
    header = (f"Adjudicated {review_date} by {annotator} via scripts/adjudicate.py "
              f"from model-assisted draft. Decisions: "
              f"{sum(1 for line in log if 'ACCEPT' in line)} accept, "
              f"{sum(1 for line in log if 'EDIT' in line)} edit, "
              f"{sum(1 for line in log if 'REJECT' in line)} reject, "
              f"{sum(1 for line in log if 'ADD' in line)} add.")
    q["review_notes"] = header + "\n" + "\n".join(log)
    if prior and "Model-assisted DRAFT" not in prior:
        q["review_notes"] = prior + "\n---\n" + q["review_notes"]
    gold["quality"] = q
    return gold, log


# --- draft-vs-human alignment (diff mode) -----------------------------------
def _tokset(text: str) -> set[str]:
    return {w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split() if len(w) > 2}


def align_constraints(draft_cons: list[dict], human_cons: list[dict],
                      thresh: float = 0.4) -> dict[str, list]:
    """Greedy Jaccard alignment of constraint text between the two annotations.
    Returns {'both': [(d,h,score)], 'draft_only': [d], 'human_only': [h]}."""
    used_h: set[int] = set()
    both, draft_only = [], []
    for d in draft_cons:
        dt = _tokset(d.get("text", ""))
        best_j, best_s = -1, 0.0
        for j, h in enumerate(human_cons):
            if j in used_h:
                continue
            ht = _tokset(h.get("text", ""))
            if not dt or not ht:
                continue
            s = len(dt & ht) / len(dt | ht)
            if s > best_s:
                best_j, best_s = j, s
        if best_j >= 0 and best_s >= thresh:
            used_h.add(best_j)
            both.append((d, human_cons[best_j], round(best_s, 2)))
        else:
            draft_only.append(d)
    human_only = [h for j, h in enumerate(human_cons) if j not in used_h]
    return {"both": both, "draft_only": draft_only, "human_only": human_only}


# --- interactive prompting ---------------------------------------------------
def _ask(prompt: str, choices: str) -> str:
    while True:
        r = input(f"{prompt} [{choices}] ").strip().lower()
        if r in choices.split("/"):
            return r
        print(f"  please answer one of: {choices}")


def _edit_fields(obj: dict, fields: list[str]) -> dict:
    """Prompt for edits to named fields; blank keeps current value."""
    edits: dict[str, Any] = {}
    for f in fields:
        cur = obj.get(f)
        new = input(f"    {f} [{cur!r}]: ").strip()
        if not new:
            continue
        if f == "attached_to":
            edits[f] = [x.strip() for x in new.split(",") if x.strip()]
        else:
            edits[f] = new
    return edits


def interactive_review(draft: dict) -> list[dict]:
    """Single-draft element-by-element review -> decisions list."""
    decisions: list[dict] = []
    steps = draft.get("steps", [])
    print(f"\n=== {draft['procedure'].get('doc_id')} :: {draft['procedure'].get('title')} ===")
    print(f"{len(steps)} steps, {len(flatten_constraints(draft))} constraints. "
          "For each: (a)ccept / (e)dit / (r)eject.\n")
    for s in steps:
        print(f"STEP {s['id']}: {s.get('label','')}")
        act = _ask("  decision", "a/e/r")
        if act == "r":
            decisions.append({"kind": "step", "id": s["id"], "action": "reject",
                              "note": input("    reject reason: ").strip()})
            continue
        if act == "e":
            edits = _edit_fields(s, ["label", "action_verb", "action_object"])
            decisions.append({"kind": "step", "id": s["id"], "action": "edit", "edits": edits})
        else:
            decisions.append({"kind": "step", "id": s["id"], "action": "accept"})
        for c in s.get("constraints", []) or []:
            print(f"  CONSTRAINT {c['id']} [{c.get('type')}/{c.get('enforcement')}] -> {c.get('attached_to')}")
            print(f"    text: {c.get('text','')[:100]}")
            cact = _ask("    decision", "a/e/r")
            if cact == "r":
                decisions.append({"kind": "constraint", "id": c["id"], "action": "reject",
                                  "note": input("      reject reason: ").strip()})
            elif cact == "e":
                edits = _edit_fields(c, ["type", "enforcement", "text", "attached_to"])
                decisions.append({"kind": "constraint", "id": c["id"], "action": "edit", "edits": edits})
            else:
                decisions.append({"kind": "constraint", "id": c["id"], "action": "accept"})
    return decisions


def interactive_diff(draft: dict, human: dict, thresh: float = 0.4) -> list[dict]:
    """Reconcile draft vs independent human pass -> decisions list on the draft."""
    decisions: list[dict] = []
    dcons = flatten_constraints(draft)
    hcons = flatten_constraints(human)
    al = align_constraints(dcons, hcons, thresh)
    print(f"\n=== DIFF: {draft['procedure'].get('doc_id')} ===")
    print(f"agreements: {len(al['both'])}, model-only: {len(al['draft_only'])}, "
          f"human-only: {len(al['human_only'])}\n")
    print("-- AGREEMENTS (both annotators found; accept unless one is wrong) --")
    for d, h, sc in al["both"]:
        print(f"  ~{sc} {d['id']} [{d.get('type')}/{d.get('enforcement')}]: {d.get('text','')[:80]}")
        if d.get("type") != h.get("type") or d.get("enforcement") != h.get("enforcement"):
            print(f"      TYPE/ENF DISAGREE  model={d.get('type')}/{d.get('enforcement')}"
                  f"  human={h.get('type')}/{h.get('enforcement')}")
        act = _ask("    accept model / edit / reject", "a/e/r")
        if act == "r":
            decisions.append({"kind": "constraint", "id": d["id"], "action": "reject", "note": "diff-reject"})
        elif act == "e":
            edits = _edit_fields(d, ["type", "enforcement", "text", "attached_to"])
            decisions.append({"kind": "constraint", "id": d["id"], "action": "edit", "edits": edits})
        else:
            decisions.append({"kind": "constraint", "id": d["id"], "action": "accept"})
    print("\n-- MODEL-ONLY (human did not find; keep only if real) --")
    for d in al["draft_only"]:
        print(f"  {d['id']} [{d.get('type')}/{d.get('enforcement')}]: {d.get('text','')[:80]}")
        act = _ask("    keep / reject", "a/r")
        decisions.append({"kind": "constraint", "id": d["id"],
                          "action": "accept" if act == "a" else "reject", "note": "model-only"})
    print("\n-- HUMAN-ONLY (model missed; add if real) --")
    for h in al["human_only"]:
        print(f"  {h.get('id')} [{h.get('type')}/{h.get('enforcement')}]: {h.get('text','')[:80]}")
        act = _ask("    add / drop", "a/r")
        if act == "a":
            decisions.append({"kind": "constraint", "id": h.get("id"), "action": "add",
                              "edits": h, "note": "human-only, added in reconciliation"})
    return decisions


# --- IO ----------------------------------------------------------------------
def _today() -> str:
    return _dt.date.today().isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description="Adjudicate a model-assisted draft into a reviewed gold.")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_rev = sub.add_parser("review", help="single-draft element-by-element review")
    p_rev.add_argument("draft")
    p_diff = sub.add_parser("diff", help="reconcile draft vs independent human pass")
    p_diff.add_argument("draft")
    p_diff.add_argument("human")
    p_diff.add_argument("--thresh", type=float, default=0.4)
    p_rep = sub.add_parser("replay", help="non-interactive replay of a saved decisions file")
    p_rep.add_argument("draft")
    p_rep.add_argument("--decisions", required=True)

    for p in (p_rev, p_diff, p_rep):
        p.add_argument("--annotator", required=True)
        p.add_argument("--review-date", default=_today())
        p.add_argument("--scope", default="full_subprocedure")
        p.add_argument("--out-dir", default="datasets/paper/gold")
        p.add_argument("--decisions-out", default=None,
                       help="where to save the decision log for reproducible replay")
        p.add_argument("--dry-run", action="store_true", help="validate + print, do not write gold")

    args = ap.parse_args()
    draft = json.load(open(args.draft))

    if args.mode == "review":
        decisions = interactive_review(draft)
    elif args.mode == "diff":
        human = json.load(open(args.human))
        decisions = interactive_diff(draft, human, args.thresh)
    else:  # replay
        decisions = json.load(open(args.decisions))
        if isinstance(decisions, dict):
            decisions = decisions.get("decisions", [])

    gold, log = apply_decisions(draft, decisions, annotator=args.annotator,
                                review_date=args.review_date, scope=args.scope)

    errs = structural_errors(gold["steps"], flatten_constraints(gold))
    print("\n=== structural check on adjudicated gold ===")
    if errs:
        print(f"  {len(errs)} STRUCTURAL ERRORS (fix before this is a valid gold):")
        for e in errs:
            print("   -", e)
    else:
        print("  clean: locked vocab + enforcement + resolved attachments OK")
    n_steps = len(gold["steps"])
    n_cons = len(flatten_constraints(gold))
    print(f"  {n_steps} steps, {n_cons} constraints after adjudication")

    doc_id = gold["procedure"]["doc_id"]
    if args.decisions_out or (not args.dry_run):
        dpath = Path(args.decisions_out or f"decisions/{doc_id}.json")
        dpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"draft": str(args.draft), "annotator": args.annotator,
                   "review_date": args.review_date, "decisions": decisions},
                  open(dpath, "w"), indent=2)
        print(f"  decision log -> {dpath}")

    if args.dry_run:
        print("  --dry-run: gold NOT written")
        return 1 if errs else 0

    out = Path(args.out_dir) / f"{doc_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(gold, open(out, "w"), indent=2)
    print(f"  reviewed gold -> {out}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
