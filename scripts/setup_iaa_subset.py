#!/usr/bin/env python3
"""
setup_iaa_subset.py -- prepare the >=30% independent double-annotation subset.

Inter-annotator agreement (IAA) is what lets IPKE-Bench claim its golds are
reproducible rather than one annotator's opinion. The resource-paper protocol
requires that at least 30% of the corpus be independently re-annotated by a
second annotator who has NOT seen the first-pass gold, and that agreement be
reported (scripts/compute_iaa.py: step / constraint / relation F1 + token-label
Cohen's kappa).

This script does three things:

  select    Choose the double-annotation subset (default: ceil(0.30 * N) docs),
            stratified across domains so agreement is measured on diverse
            procedure shapes, not N near-duplicates. Writes the manifest to
            datasets/paper/iaa_subset.json.

  scaffold  For each subset doc, emit an EMPTY independent-pass scaffold at
            datasets/paper/second_pass/<doc_id>.json. The scaffold carries ONLY
            what the second annotator legitimately needs -- doc_id, procedure
            title, and the exact source char span / section boundary that
            defines the unit -- and a steps/constraints skeleton they fill in.
            It deliberately does NOT contain the first-pass steps or
            constraints: that is the anchoring-bias control required by
            docs/annotation/independent-annotator-workflow.md. It also writes
            the raw source text of the unit to
            datasets/paper/second_pass/_source/<doc_id>.txt so the annotator
            reads exactly the same span the first pass did.

  report    Thin wrapper over scripts/compute_iaa.py: runs IAA over whichever
            subset docs have a completed second_pass file, writes the metrics
            JSON, and prints a short agreement summary + how many of the
            required subset are done.

Usage:
  python scripts/setup_iaa_subset.py select   --frac 0.30
  python scripts/setup_iaa_subset.py scaffold                # all subset docs
  python scripts/setup_iaa_subset.py report   --out results/iaa.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "datasets" / "paper" / "gold"
TEXT = REPO / "datasets" / "paper" / "text"
SECOND = REPO / "datasets" / "paper" / "second_pass"
SUBSET_MANIFEST = REPO / "datasets" / "paper" / "iaa_subset.json"


def _load_golds() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(GOLD.glob("*.json"))}


def stratified_select(golds: dict[str, dict], frac: float) -> list[str]:
    """Pick ceil(frac*N) docs, maximizing domain diversity. Deterministic:
    walks domains in first-seen order, taking the constraint-richest doc from
    each new domain first, then falls back to remaining richest docs."""
    n_target = max(1, math.ceil(frac * len(golds)))

    def n_constraints(g: dict) -> int:
        return sum(len(s.get("constraints", []) or []) for s in g.get("steps", [])) + \
               len(g.get("constraints", []) or [])

    rows = []
    for doc, g in golds.items():
        rows.append((doc, g["procedure"].get("domain") or "unknown", n_constraints(g)))

    chosen: list[str] = []
    seen_domains: set[str] = set()
    # first pass: one doc per domain (richest), in domain first-seen order
    for domain in dict.fromkeys(r[1] for r in rows):
        cand = sorted([r for r in rows if r[1] == domain and r[0] not in chosen],
                      key=lambda r: -r[2])
        if cand and len(chosen) < n_target:
            chosen.append(cand[0][0])
            seen_domains.add(domain)
    # second pass: fill remaining slots with richest not-yet-chosen
    if len(chosen) < n_target:
        rest = sorted([r for r in rows if r[0] not in chosen], key=lambda r: -r[2])
        for r in rest[: n_target - len(chosen)]:
            chosen.append(r[0])
    return chosen


def do_select(frac: float) -> list[str]:
    golds = _load_golds()
    subset = stratified_select(golds, frac)
    manifest = {
        "frac_required": frac,
        "n_total": len(golds),
        "n_subset": len(subset),
        "subset": subset,
        "domains": {d: golds[d]["procedure"].get("domain") for d in subset},
    }
    SUBSET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"IAA subset: {len(subset)}/{len(golds)} docs "
          f"({len(subset)/len(golds)*100:.0f}%, required {frac*100:.0f}%)")
    for d in subset:
        print(f"  - {d}  [{golds[d]['procedure'].get('domain')}]")
    print(f"manifest -> {SUBSET_MANIFEST.relative_to(REPO)}")
    return subset


def _empty_scaffold(gold: dict) -> dict:
    """A blank independent-pass file: source-defining fields ONLY, no first-pass
    steps/constraints (anchoring control)."""
    proc = gold["procedure"]
    src = proc.get("source", {})
    return {
        "procedure": {
            "doc_id": proc["doc_id"],
            "title": proc.get("title"),
            "version": proc.get("version"),
            "domain": proc.get("domain"),
            "source": {
                "doc_id": src.get("doc_id", proc["doc_id"]),
                "section": src.get("section"),
                "page": src.get("page"),
                "char_start": src.get("char_start"),
                "char_end": src.get("char_end"),
            },
        },
        # annotator fills these from the source text WITHOUT reading the gold:
        "steps": [
            {"id": "S1", "label": "", "action_verb": "", "action_object": "",
             "arguments": [], "parameters": [], "constraints": [],
             "flags": {}, "provenance": {"doc_id": proc["doc_id"], "section": src.get("section")}}
        ],
        "constraints": [],
        "relations": [],
        "quality": {
            "annotation_scope": gold["quality"].get("annotation_scope", "full_subprocedure"),
            "review_status": "reviewed",
            "annotator": "<YOUR_HANDLE>",
            "review_date": "<YYYY-MM-DD>",
            "review_notes": ("Independent second-pass annotation for IAA. Authored WITHOUT reading "
                             "datasets/paper/gold/<doc>.json (anchoring control). Same source span as "
                             "the first pass (see procedure.source char range / _source/<doc>.txt)."),
        },
    }


def do_scaffold(only: list[str] | None, force: bool = False) -> None:
    if not SUBSET_MANIFEST.exists():
        raise SystemExit("run `select` first (no iaa_subset.json)")
    manifest = json.loads(SUBSET_MANIFEST.read_text())
    subset = only or manifest["subset"]
    golds = _load_golds()
    SECOND.mkdir(parents=True, exist_ok=True)
    (SECOND / "_source").mkdir(parents=True, exist_ok=True)
    for doc in subset:
        if doc not in golds:
            print(f"  SKIP {doc}: no gold")
            continue
        out = SECOND / f"{doc}.json"
        if out.exists() and not force:
            # only preserve a genuine in-progress human file; refresh stale scaffolds
            existing = json.loads(out.read_text())
            handle = existing.get("quality", {}).get("annotator", "")
            if handle and handle not in ("<YOUR_HANDLE>",):
                print(f"  exists w/ annotator={handle!r} (not overwriting; pass --force) {out.relative_to(REPO)}")
                continue
        out.write_text(json.dumps(_empty_scaffold(golds[doc]), indent=2))
        # write the exact source span for the annotator to read
        src = golds[doc]["procedure"].get("source", {})
        full = (TEXT / f"{doc}.txt").read_text(errors="ignore")
        cs, ce = src.get("char_start"), src.get("char_end")
        span = full[cs:ce] if (cs is not None and ce is not None) else full
        (SECOND / "_source" / f"{doc}.txt").write_text(span)
        print(f"  scaffold -> {out.relative_to(REPO)}  (+ _source/{doc}.txt, {len(span.split())} words)")
    print("\nAnnotator: fill each scaffold from _source/<doc>.txt WITHOUT opening datasets/paper/gold/. "
          "Then run: python scripts/setup_iaa_subset.py report")


def do_report(out: Path) -> int:
    if not SUBSET_MANIFEST.exists():
        raise SystemExit("run `select` first")
    manifest = json.loads(SUBSET_MANIFEST.read_text())
    subset = manifest["subset"]
    done = [d for d in subset if (SECOND / f"{d}.json").exists()
            and "<YOUR_HANDLE>" not in (SECOND / f"{d}.json").read_text()]
    print(f"second-pass completed: {len(done)}/{len(subset)} required subset docs")
    for d in subset:
        print(f"  [{'x' if d in done else ' '}] {d}")
    if not done:
        print("no completed second-pass files yet; nothing to score.")
        return 1
    # compute_iaa scores every doc present in BOTH dirs; stage ONLY completed
    # subset docs into a temp dir so stale/non-subset second_pass files can't
    # pollute the run.
    import shutil
    import tempfile
    stage = Path(tempfile.mkdtemp(prefix="iaa_second_"))
    for d in done:
        shutil.copy(SECOND / f"{d}.json", stage / f"{d}.json")
    cmd = [sys.executable, str(REPO / "scripts" / "compute_iaa.py"),
           "--gold-dir", str(GOLD), "--second-dir", str(stage), "--out", str(out)]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO), env={"PYTHONPATH": str(REPO), **_env()},
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return r.returncode
    metrics = json.loads(out.read_text())
    agg = metrics.get("aggregate", {})
    print("\n=== IAA aggregate ===")
    for k in ("step_exact", "constraint_exact", "relation_exact"):
        block = agg.get(k) or {}
        f1 = block.get("f1")
        print(f"  {k:16} F1={f1:.3f}  P={block.get('precision')}  R={block.get('recall')}"
              if isinstance(f1, (int, float)) else f"  {k:16} F1=n/a")
    kappa = agg.get("token_label_kappa")
    print(f"  {'token_label':16} kappa={kappa:.3f}" if isinstance(kappa, (int, float))
          else "  token_label      kappa=n/a")
    print(f"metrics -> {out}")
    return 0


def _env() -> dict:
    import os
    return {k: v for k, v in os.environ.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare/report the >=30% IAA double-annotation subset.")
    sub = ap.add_subparsers(dest="mode", required=True)
    p_sel = sub.add_parser("select")
    p_sel.add_argument("--frac", type=float, default=0.30)
    p_sca = sub.add_parser("scaffold")
    p_sca.add_argument("--only", nargs="*", default=None)
    p_sca.add_argument("--force", action="store_true", help="overwrite existing scaffolds")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--out", type=Path,
                                                         default=REPO / "results" / "iaa.json")
    args = ap.parse_args()
    if args.mode == "select":
        do_select(args.frac)
        return 0
    if args.mode == "scaffold":
        do_scaffold(args.only, force=args.force)
        return 0
    if args.mode == "report":
        return do_report(args.out)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
