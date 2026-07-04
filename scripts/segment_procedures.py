#!/usr/bin/env python3
"""Segment a whole manual into bounded, annotatable procedure units.

The IPKE-Bench seed documents are *whole manuals* (the filter-sampling SOP is
~108k words spanning dozens of sub-procedures). The locked annotation scope
(docs/annotation/guidelines.md, 2026-07-04) is **one coherent complete
procedure, 15-40 steps**. This script does the mechanical half of that scope
decision: it detects the document's dominant hierarchical heading scheme,
builds a section tree, and scores each leaf/near-leaf section for *procedural
richness* (imperative density, numbered-step count, modal/parameter signals)
so the annotator can pick the richest coherent unit instead of eyeballing a
100k-word file.

It does NOT decide the final boundary — that is a human/adjudicated call per
the guidelines. It produces ranked candidates + the exact character span of
each, which `annotate_assisted.py` then feeds to the model one unit at a time.

Usage:
    uv run python scripts/segment_procedures.py \
        --text datasets/paper/text/epa_field_operations_manual_filter_sampling_sop.txt \
        --out datasets/paper/segments/epa_field_operations_manual_filter_sampling_sop.segments.json \
        --top 15

    # All documents at once:
    uv run python scripts/segment_procedures.py \
        --text-dir datasets/paper/text \
        --out-dir datasets/paper/segments

Output JSON (per document):
    {
      "doc_id": "...",
      "scheme": "dotted_numeric" | "niosh_method" | "flat_numbered" | "fallback_blocks",
      "n_chars": 123456,
      "candidates": [
        {
          "section": "6.3",
          "title": "6.3 Filter Sample Collection",
          "char_start": 12000, "char_end": 15800,
          "n_words": 640, "depth": 2,
          "richness": {
            "score": 0.82, "n_numbered_children": 11, "n_imperatives": 24,
            "n_modals": 9, "n_parameters": 6, "n_step_lines": 11
          },
          "recommended": true      # 15-40 estimated steps & high richness
        }, ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# --- heading detectors -------------------------------------------------------
# Dotted numeric: "6.3", "6.3.3", "1.5.1.1" followed by a title on the same line.
RE_DOTTED = re.compile(r"^[ \t]*(\d+(?:\.\d+){0,4})\s+(\S.*)$")
# NIOSH NMAM method markers: a 4-5 digit method code line, often on its own.
RE_NIOSH = re.compile(r"^[ \t]*(?:METHOD|Method)?\s*(\d{4,5})\b.*$")
# Flat numbered steps: "1. Open the well." / "1) Open the well."
RE_FLAT = re.compile(r"^[ \t]*(\d{1,3})[.)]\s+(\S.*)$")
# OLSK / USGS-style: "01.1", "GWPD 1", "Step 4"
RE_ALT = re.compile(r"^[ \t]*((?:GWPD|Step|STEP)\s+\d+|\d{2}\.\d+)\s*(.*)$")

IMPERATIVE_HINT = re.compile(
    r"\b(attach|remove|open|close|record|measure|calibrate|collect|install|set|"
    r"place|insert|connect|check|verify|ensure|wear|clean|rinse|label|fill|"
    r"turn|press|select|adjust|mount|load|start|stop|wait|repeat|discard|"
    r"transfer|weigh|prepare|mix|apply|tighten|loosen|position|align|inspect|"
    r"document|sign|review|withdraw|store|seal|pour|read|note)\b",
    re.IGNORECASE,
)
MODAL_HINT = re.compile(r"\b(shall|must|should|may|will|required|do not|must not)\b", re.IGNORECASE)
PARAM_HINT = re.compile(
    r"(\d+\s*(?:-|to|–)\s*\d+)|(\d+\s*(?:%|L/min|mL|mg|kg|°C|C|mm|cm|psi|Hz|V|min|hours?|days?|years?))"
    r"|(\bevery\s+\d+)",
    re.IGNORECASE,
)
# A line that looks like an ordered sub-step under a section.
RE_STEP_LINE = re.compile(r"^[ \t]*(?:\d+(?:\.\d+){1,4}|\(?[a-z]\)|\d{1,3}[.)])\s+\S")
# ALLCAPS block header (NIOSH: method names, SAMPLING, MEASUREMENT, ...).
RE_ALLCAPS = re.compile(r"^[ \t]*([A-Z][A-Z0-9 ,/&()\-]{4,60})[ \t]*:?[ \t]*$")
# NIOSH procedural anchor: a SAMPLING (and optional SAMPLE PREPARATION) block.
RE_NIOSH_PROC = re.compile(r"^[ \t]*(SAMPLING|SAMPLE PREPARATION)[ \t]*:?[ \t]*$")


# Table-of-contents / list-of-figures line: dotted leaders or ends in a page number.
RE_TOC = re.compile(r"\.{4,}\s*\d+\s*$|\s\.\s\.\s\.|………")


def is_toc_line(ln: str) -> bool:
    if RE_TOC.search(ln):
        return True
    # ends in a bare page number after >=3 words and has leader dots anywhere
    if ".." in ln and re.search(r"\d+\s*$", ln):
        return True
    return False


def is_heading_title(title: str) -> bool:
    """A real section heading, not a wrapped prose line or footnote.

    Accept: Title Case ("Filter Sample Collection") or UPPERCASE
    ("MATERIALS AND SUPPLIES"). Reject: starts lowercase ("data collected
    must be invalidated"), or reads as a full sentence (long + ends with a
    period + not allcaps).
    """
    t = title.strip()
    if not t or not t[0].isupper():
        return False
    words = t.split()
    if len(words) > 12:
        return False
    if t.isupper():
        return True
    # Title Case: majority of alphabetic tokens start uppercase.
    alpha = [w for w in words if w[:1].isalpha()]
    if not alpha:
        return False
    cap = sum(1 for w in alpha if w[0].isupper())
    if cap / len(alpha) < 0.5:
        return False
    # Reject a capitalised full sentence (ends with '.' and has a lowercase verb-y tail).
    if t.endswith(".") and len(words) > 6 and not t.isupper():
        return False
    return True


@dataclass
class Section:
    number: str
    title: str
    depth: int
    line_start: int
    char_start: int
    char_end: int = 0
    line_end: int = 0
    text: str = ""
    children: list[str] = field(default_factory=list)


def detect_scheme(lines: list[str]) -> str:
    dotted = flat = alt = 0
    for ln in lines:
        if is_toc_line(ln):
            continue
        m = RE_DOTTED.match(ln)
        if m and m.group(1).count(".") >= 1 and is_heading_title(m.group(2).strip()):
            dotted += 1
        if RE_ALT.match(ln):
            alt += 1
        elif RE_FLAT.match(ln):
            flat += 1
    # NIOSH: repeated SAMPLING: procedural anchors are the giveaway.
    niosh_proc = sum(1 for ln in lines if RE_NIOSH_PROC.match(ln))
    if niosh_proc >= 5:
        return "niosh_method"
    # Multi-level dotted headings (e.g. 6.3.3) are strong structure; prefer them
    # over flat when there are enough real ones, since flat catches inline step
    # numbers and list items.
    if dotted >= 8:
        return "dotted_numeric"
    counts = {"dotted_numeric": dotted, "flat_numbered": flat, "alt_numbered": alt}
    scheme = max(counts, key=counts.get)
    if counts[scheme] < 3:
        return "fallback_blocks"
    return scheme


def parse_niosh(text: str, lines: list[str], char_offsets: list[int]) -> list[Section]:
    """Each NIOSH method's SAMPLING block is a procedure unit.

    A method is anchored at an ALLCAPS method-name header; its procedure spans
    from the `SAMPLING:` line to the next method-name header (or next SAMPLING).
    We title the unit with the nearest preceding ALLCAPS name.
    """
    # Index ALLCAPS header lines as potential method names (excluding the section
    # keywords themselves, which are also allcaps).
    section_kw = {"SAMPLING", "SAMPLE PREPARATION", "MEASUREMENT", "ACCURACY", "OVERALL",
                  "CALIBRATION AND QUALITY CONTROL", "INTERNAL CAPSULE", "GRAVIMETRIC",
                  "INJECTION", "ANALYSIS", "REAGENTS", "EQUIPMENT", "SPECIAL PRECAUTIONS",
                  "APPLICABILITY", "INTERFERENCES", "METHODS", "CHAPTERS"}
    name_at = {}
    for i, ln in enumerate(lines):
        m = RE_ALLCAPS.match(ln)
        if m:
            name = m.group(1).strip().rstrip(":").strip()
            if name not in section_kw and 2 <= len(name.split()) <= 6 and not name.isdigit():
                name_at[i] = name
    def nearest_name(li: int) -> str:
        best = None
        for i in name_at:
            if i < li:
                best = name_at[i]
            else:
                break
        return best or "NIOSH method"

    anchors = [i for i, ln in enumerate(lines) if RE_NIOSH_PROC.match(ln) and ln.strip().upper().startswith("SAMPLING")]
    secs: list[Section] = []
    for k, li in enumerate(anchors):
        name = nearest_name(li)
        secs.append(Section(number=f"M{k+1}", title=f"{name} — SAMPLING"[:140], depth=1,
                            line_start=li, char_start=char_offsets[li]))
    return secs


def parse_dotted(text: str, lines: list[str], char_offsets: list[int]) -> list[Section]:
    secs: list[Section] = []
    for i, ln in enumerate(lines):
        if is_toc_line(ln):
            continue
        m = RE_DOTTED.match(ln)
        if not m:
            continue
        num, title = m.group(1), m.group(2).strip()
        depth = num.count(".")
        # Require a real subsection (>=1 dot): bare integers ("2 Meter", "71825
        # Hydrogen Sulfide") are dimensions / parameter codes, not headings.
        if depth < 1:
            continue
        # A heading title is short-ish and not a full sentence ending in a period+lowercase.
        if len(title) > 120 or not is_heading_title(title):
            continue
        secs.append(Section(number=num, title=f"{num} {title}"[:140], depth=depth,
                            line_start=i, char_start=char_offsets[i]))
    return secs


def parse_generic(text: str, lines: list[str], char_offsets: list[int], regex: re.Pattern,
                  depth_fn=lambda m: 1) -> list[Section]:
    secs: list[Section] = []
    for i, ln in enumerate(lines):
        if is_toc_line(ln):
            continue
        m = regex.match(ln)
        if not m:
            continue
        num = m.group(1)
        title = (m.group(2).strip() if m.lastindex and m.lastindex >= 2 else "")[:120]
        secs.append(Section(number=num, title=f"{num} {title}".strip()[:140], depth=depth_fn(m),
                            line_start=i, char_start=char_offsets[i]))
    return secs


def close_spans(secs: list[Section], total_chars: int, total_lines: int) -> None:
    """Each section runs until the next heading of equal-or-shallower depth."""
    for idx, s in enumerate(secs):
        end_char = total_chars
        end_line = total_lines
        for t in secs[idx + 1:]:
            if t.depth <= s.depth:
                end_char = t.char_start
                end_line = t.line_start
                break
        s.char_end = end_char
        s.line_end = end_line


def richness(section_text: str) -> dict:
    lines = section_text.splitlines()
    n_step_lines = sum(1 for ln in lines if RE_STEP_LINE.match(ln))
    n_imp = len(IMPERATIVE_HINT.findall(section_text))
    n_modal = len(MODAL_HINT.findall(section_text))
    n_param = len(PARAM_HINT.findall(section_text))
    n_words = len(section_text.split())
    # Estimate step count: prefer explicit numbered step lines; fall back to imperative density.
    est_steps = n_step_lines if n_step_lines >= 3 else round(n_imp / 2)
    # Score: reward step count in the 15-40 sweet spot + constraint signal density.
    span_fit = 0.0
    if 15 <= est_steps <= 40:
        span_fit = 1.0
    elif 8 <= est_steps < 15:
        span_fit = 0.6
    elif 40 < est_steps <= 60:
        span_fit = 0.7
    elif 3 <= est_steps < 8:
        span_fit = 0.3
    density = min(1.0, (n_modal + n_param) / max(est_steps, 1) / 2.0) if est_steps else 0.0
    score = round(0.6 * span_fit + 0.4 * density, 3)
    return {
        "score": score, "est_steps": est_steps, "n_step_lines": n_step_lines,
        "n_imperatives": n_imp, "n_modals": n_modal, "n_parameters": n_param,
        "n_words": n_words,
    }


def segment_document(path: Path, top: int) -> dict:
    raw = path.read_text(errors="ignore")
    lines = raw.splitlines()
    # char offset of the start of each line
    offs, acc = [], 0
    for ln in lines:
        offs.append(acc)
        acc += len(ln) + 1
    scheme = detect_scheme(lines)

    if scheme == "dotted_numeric":
        secs = parse_dotted(raw, lines, offs)
    elif scheme == "niosh_method":
        secs = parse_niosh(raw, lines, offs)
    elif scheme == "alt_numbered":
        secs = parse_generic(raw, lines, offs, RE_ALT)
    elif scheme == "flat_numbered":
        secs = parse_generic(raw, lines, offs, RE_FLAT)
    else:
        secs = []

    candidates = []
    if secs:
        close_spans(secs, len(raw), len(lines))
        for s in secs:
            s.text = raw[s.char_start:s.char_end]
        # Candidate units: sections whose span is a plausible procedure (not the whole doc,
        # not a one-liner). Prefer depth>=1. Score each.
        for s in secs:
            n_words = len(s.text.split())
            if n_words < 40 or n_words > 6000:
                continue
            r = richness(s.text)
            candidates.append({
                "section": s.number,
                "title": s.title,
                "char_start": s.char_start, "char_end": s.char_end,
                "line_start": s.line_start + 1, "line_end": s.line_end,
                "depth": s.depth, "n_words": n_words,
                "richness": r,
                "recommended": r["score"] >= 0.6 and 10 <= r["est_steps"] <= 45,
            })
        candidates.sort(key=lambda c: c["richness"]["score"], reverse=True)
    else:
        # fallback: split into ~2000-word blocks so a human can still scan.
        words = raw.split()
        block = 1500
        for bi in range(0, len(words), block):
            chunk = " ".join(words[bi:bi + block])
            r = richness(chunk)
            candidates.append({
                "section": f"block_{bi // block}", "title": f"block {bi // block}",
                "char_start": None, "char_end": None, "depth": 0,
                "n_words": len(chunk.split()), "richness": r,
                "recommended": r["score"] >= 0.6,
            })
        candidates.sort(key=lambda c: c["richness"]["score"], reverse=True)

    return {
        "doc_id": path.stem,
        "scheme": scheme,
        "n_chars": len(raw),
        "n_sections_detected": len(secs),
        "candidates": candidates[:top],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", type=Path, help="single source .txt")
    ap.add_argument("--out", type=Path, help="output .segments.json (single)")
    ap.add_argument("--text-dir", type=Path, help="directory of source .txt files")
    ap.add_argument("--out-dir", type=Path, help="output directory (batch)")
    ap.add_argument("--top", type=int, default=15, help="max candidates to emit per doc")
    args = ap.parse_args()

    jobs: list[tuple[Path, Path]] = []
    if args.text:
        out = args.out or args.text.with_suffix(".segments.json")
        jobs.append((args.text, out))
    elif args.text_dir:
        outdir = args.out_dir or (args.text_dir.parent / "segments")
        outdir.mkdir(parents=True, exist_ok=True)
        for tf in sorted(args.text_dir.glob("*.txt")):
            jobs.append((tf, outdir / f"{tf.stem}.segments.json"))
    else:
        ap.error("provide --text or --text-dir")

    for src, out in jobs:
        result = segment_document(src, args.top)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        rec = sum(1 for c in result["candidates"] if c["recommended"])
        top = result["candidates"][0] if result["candidates"] else None
        top_s = f"{top['section']} (est {top['richness']['est_steps']} steps, score {top['richness']['score']})" if top else "none"
        print(f"{src.name}: scheme={result['scheme']} sections={result['n_sections_detected']} "
              f"candidates={len(result['candidates'])} recommended={rec} | top: {top_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
