#!/usr/bin/env bash

set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
GOLD_TIERB="${1:-$ROOT_DIR/datasets/archive/gold_human_tierb}"
export GOLD_TIERB
EMBED_MODEL="$ROOT_DIR/models/embeddings/all-mpnet-base-v2"
export EMBED_MODEL

python <<'PY'
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from tools import evaluate as evaluator
from src.graph.adapter import convert_to_tierb

ROOT = Path(os.environ.get("ROOT_DIR"))
GOLD_ROOT = Path(os.environ.get("GOLD_TIERB"))
EMBED_MODEL = os.environ.get("EMBED_MODEL")

if not ROOT or not ROOT.exists():
    raise SystemExit("ROOT_DIR missing or invalid")
if not GOLD_ROOT.exists():
    raise SystemExit(f"Gold Tier-B directory missing: {GOLD_ROOT}")

preprocessor, embedder = evaluator.prepare_evaluator(
    spacy_model="en_core_web_sm",
    embedding_model=EMBED_MODEL,
    device="cpu",
)

# Ensure Tier-B files exist for legacy predictions
for prediction_file in ROOT.glob("logs/**/predictions.json"):
    if prediction_file.parent.name == "tierb":
        continue
    tierb_dir = prediction_file.parent / "tierb"
    tierb_file = tierb_dir / "predictions.json"
    if tierb_file.exists():
        continue
    try:
        payload = json.loads(prediction_file.read_text())
    except json.JSONDecodeError:
        continue
    tierb_payload = convert_to_tierb(payload)
    tierb_dir.mkdir(parents=True, exist_ok=True)
    tierb_file.write_text(json.dumps(tierb_payload, indent=2), encoding="utf-8")

entries = []
tierb_pattern = ROOT.glob("logs/**/tierb")
for tierb_dir in tierb_pattern:
    if not tierb_dir.is_dir():
        continue
    predictions_file = tierb_dir / "predictions.json"
    if predictions_file.exists():
        entries.append(predictions_file)
    for extra in tierb_dir.glob("*.json"):
        if extra.name != "predictions.json":
            entries.append(extra)

def determine_method_and_doc(path: Path) -> tuple[str, str, Path, Path]:
    rel_parts = path.relative_to(ROOT).parts
    tierb_idx = rel_parts.index("tierb")
    if path.name == "predictions.json":
        doc_dir = path.parents[1]
        doc_id = doc_dir.name
        method_parts = rel_parts[1:tierb_idx - 1]
        method = "/".join(method_parts) or doc_dir.parent.name
    else:
        doc_id = path.stem
        method_parts = rel_parts[1:tierb_idx]
        method = "/".join(method_parts)
        doc_dir = path.parents[1]
    return method, doc_id, path, doc_dir

analysis: dict[str, dict[str, any]] = {}

for path in entries:
    method, doc_id, pred_path, doc_dir = determine_method_and_doc(path)
    gold_file = GOLD_ROOT / f"{doc_id}.json"
    if not gold_file.exists():
        continue

    try:
        tier_b_metrics = evaluator.run_evaluation(
            gold_file,
            pred_path,
            tiers=("B",),
            preprocessor=preprocessor,
            embedder=embedder,
        )
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"[tierb] Failed to evaluate {pred_path}: {exc}")
        continue

    tier_a_metrics = {}
    metrics_path = doc_dir / "metrics.json"
    if metrics_path.exists():
        try:
            tier_a_metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            tier_a_metrics = {}

    method_entry = analysis.setdefault(method, {"docs": {}})
    method_entry["docs"][doc_id] = {"tier_a": tier_a_metrics, "tier_b": tier_b_metrics}

def average_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                buckets[key].append(float(value))
    return {key: round(sum(vals) / len(vals), 3) for key, vals in buckets.items() if vals}

for method in analysis:
    docs = analysis[method]["docs"]
    tier_a_records = [entry["tier_a"] for entry in docs.values() if entry["tier_a"]]
    tier_b_records = [entry["tier_b"] for entry in docs.values() if entry["tier_b"]]
    analysis[method]["macro_tier_a"] = average_metrics(tier_a_records) if tier_a_records else {}
    analysis[method]["macro_tier_b"] = average_metrics(tier_b_records) if tier_b_records else {}

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)
analysis_path = results_dir / "tierb_analysis.json"
analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

def metric_or_dash(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"

table_lines = [
    "| Method | Tier | StepF1 | GraphF1 | NEXT_EdgeF1 | Logic_EdgeF1 |",
    "| --- | --- | --- | --- | --- | --- |",
]

for method in sorted(analysis.keys()):
    macro_a = analysis[method]["macro_tier_a"]
    macro_b = analysis[method]["macro_tier_b"]
    table_lines.append(
        "| {} | Tier-A | {} | {} | {} | {} |".format(
            method,
            metric_or_dash(macro_a.get("StepF1")),
            metric_or_dash(macro_a.get("GraphF1")),
            metric_or_dash(macro_a.get("NEXT_EdgeF1")),
            metric_or_dash(macro_a.get("Logic_EdgeF1")),
        )
    )
    table_lines.append(
        "| {} | Tier-B | {} | {} | {} | {} |".format(
            method,
            metric_or_dash(macro_b.get("StepF1")),
            metric_or_dash(macro_b.get("GraphF1")),
            metric_or_dash(macro_b.get("NEXT_EdgeF1")),
            metric_or_dash(macro_b.get("Logic_EdgeF1")),
        )
    )

table_path = results_dir / "tierb_comparison_table.md"
table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

def draw_chart(methods: list[str], graph_a: list[float], graph_b: list[float], path: Path) -> None:
    if not methods:
        return
    width = max(600, len(methods) * 160)
    height = 420
    margin_left, margin_right, margin_top, margin_bottom = 100, 40, 40, 80
    chart_height = height - margin_top - margin_bottom
    chart_width = width - margin_left - margin_right
    pixels = [bytearray([255, 255, 255] * width) for _ in range(height)]

    def draw_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]):
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(y0, y1 + 1):
            row = pixels[y]
            for x in range(x0, x1 + 1):
                idx = x * 3
                row[idx: idx + 3] = bytes(color)

    def draw_line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            draw_rect(x0, y0, x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    FONT = {
        "A": ["  #  ", " # # ", "#   #", "#####", "#   #", "#   #", "#   #"],
        "B": ["#### ", "#   #", "#### ", "#   #", "#   #", "#   #", "#### "],
        "C": [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
        "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
        "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
        "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
        "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####"],
        "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
        "I": [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
        "J": ["  ###", "   #", "   #", "   #", "   #", "#  #", " ## "],
        "K": ["#  # ", "# #  ", "##   ", "###  ", "#  # ", "#  # ", "#   #"],
        "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
        "M": ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"],
        "N": ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
        "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
        "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
        "Q": [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
        "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
        "S": [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
        "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
        "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
        "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
        "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
        "X": ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
        "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
        "Z": ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
        "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
        "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
        "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
        "3": [" ### ", "#   #", "    #", " ### ", "    #", "#   #", " ### "],
        "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
        "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
        "6": [" ### ", "#   #", "#    ", "#### ", "#   #", "#   #", " ### "],
        "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
        "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
        "9": [" ### ", "#   #", "#   #", " ####", "    #", "#   #", " ### "],
        "-": ["     ", "     ", "     ", " ### ", "     ", "     ", "     "],
        "_": ["     ", "     ", "     ", "     ", "     ", "     ", "#####"],
        "/": ["    #", "   # ", "  #  ", " #   ", "#    ", "     ", "     "],
        " ": ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
    }

    def draw_text(x: int, y: int, text: str, color: tuple[int, int, int]):
        cursor_x = x
        text = text.upper()
        for char in text:
            glyph = FONT.get(char, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, pixel in enumerate(row):
                    if pixel == "#":
                        draw_rect(cursor_x + gx, y + gy, cursor_x + gx, y + gy, color)
            cursor_x += 6

    # Axes
    axis_color = (0, 0, 0)
    draw_line(margin_left, height - margin_bottom, width - margin_right, height - margin_bottom, axis_color)
    draw_line(margin_left, margin_top, margin_left, height - margin_bottom, axis_color)

    max_value = max(graph_a + graph_b + [0.0])
    max_value = max_value if max_value > 0 else 1.0
    spacing = chart_width / len(methods)
    bar_width = spacing * 0.3
    tier_a_color = (31, 119, 180)
    tier_b_color = (255, 127, 14)

    for idx, method in enumerate(methods):
        base_x = margin_left + spacing * idx + spacing / 2
        def bar(value: float, offset: float, color: tuple[int, int, int]):
            height_ratio = value / max_value
            top = int((height - margin_bottom) - height_ratio * chart_height)
            left = int(base_x + offset - bar_width / 2)
            right = int(base_x + offset + bar_width / 2)
            draw_rect(left, top, right, height - margin_bottom - 1, color)
        bar(graph_a[idx], -bar_width / 2, tier_a_color)
        bar(graph_b[idx], bar_width / 2, tier_b_color)
        label = method.split('/')[-1]
        draw_text(int(base_x - len(label) * 3), height - margin_bottom + 10, label[:12], axis_color)

    draw_text(margin_left, margin_top - 25, "GraphF1 Tier Comparison", axis_color)
    draw_text(margin_left + 10, margin_top + 10, "Tier-A", tier_a_color)
    draw_rect(margin_left - 20, margin_top + 10, margin_left - 10, margin_top + 20, tier_a_color)
    draw_text(margin_left + 10, margin_top + 30, "Tier-B", tier_b_color)
    draw_rect(margin_left - 20, margin_top + 30, margin_left - 10, margin_top + 40, tier_b_color)

    import struct, zlib

    def write_png(out_path: Path, rows: list[bytearray]):
        with out_path.open("wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
            def chunk(tag: bytes, data: bytes) -> None:
                handle.write(struct.pack(">I", len(data)))
                handle.write(tag)
                handle.write(data)
                crc = zlib.crc32(tag)
                crc = zlib.crc32(data, crc)
                handle.write(struct.pack(">I", crc & 0xFFFFFFFF))
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            raw = b"".join(b"\x00" + bytes(row) for row in rows)
            chunk(b"IDAT", zlib.compress(raw, 9))
            chunk(b"IEND", b"")

    write_png(path, pixels)

chart_path = results_dir / "tierb_comparison_chart.png"
draw_chart(
    sorted(analysis.keys()),
    [analysis[m]["macro_tier_a"].get("GraphF1") or 0.0 for m in sorted(analysis.keys())],
    [analysis[m]["macro_tier_b"].get("GraphF1") or 0.0 for m in sorted(analysis.keys())],
    chart_path,
)

print(f"[tierb] Analysis written to {analysis_path}")
print(f"[tierb] Table written to {table_path}")
print(f"[tierb] Chart written to {chart_path}")
PY
