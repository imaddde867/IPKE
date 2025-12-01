import json
import os
from glob import glob

RUNS_GLOB = os.path.join("logs", "baseline_runs", "run_*", "evaluation_report.json")
OUT_DIR = os.path.join("logs", "baseline_runs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CSV_MACRO = os.path.join(OUT_DIR, "baseline_macro_metrics_by_run.csv")
CSV_TRENDS = os.path.join(OUT_DIR, "baseline_metric_trends.csv")
CSV_AGG = os.path.join(OUT_DIR, "baseline_aggregate_averages.csv")
AGG_PATH = os.path.join(OUT_DIR, "aggregate_metrics.json")

METRICS = [
    "StepF1",
    "AdjacencyF1",
    "Kendall",
    "ConstraintCoverage",
    "ConstraintAttachmentF1",
    "A_score",
]


def load_reports():
    reports = []
    for path in sorted(glob(RUNS_GLOB)):
        with open(path, "r") as f:
            report = json.load(f)
        run_name = os.path.basename(os.path.dirname(path))
        reports.append((run_name, report))
    return reports


def load_macro_metrics(reports):
    rows = []
    for run_name, report in reports:
        macro = report.get("macro_avg", {})
        row = {"run": run_name}
        for m in METRICS:
            row[m] = float(macro.get(m, 0.0))
        rows.append(row)
    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = ["run"] + METRICS
    with open(out_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join([str(r[h]) for h in headers]) + "\n")

def build_metric_trend_rows(reports):
    if not reports:
        return []
    # Identify documents (exclude macro_avg)
    doc_names = [k for k in reports[0][1].keys() if k != "macro_avg"]
    doc_names.sort()
    run_names = [r[0] for r in reports]

    rows = []
    for m in METRICS:
        for doc in doc_names:
            row = {"metric": m, "document": doc}
            vals = []
            for rn, report in reports:
                v = float(report.get(doc, {}).get(m, 0.0))
                row[rn] = v
                vals.append(v)
            mean_v = sum(vals) / len(vals) if vals else 0.0
            mu = mean_v
            std_v = ((sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5) if len(vals) > 1 else 0.0
            row["mean"] = mean_v
            row["std"] = std_v
            row["delta_from_run_1"] = vals[-1] - vals[0] if len(vals) >= 2 else 0.0
            rows.append(row)
    return rows, run_names


def write_metric_trend_csv(rows, run_names, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = ["metric", "document"] + run_names + ["mean", "std", "delta_from_run_1"]
    with open(out_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join([str(r.get(h, "")) for h in headers]) + "\n")


def try_plot_macro(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping plots:", e)
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Line plot across runs for each metric
    x = [r["run"] for r in rows]
    plt.figure(figsize=(10, 5))
    for m in METRICS:
        y = [r[m] for r in rows]
        plt.plot(x, y, marker="o", label=m)
    plt.title("Baseline Macro Metrics Across Runs")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    line_path = os.path.join(PLOTS_DIR, "baseline_macro_metrics_across_runs.png")
    plt.savefig(line_path, dpi=200)
    plt.close()

    # Bar chart with mean and std for each metric
    import math
    import statistics as stats

    means = []
    stds = []
    for m in METRICS:
        vals = [r[m] for r in rows]
        means.append(sum(vals) / len(vals) if vals else float("nan"))
        stds.append(stats.pstdev(vals) if len(vals) > 1 else 0.0)

    plt.figure(figsize=(8, 5))
    idx = range(len(METRICS))
    plt.bar(idx, means, yerr=stds, capsize=4)
    plt.xticks(list(idx), METRICS, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Score (mean ± std)")
    plt.title("Baseline Macro Metrics — Mean ± Std over 5 Runs")
    plt.tight_layout()
    bar_path = os.path.join(PLOTS_DIR, "baseline_macro_metrics_mean_std.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    print("Saved macro plots:")
    print(" -", line_path)
    print(" -", bar_path)


def try_plot_metric_trends(rows, run_names):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping metric trend plots:", e)
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # For each metric, plot three document lines over runs
    metrics = sorted(set(r["metric"] for r in rows))
    docs = sorted(set(r["document"] for r in rows))

    for m in metrics:
        plt.figure(figsize=(8, 5))
        for doc in docs:
            r = next(rr for rr in rows if rr["metric"] == m and rr["document"] == doc)
            y = [r[rn] for rn in run_names]
            plt.plot(run_names, y, marker="o", label=doc)
        plt.title(f"{m} over Rounds by Document")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        out_path = os.path.join(PLOTS_DIR, f"{m}_trend_by_document.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved:", out_path)


def load_aggregate(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_aggregate_csv(agg, out_path):
    # agg is a dict: {doc: {metric: val}, "macro_avg": {...}}
    if not agg:
        return
    docs = [k for k in agg.keys() if k != "macro_avg"]
    docs.sort()
    headers = ["document"] + METRICS
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for d in docs + ["macro_avg"]:
            row = [d]
            for m in METRICS:
                row.append(str(float(agg.get(d, {}).get(m, 0.0))))
            f.write(",".join(row) + "\n")


def try_plot_aggregate_averages(agg):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("matplotlib not available, skipping aggregate average plots:", e)
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    docs = [k for k in agg.keys() if k != "macro_avg"]
    docs.sort()

    # Overview: grouped bars per metric with one bar per document
    x = np.arange(len(METRICS))
    width = 0.22 if len(docs) == 3 else 0.8 / max(1, len(docs))
    plt.figure(figsize=(10, 5))
    for i, d in enumerate(docs):
        vals = [float(agg[d].get(m, 0.0)) for m in METRICS]
        plt.bar(x + i * width, vals, width=width, label=d)
    plt.xticks(x + (len(docs) - 1) * width / 2, METRICS, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Average Score")
    plt.title("Aggregate Averages by Metric and Document")
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    overview_path = os.path.join(PLOTS_DIR, "aggregate_averages_overview.png")
    plt.savefig(overview_path, dpi=200)
    plt.close()
    print("Saved:", overview_path)

    # Per-metric bars: three bars (one per document)
    for m in METRICS:
        plt.figure(figsize=(6, 4))
        vals = [float(agg[d].get(m, 0.0)) for d in docs]
        plt.bar(docs, vals)
        plt.ylim(0, 1.05)
        plt.ylabel("Average Score")
        plt.title(f"{m} — Aggregate Average by Document")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(PLOTS_DIR, f"aggregate_avg_{m}.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print("Saved:", p)


def main():
    reports = load_reports()
    if not reports:
        raise SystemExit("No runs found at " + RUNS_GLOB)

    # Macro CSV (unchanged)
    macro_rows = load_macro_metrics(reports)
    write_csv(macro_rows, CSV_MACRO)
    print("Wrote:", CSV_MACRO)

    # Metric trends CSV (documents x runs)
    trend_rows, run_names = build_metric_trend_rows(reports)
    write_metric_trend_csv(trend_rows, run_names, CSV_TRENDS)
    print("Wrote:", CSV_TRENDS)

    # Aggregate averages (from aggregate_metrics.json)
    agg = load_aggregate(AGG_PATH)
    if agg:
        write_aggregate_csv(agg, CSV_AGG)
        print("Wrote:", CSV_AGG)

    # Plots
    try:
        try_plot_macro(macro_rows)
        try_plot_metric_trends(trend_rows, run_names)
        if agg:
            try_plot_aggregate_averages(agg)
    except Exception as e:
        print("Plotting failed:", e)


if __name__ == "__main__":
    main()
