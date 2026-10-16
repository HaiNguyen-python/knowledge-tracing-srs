"""`ktrace report`: run the whole study and write docs/results.md plus two charts.

1. Generate synthetic practice logs, fit every model, score held-out learners.
2. Run the course simulation with each scheduler on 20 cohorts and compare final-test recall.
3. Repeat the scheduler comparison at other daily budgets (sensitivity check).

Every number in docs/results.md comes from this file; nothing is typed in by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import CEILING, Benchmark, beats_baselines, default_models, run_benchmark
from .metrics import calibration_table
from .models.baselines import GlobalMean, ItemMean
from .scheduler import CourseConfig, compare_schedulers, paired_gain, summarize
from .simulator import Catalog, generate_log, make_catalog

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREY = "#2a78d6", "#b7b7b2"
HLR_NAME = "HLR (half-life regression)"
MODEL_SCHED, ORACLE_SCHED = "Model-based (HLR)", "Model-based (true recall)"
FIXED = ["Leitner boxes", "SM-2"]
DATA_LABEL = "Synthetic data"


@dataclass(frozen=True)
class ReportConfig:
    n_learners: int = 400
    n_days: int = 30
    log_seed: int = 1
    split_seed: int = 0
    scheduler_seeds: tuple[int, ...] = tuple(range(100, 120))
    course: CourseConfig = field(default_factory=CourseConfig)
    budgets: tuple[int, ...] = (10, 20, 40)
    include_deep: bool = False


@dataclass
class Study:
    catalog: Catalog
    log: pd.DataFrame
    bench: Benchmark
    schedules: pd.DataFrame  # one row per (seed, scheduler) at the main budget
    sensitivity: pd.DataFrame  # (budget, scheduler) -> mean retention and reviews per day
    config: ReportConfig


def run_study(cfg: ReportConfig) -> Study:
    catalog = make_catalog()
    log = generate_log(catalog, cfg.n_learners, cfg.n_days, seed=cfg.log_seed)
    bench = run_benchmark(log, cfg.split_seed, default_models(cfg.include_deep))
    schedules = compare_schedulers(catalog, bench.bundle, cfg.scheduler_seeds, cfg.course)
    runs = []
    for budget in cfg.budgets:
        if budget == cfg.course.budget:
            res = schedules
        else:
            course = replace(cfg.course, budget=budget)
            res = compare_schedulers(catalog, bench.bundle, cfg.scheduler_seeds, course)
        runs.append(res.assign(budget=budget))
    sensitivity = (
        pd.concat(runs)
        .groupby(["budget", "scheduler"], sort=False)[["retention", "reviews_per_day"]]
        .mean()
        .reset_index()
    )
    return Study(catalog, log, bench, schedules, sensitivity, cfg)


def _fmt(x: float, digits: int = 3) -> str:
    return "n/a" if np.isnan(x) else f"{x:.{digits}f}"


def _best_fixed(summary: pd.DataFrame) -> str:
    return str(summary.loc[FIXED, "retention"].idxmax())


def results_markdown(study: Study) -> str:
    b, cfg, course = study.bench, study.config, study.config.course
    m = b.metrics
    n_train, n_test = b.split.train["user_id"].nunique(), b.split.test["user_id"].nunique()
    lines = [
        "# Results",
        "",
        f"Data: **{DATA_LABEL}** from `ktrace.simulator` (seeded). All numbers below come from "
        "`ktrace report`; rerun it to reproduce them.",
        "",
        "## 1. Predicting the next answer",
        "",
        f"Log: {len(study.log):,} answers from {cfg.n_learners} learners over {cfg.n_days} days, "
        f"{len(study.catalog)} words in {len(set(study.catalog.skills))} IELTS topics. "
        f"Models fit on {n_train} training learners before day {b.split.cutoff:.1f}; scored on "
        f"{len(b.scored):,} later answers of {n_test} held-out learners.",
        "",
        "| Model | AUC | Log loss | ECE | AUC (reviews only) | Beats both baselines |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in m.index:
        row = m.loc[name]
        if name == CEILING:
            beats = "(ceiling)"
        elif name in (GlobalMean.name, ItemMean.name):
            beats = "(baseline)"
        else:
            beats = "yes" if beats_baselines(b, name) else "no"
        lines.append(
            f"| {name} | {_fmt(row['auc'])} | {_fmt(row['log_loss'])} | {_fmt(row['ece'])} | "
            f"{_fmt(row['auc_reviews'])} | {beats} |"
        )
    lines += [
        "",
        f"`{CEILING}` scores the simulator's hidden P(correct): no model can beat it on average, "
        "because answers are random draws from it. ECE = expected calibration error (10 bins).",
        "",
        "AUC per IELTS topic:",
        "",
    ]
    cols = [c for c in b.per_skill.columns if c not in (GlobalMean.name,)]
    lines.append("| Topic | " + " | ".join(cols) + " |")
    lines.append("| --- |" + " --- |" * len(cols))
    for topic, row in b.per_skill.iterrows():
        lines.append(f"| {topic} | " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    w = b.bundle.model.to_dict()["w"]
    lines += [
        "",
        "Learned HLR weights (log2 half-life = w · x): "
        + ", ".join(f"`{k}` {v:+.2f}" for k, v in w.items())
        + ".",
        "",
        "## 2. Scheduling reviews",
        "",
        f"Course: {course.n_learners} new learners per cohort, {course.n_days} days, "
        f"{course.new_per_day} new words a day, at most {course.budget} reviews a day, final "
        f"test {course.test_delay_days:.0f} days after the last session. "
        f"{len(cfg.scheduler_seeds)} cohorts (seeds); every scheduler meets the same cohorts. "
        "Final-test recall = mean true P(correct) over the course words.",
        "",
        "| Scheduler | Final-test recall | 95% CI | Reviews used per day |",
        "| --- | --- | --- | --- |",
    ]
    summary = summarize(study.schedules)
    for name, row in summary.iterrows():
        lines.append(
            f"| {name} | {row['retention']:.3f} | {row['ci_low']:.3f} to {row['ci_high']:.3f} | "
            f"{row['reviews_per_day']:.1f} |"
        )
    lines += ["", "Paired difference, same cohorts:", ""]
    for other in [*FIXED, ORACLE_SCHED]:
        mean, low, high = paired_gain(study.schedules, MODEL_SCHED, other)
        lines.append(
            f"- {MODEL_SCHED} minus {other}: {mean:+.3f} (95% CI {low:+.3f} to {high:+.3f})"
        )
    sens = study.sensitivity
    names = list(dict.fromkeys(sens["scheduler"]))
    lines += [
        "",
        "Sensitivity to the daily review budget: mean final-test recall (reviews used per day).",
        "",
        "| Budget per day | " + " | ".join(names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for budget, g in sens.groupby("budget"):
        cells = [f"{r.retention:.3f} ({r.reviews_per_day:.1f})" for r in g.itertuples()]
        lines.append(f"| {budget} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def _header(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9.5, color=INK_2, va="top")


def plot_models(study: Study, path: Path) -> None:
    import matplotlib.pyplot as plt

    b = study.bench
    m = b.metrics.drop(index=CEILING, errors="ignore").sort_values("auc")
    best = str(m["auc"].idxmax())
    auc: dict[str, float] = {str(k): float(v) for k, v in b.metrics["auc"].items()}
    base_auc = auc[ItemMean.name]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.4), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.set_facecolor(SURFACE)
    colors = [BLUE if n in (best, HLR_NAME) else GREY for n in m.index]
    ax.barh(m.index, m["auc"], color=colors, height=0.6)
    for y, (_, row) in enumerate(m.iterrows()):
        inside = row["auc"] > 0.6  # short bars get their label outside
        ink = "white" if inside and colors[y] == BLUE else INK
        x = row["auc"] - 0.008 if inside else row["auc"] + 0.008
        ha = "right" if inside else "left"
        ax.text(x, y, f"{row['auc']:.3f}", va="center", ha=ha, fontsize=8.5, color=ink)
    if CEILING in b.metrics.index:
        ceiling = auc[CEILING]
        ax.axvline(ceiling, color=INK_2, linestyle=":", linewidth=1)
        ax.text(
            ceiling + 0.006,
            -0.45,
            f"true recall\n{ceiling:.3f}",
            fontsize=8,
            color=INK_2,
            va="bottom",
        )
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC on held-out learners (higher is better)", fontsize=9, color=INK_2)

    y_true = b.scored["correct"].to_numpy()
    ax2.plot([0, 1], [0, 1], color=INK_2, linestyle=":", linewidth=1)
    styles = {
        "BKT (per skill)": (GREY, "--", "BKT (per skill)"),
        "PFA (logistic)": (GREY, "-", "PFA"),
        HLR_NAME: (BLUE, "-", "HLR"),
    }
    for name, (color, dash, label) in styles.items():
        if name not in b.scored:
            continue
        table = calibration_table(y_true, b.scored[name].to_numpy())
        ax2.plot(
            table["mean_pred"],
            table["observed"],
            color=color,
            linestyle=dash,
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )
    legend = ax2.legend(loc="lower right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(INK_2)
    _style(ax2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Predicted P(correct)", fontsize=9, color=INK_2)
    ax2.set_ylabel("Observed share correct", fontsize=9, color=INK_2)
    hlr_auc = auc[HLR_NAME]
    _header(
        fig,
        f"History-aware models reach AUC {hlr_auc:.2f}; the item-mean baseline gets {base_auc:.2f}",
        f"Left: ranking quality. Right: reliability (dotted = perfect calibration). "
        f"{len(b.scored):,} answers of held-out learners. {DATA_LABEL}.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_schedulers(study: Study, path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = summarize(study.schedules).iloc[::-1]
    course = study.config.course
    fixed = _best_fixed(summary)
    fig, ax = plt.subplots(figsize=(10, 3.9), dpi=150)
    fig.set_facecolor(SURFACE)
    for y, (name, row) in enumerate(summary.iterrows()):
        color = BLUE if name == MODEL_SCHED else GREY
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [y, y],
            color=color,
            linewidth=3,
            solid_capstyle="round",
        )
        face = SURFACE if name == ORACLE_SCHED else color
        ax.plot(
            row["retention"],
            y,
            "o",
            markersize=9,
            color=color,
            markerfacecolor=face,
            markeredgewidth=2,
        )
        ax.text(
            row["ci_high"] + 0.012, y, f"{row['retention']:.1%}", va="center", fontsize=9, color=INK
        )
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([n + ("  (ceiling)" if n == ORACLE_SCHED else "") for n in summary.index])
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(0.8, float(summary["ci_high"].max()) + 0.1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(
        "Recall on the final test (mean and 95% CI across cohorts)", fontsize=9, color=INK_2
    )
    mb, fx = summary.loc[MODEL_SCHED, "retention"], summary.loc[fixed, "retention"]
    _header(
        fig,
        f"Model-based reviews: {mb:.0%} recall one week after the course vs {fx:.0%} with {fixed}",
        f"{len(study.catalog)} IELTS words, {course.n_days} days, at most {course.budget} "
        f"reviews/day, test {course.test_delay_days:.0f} days later. "
        f"{len(study.config.scheduler_seeds)} simulated cohorts. {DATA_LABEL}.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write(out_dir: Path, cfg: ReportConfig | None = None) -> Study:
    study = run_study(cfg or ReportConfig())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(results_markdown(study))
    plot_models(study, out_dir / "img" / "model_comparison.png")
    plot_schedulers(study, out_dir / "img" / "retention_by_scheduler.png")
    return study
