import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

STARBUCKS_GREEN = "#00704A"
STARBUCKS_LIGHT = "#CBA258"
NEUTRAL_GRAY = "#C0C0C0"
PALETTE = [STARBUCKS_GREEN, STARBUCKS_LIGHT, NEUTRAL_GRAY, "#1a1a2e", "#e94560"]

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("Saved figure: %s", path)
    plt.close(fig)


def bar_comparison(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    highlight_idx: int = 0,
    fmt: str = "{:.1f}",
    save_path: Path | None = None,
) -> plt.Figure:
    colors = [STARBUCKS_GREEN if i == highlight_idx else NEUTRAL_GRAY for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, pad=14)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            fmt.format(val),
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def funnel_chart(
    stages: list[str],
    counts: list[int],
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    pcts = [c / counts[0] * 100 for c in counts]
    colors = sns.color_palette("Greens_r", len(stages))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(stages[::-1], pcts[::-1], color=colors, edgecolor="white")
    for bar, cnt, pct in zip(bars, counts[::-1], pcts[::-1]):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}  ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("% of Received")
    ax.set_title(title, pad=14)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def uplift_heatmap(
    pivot: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Uplift (%)"},
    )
    ax.set_title(title, pad=14)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def hypothesis_result_table(results: list[dict], save_path: Path | None = None) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["decision"] = df["significant"].map({True: "Reject H₀", False: "Fail to Reject H₀"})
    df["p_value_str"] = df["p_value"].apply(
        lambda p: f"{p:.4f}" + (" *" if p < 0.05 else "")
    )
    return df
