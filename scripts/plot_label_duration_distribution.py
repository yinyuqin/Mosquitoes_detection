"""Plot HumBugDB label-duration distributions in three subplots.

The default logarithmic x-axis keeps sub-second labels visible while still
showing the long tail of background recordings.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data" / "raw" / "humbugdb_0_0_1" / "neurips_2021_zenodo_0_0_1.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "humbugdb_label_duration_distribution.png"
CATEGORIES = ("mosquito", "background", "audio")
COLORS = {
    "mosquito": "#2A9D8F",
    "background": "#E76F51",
    "audio": "#457B9D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot duration histograms for the three HumBugDB sound types."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Metadata CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear x-axis instead of the default logarithmic axis",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI")
    return parser.parse_args()


def load_durations(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {category: [] for category in CATEGORIES}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            category = (row.get("sound_type") or "").strip().lower()
            if category not in values:
                continue
            try:
                duration = float(row.get("length") or "nan")
            except ValueError as exc:
                raise ValueError(f"Invalid length at CSV row {row_number}") from exc
            if math.isfinite(duration) and duration > 0:
                values[category].append(duration)

    missing = [category for category, durations in values.items() if not durations]
    if missing:
        raise ValueError(f"No valid positive durations for: {', '.join(missing)}")
    return {category: np.asarray(durations) for category, durations in values.items()}


def seconds_label(value: float, _position: float) -> str:
    if value < 0.01:
        return f"{value:.3g}"
    if value < 1:
        return f"{value:.2g}"
    return f"{value:g}"


def plot_distributions(
    durations: dict[str, np.ndarray], output: Path, bins: int, linear: bool, dpi: int
) -> None:
    if bins < 2:
        raise ValueError("--bins must be at least 2")

    all_values = np.concatenate(tuple(durations.values()))
    if linear:
        # Limit the default linear view to P99.5 so a few very long background
        # recordings do not flatten the useful part of every histogram.
        upper = float(np.percentile(all_values, 99.5))
        bin_edges = np.linspace(0, upper, bins + 1)
    else:
        lower = float(all_values.min())
        upper = float(all_values.max())
        bin_edges = np.geomspace(lower, upper, bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=not linear, sharey=True)
    for axis, category in zip(axes, CATEGORIES):
        values = durations[category]
        median = float(np.median(values))
        p95 = float(np.percentile(values, 95))
        weights = np.full(values.size, 100.0 / values.size)

        axis.hist(
            values,
            bins=bin_edges,
            weights=weights,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
        )
        axis.axvline(median, color="#222222", linestyle="--", linewidth=1.5, label=f"Median: {median:.2f}s")
        axis.axvline(p95, color="#6A1B9A", linestyle=":", linewidth=1.8, label=f"P95: {p95:.2f}s")
        axis.set_title(f"{category.capitalize()} (n={values.size:,})", fontweight="bold")
        axis.set_xlabel("Label duration (seconds)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.legend(frameon=False, fontsize=9, loc="upper right")

        if linear:
            overflow = int(np.count_nonzero(values > upper))
            axis.set_xlim(0, upper)
            if overflow:
                axis.text(
                    0.98,
                    0.79,
                    f"> {upper:.1f}s: {overflow}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                )
        else:
            axis.set_xscale("log")
            axis.xaxis.set_major_formatter(FuncFormatter(seconds_label))

    axes[0].set_ylabel("Share of category labels per bin (%)")
    fig.suptitle("HumBugDB Label Duration Distributions", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Dashed line = median; dotted line = 95th percentile",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    durations = load_durations(args.input)
    plot_distributions(durations, args.output, args.bins, args.linear, args.dpi)
    print(f"Saved chart to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
