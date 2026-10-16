"""Report figures. White surface, quiet axes, a fixed hue per policy tier."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .config import FIG_DIR

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
NEUTRAL = "#a8a7a0"
TIER_COLOR = {"1": SERIES[0], "2": SERIES[1], "3+": SERIES[2]}
TIER_LABEL = {"1": "1 policy", "2": "2 policies", "3+": "3+ policies"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})
dollars = mticker.FuncFormatter(lambda v, _: f"${v/1000:g}k" if v >= 1000 else f"${v:,.0f}")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def clv_structure(df, tier):
    """The finding behind the whole project: CLV ~ premium x tier multiplier."""
    ratio = df["Customer Lifetime Value"] / df["Monthly Premium Auto"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for n in range(1, 10):
        m = df["Number of Policies"] == n
        t = tier[m].iloc[0]
        x = n + rng.uniform(-0.28, 0.28, m.sum())
        ax.scatter(x, ratio[m], s=6, alpha=0.35, color=TIER_COLOR[t], linewidths=0)
        ax.plot([n - 0.32, n + 0.32], [ratio[m].median()] * 2, color=INK, lw=2)
    ax.set_xticks(range(1, 10))
    ax.set(xlabel="Number of policies held", ylabel="CLV ÷ monthly premium")
    ax.set_title("CLV per premium dollar, by policies held", loc="left")
    ax.text(0.65, 55, "1 policy: ~39x", fontsize=9, color=INK)
    ax.text(3.6, 100, "3 to 9 policies: ~78x, flat", fontsize=9, color=INK)
    ax.text(2.45, 290, "2 policies: 85x to 366x in\ndiscrete bands, no feature\nexplains which band", fontsize=9, color=INK)

    ax = axes[1]
    for t in ["1", "3+", "2"]:
        m = tier == t
        ax.scatter(df.loc[m, "Monthly Premium Auto"], df.loc[m, "Customer Lifetime Value"],
                   s=6, alpha=0.4, color=TIER_COLOR[t], linewidths=0, label=TIER_LABEL[t])
    ax.yaxis.set_major_formatter(dollars)
    ax.set(xlabel="Monthly premium ($)", ylabel="Customer lifetime value")
    ax.set_title("CLV vs. premium: two straight lines and a cloud", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(TIER_LABEL[t]) for t in ["1", "2", "3+"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              markerscale=3, loc="upper left")
    fig.tight_layout()
    _save(fig, "clv_structure.png")


def cv_comparison(cv_df):
    d = cv_df[~cv_df["model"].str.startswith("Baseline")].sort_values("cv_mae_mean", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    y = np.arange(len(d))
    for ax, col, label, fmt in [
        (axes[0], "cv_mae", "Mean absolute error, $ (lower is better)", "${:,.0f}"),
        (axes[1], "cv_r2_dollar", "R², dollar scale (higher is better)", "{:.3f}"),
    ]:
        ax.errorbar(d[f"{col}_mean"], y, xerr=d[f"{col}_std"], fmt="o", color=SERIES[0],
                    elinewidth=2, capsize=4, ms=8)
        for yi, m in zip(y, d[f"{col}_mean"]):
            ax.text(m, yi + 0.22, fmt.format(m), ha="center", fontsize=9, color=INK)
        ax.set_yticks(y, d["model"])
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(label, loc="left")
    fig.suptitle("5-fold cross-validation on the training set (mean ± std)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "cv_model_comparison.png")


def actual_vs_predicted(y, pred, tier, name):
    fig, ax = plt.subplots(figsize=(7, 6))
    for t in ["1", "3+", "2"]:
        m = np.asarray(tier) == t
        ax.scatter(np.asarray(y)[m], np.asarray(pred)[m], s=8, alpha=0.5,
                   color=TIER_COLOR[t], linewidths=0, label=TIER_LABEL[t])
    lim = [1500, 90000]
    ax.plot(lim, lim, ls="--", color=NEUTRAL, lw=1, label="Perfect prediction")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="Actual CLV (log scale)", ylabel="Predicted CLV (log scale)")
    ticks = [2000, 5000, 10000, 20000, 50000]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator(ticks))
        axis.set_major_formatter(dollars)
        axis.set_minor_formatter(mticker.NullFormatter())
    handles, labels = ax.get_legend_handles_labels()
    want = [TIER_LABEL["1"], TIER_LABEL["2"], TIER_LABEL["3+"], "Perfect prediction"]
    ax.legend([handles[labels.index(w)] for w in want], want, markerscale=3, loc="upper left")
    ax.set_title(f"Actual vs. predicted, held-out test set ({name})", loc="left")
    fig.tight_layout()
    _save(fig, "actual_vs_predicted.png")
    return
    ax.set_title(f"Actual vs. predicted, held-out test set ({name})", loc="left")
    ax.legend(markerscale=3, loc="upper left")
    fig.tight_layout()
    _save(fig, "actual_vs_predicted.png")


def error_share(tier_table):
    t = tier_table.reindex(["1", "2", "3+"])
    x = np.arange(len(t))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w / 2 - 0.01, t["share_of_customers"], w, color=SERIES[0], label="Share of customers")
    b2 = ax.bar(x + w / 2 + 0.01, t["share_of_error"], w, color=SERIES[1], label="Share of total $ error")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.0%}",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [TIER_LABEL[i] for i in t.index])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title("Where the prediction error comes from (test set)", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "error_by_tier.png")


def interval_widths(report, bounds, coverage):
    """Floating bars: each tier's interval as a % range around the prediction."""
    a = (1 - coverage) / 2
    tiers = ["1", "2", "3+"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(tiers))[::-1]
    for yi, t in zip(y, tiers):
        lo, hi = np.exp(bounds[t][a]) - 1, np.exp(bounds[t][1 - a]) - 1
        ax.barh(yi, hi - lo, left=lo, color=TIER_COLOR[t], height=0.5)
        ax.text(max(hi, 0.05) + 0.03, yi,
                f"{lo:+.0%} to {hi:+.0%}   (test coverage {report.loc[t, 'coverage']:.0%})",
                va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, [TIER_LABEL[t] for t in tiers])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlim(-0.6, 2.0)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Interval around the predicted CLV")
    ax.set_title(f"{coverage:.0%} prediction intervals by policy tier", loc="left")
    fig.tight_layout()
    _save(fig, "prediction_intervals.png")


def importance(imp, name):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v, i, f"  ${v:,.0f}", va="center", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(dollars)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Increase in test MAE when the feature is shuffled")
    ax.set_title(f"Permutation importance ({name})", loc="left")
    fig.tight_layout()
    _save(fig, "permutation_importance.png")
