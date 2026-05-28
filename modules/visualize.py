"""
modules/visualize.py
────────────────────
All chart generation functions.
Each function saves a PNG to outputs/ AND returns the Figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from modules.analysis import (
    monthly_summary,
    category_breakdown,
    monthly_category_pivot,
    rolling_expense_trend,
    detect_anomalies,
    forecast_next_month,
)

# ── Theming ────────────────────────────────────────────────────────────────

PALETTE = {
    "primary":    "#4F46E5",   # indigo
    "accent":     "#10B981",   # emerald
    "danger":     "#EF4444",   # red
    "warning":    "#F59E0B",   # amber
    "muted":      "#94A3B8",   # slate
    "bg":         "#0F172A",   # dark bg
    "card":       "#1E293B",   # card bg
    "text":       "#F1F5F9",   # light text
    "grid":       "#334155",   # grid lines
}

CAT_COLORS = [
    "#4F46E5", "#10B981", "#F59E0B", "#EF4444",
    "#8B5CF6", "#06B6D4", "#EC4899", "#F97316", "#84CC16",
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def _apply_dark_theme(fig: plt.Figure, axes):
    """Apply consistent dark theme to a figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(PALETTE["card"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        if ax.get_title():
            ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k" if x >= 1000 else f"₹{x:.0f}")
        )
    return fig


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  💾  Saved → {path}")
    return path


# ── 1. Monthly Income vs Expense Bar Chart ─────────────────────────────────

def plot_income_vs_expense(df: pd.DataFrame) -> plt.Figure:
    ms = monthly_summary(df)
    x  = np.arange(len(ms))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_inc = ax.bar(x - w/2, ms["income"],   width=w, color=PALETTE["accent"],  label="Income",   zorder=3)
    bars_exp = ax.bar(x + w/2, ms["expenses"], width=w, color=PALETTE["danger"],  label="Expenses", zorder=3)

    # Savings line overlay
    ax2 = ax.twinx()
    ax2.plot(x, ms["savings"], color=PALETTE["primary"], marker="o",
             linewidth=2.5, markersize=7, label="Savings", zorder=4)
    ax2.set_facecolor(PALETTE["card"])
    ax2.tick_params(colors=PALETTE["text"])
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k")
    )
    ax2.yaxis.label.set_color(PALETTE["text"])
    ax2.set_ylabel("Savings (₹)", color=PALETTE["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(ms["month_label"], rotation=15, ha="right")
    ax.set_ylabel("Amount (₹)")
    ax.set_title("Monthly Income vs Expenses", fontsize=14, fontweight="bold", pad=14)
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.5, zorder=0)

    lines, labels = ax2.get_legend_handles_labels()
    ax.legend(handles=[bars_inc, bars_exp] + lines,
              labels=["Income", "Expenses"] + labels,
              facecolor=PALETTE["card"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], loc="upper left")

    _apply_dark_theme(fig, ax)
    _save(fig, "01_income_vs_expense.png")
    return fig


# ── 2. Donut Chart — Category Breakdown ───────────────────────────────────

def plot_category_donut(df: pd.DataFrame) -> plt.Figure:
    cb    = category_breakdown(df)
    # Merge small slices into "Other"
    threshold = 2.0
    main  = cb[cb["pct"] >= threshold].copy()
    other = cb[cb["pct"] <  threshold]["total"].sum()
    if other > 0:
        main = pd.concat([
            main,
            pd.DataFrame([{"category": "Other", "total": other,
                           "pct": round(other / cb["total"].sum() * 100, 1)}])
        ], ignore_index=True)

    colors = CAT_COLORS[:len(main)]
    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        main["total"], labels=None, colors=colors,
        autopct="%1.1f%%", pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
        startangle=140,
    )
    for at in autotexts:
        at.set_color(PALETTE["text"])
        at.set_fontsize(8.5)

    centre = plt.Circle((0, 0), 0.45, color=PALETTE["card"])
    ax.add_artist(centre)
    total_spend = cb["total"].sum()
    ax.text(0, 0, f"₹{total_spend/1000:.1f}k\nTotal Spend",
            ha="center", va="center", fontsize=12,
            color=PALETTE["text"], fontweight="bold")

    legend = ax.legend(
        wedges, [f"{r.category} ({r.pct}%)" for r in main.itertuples()],
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, fontsize=8.5, framealpha=0,
        labelcolor=PALETTE["text"],
    )
    ax.set_title("Spend by Category", fontsize=14, fontweight="bold",
                 color=PALETTE["text"], pad=16)

    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    _save(fig, "02_category_donut.png")
    return fig


# ── 3. Spending Heatmap — Month × Category ────────────────────────────────

def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = monthly_category_pivot(df)
    pivot_k = pivot / 1000                           # show in ₹k

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot_k, ax=ax, annot=True, fmt=".1f",
        cmap="YlOrRd", linewidths=0.5, linecolor=PALETTE["bg"],
        cbar_kws={"label": "₹ (thousands)"},
        annot_kws={"size": 8.5},
    )
    ax.set_title("Monthly Spend Heatmap (₹k)", fontsize=13, fontweight="bold",
                 color=PALETTE["text"], pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=8.5)

    # Fix colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=PALETTE["text"])
    cbar.set_label("₹ (thousands)", color=PALETTE["text"])

    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color(PALETTE["text"])

    plt.tight_layout()
    _save(fig, "03_heatmap.png")
    return fig


# ── 4. Rolling Average Trend ───────────────────────────────────────────────

def plot_rolling_trend(df: pd.DataFrame, window: int = 3) -> plt.Figure:
    rt    = rolling_expense_trend(df, window)
    x     = np.arange(len(rt))
    col   = f"rolling_{window}m"

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(x, rt["expenses"], alpha=0.25, color=PALETTE["danger"])
    ax.plot(x, rt["expenses"],  color=PALETTE["danger"],  marker="o",
            linewidth=2, markersize=6, label="Monthly Spend")
    ax.plot(x, rt[col], color=PALETTE["warning"], linewidth=2.5,
            linestyle="--", label=f"{window}-Month Rolling Avg")

    ax.set_xticks(x)
    ax.set_xticklabels(rt["month_label"], rotation=15, ha="right")
    ax.set_title("Expense Trend & Rolling Average", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Amount (₹)")
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.4)

    _apply_dark_theme(fig, ax)
    _save(fig, "04_rolling_trend.png")
    return fig


# ── 5. Forecast Chart ─────────────────────────────────────────────────────

def plot_forecast(df: pd.DataFrame) -> plt.Figure:
    fc     = forecast_next_month(df)
    hist_y = fc["history_spend"]
    hist_x = list(range(len(hist_y)))
    next_x = len(hist_y)

    poly   = np.poly1d(fc["coefficients"])
    smooth_x = np.linspace(0, next_x, 200)
    smooth_y = poly(smooth_x)

    labels = fc["history_months"] + ["Next Month ▶"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(hist_x, hist_y, color=PALETTE["accent"], zorder=5, s=70, label="Actual Spend")
    ax.plot(smooth_x, smooth_y, color=PALETTE["primary"], linewidth=2.5,
            label="Poly Regression Fit")

    # Forecast point + confidence band
    ax.errorbar(
        next_x, fc["predicted"],
        yerr=[[fc["predicted"] - fc["lower_bound"]],
              [fc["upper_bound"] - fc["predicted"]]],
        fmt="D", color=PALETTE["warning"], markersize=10,
        capsize=6, capthick=2, elinewidth=2, label="Forecast ±1σ",
    )
    ax.axvline(next_x - 0.5, color=PALETTE["grid"], linestyle=":", linewidth=1.5)
    ax.text(next_x, fc["predicted"] * 1.04,
            f"₹{fc['predicted']/1000:.1f}k", ha="center",
            color=PALETTE["warning"], fontweight="bold", fontsize=10)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Next-Month Spend Forecast (Polynomial Regression)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Amount (₹)")
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.4)

    _apply_dark_theme(fig, ax)
    _save(fig, "05_forecast.png")
    return fig


# ── 6. Anomaly Scatter ────────────────────────────────────────────────────

def plot_anomalies(df: pd.DataFrame) -> plt.Figure:
    from modules.analysis import split_credits_debits
    _, debits = split_credits_debits(df)
    anomalies = detect_anomalies(df, threshold=2.0)
    anomaly_descs = {a.description for a in anomalies}

    fig, ax = plt.subplots(figsize=(13, 5))
    normal  = debits[~debits["description"].isin(anomaly_descs)]
    flagged = debits[debits["description"].isin(anomaly_descs)]

    ax.scatter(normal["date"],  normal["amount"],  alpha=0.35, s=25,
               color=PALETTE["muted"],  label="Normal", zorder=2)
    ax.scatter(flagged["date"], flagged["amount"], alpha=0.9,  s=70,
               color=PALETTE["danger"], label="Anomaly ⚠", zorder=3,
               edgecolors="white", linewidths=0.5)

    ax.set_title("Transaction Anomaly Detection (Z-score > 2.0)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Amount (₹)")
    ax.set_xlabel("Date")
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.grid(color=PALETTE["grid"], linestyle="--", alpha=0.3)

    fig.autofmt_xdate(rotation=20)
    _apply_dark_theme(fig, ax)
    _save(fig, "06_anomalies.png")
    return fig


# ── Master: render all charts ──────────────────────────────────────────────

def render_all(df: pd.DataFrame):
    print("\n📊  Rendering charts...")
    figs = {}
    figs["income_vs_expense"] = plot_income_vs_expense(df)
    figs["category_donut"]    = plot_category_donut(df)
    figs["heatmap"]           = plot_heatmap(df)
    figs["rolling_trend"]     = plot_rolling_trend(df)
    figs["forecast"]          = plot_forecast(df)
    figs["anomalies"]         = plot_anomalies(df)
    print(f"\n✅  All charts saved to '{OUTPUT_DIR}/'")
    return figs
