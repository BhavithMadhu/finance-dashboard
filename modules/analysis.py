"""
modules/analysis.py
───────────────────
All statistical computations:
  • Monthly income vs expense summary
  • Category-wise spend breakdown
  • Rolling averages & trend detection
  • Z-score anomaly detection
  • Savings rate & runway calculation
  • Next-month spend forecast (NumPy polynomial regression)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ── Data containers ────────────────────────────────────────────────────────

@dataclass
class MonthlySummary:
    month_label:   str
    income:        float
    expenses:      float
    savings:       float
    savings_rate:  float          # as a percentage

@dataclass
class AnomalyRecord:
    date:        str
    description: str
    amount:      float
    category:    str
    z_score:     float


# ── Splitters ──────────────────────────────────────────────────────────────

def split_credits_debits(df: pd.DataFrame):
    credits = df[df["type"] == "Credit"].copy()
    debits  = df[df["type"] == "Debit"].copy()
    return credits, debits


# ── 1. Monthly Income vs Expense ───────────────────────────────────────────

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        month_label, income, expenses, savings, savings_rate
    Ordered chronologically.
    """
    credits, debits = split_credits_debits(df)

    inc = (
        credits.groupby("month_label")["amount"]
        .sum()
        .rename("income")
    )
    exp = (
        debits.groupby("month_label")["amount"]
        .sum()
        .rename("expenses")
    )

    summary = pd.concat([inc, exp], axis=1).fillna(0)

    # Preserve chronological order
    order = (
        df[["month_label", "date"]]
        .drop_duplicates("month_label")
        .sort_values("date")["month_label"]
        .tolist()
    )
    summary = summary.reindex([m for m in order if m in summary.index])

    summary["savings"]      = summary["income"] - summary["expenses"]
    summary["savings_rate"] = np.where(
        summary["income"] > 0,
        (summary["savings"] / summary["income"] * 100).round(1),
        0.0,
    )
    return summary.reset_index()


# ── 2. Category Breakdown ──────────────────────────────────────────────────

def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spend by category (debits only).
    Adds a 'pct' column showing share of total spend.
    """
    _, debits = split_credits_debits(df)
    breakdown = (
        debits.groupby("category")["amount"]
        .agg(total="sum", txn_count="count", avg_txn="mean")
        .reset_index()
        .sort_values("total", ascending=False)
    )
    total_spend = breakdown["total"].sum()
    breakdown["pct"] = (breakdown["total"] / total_spend * 100).round(1)
    breakdown["avg_txn"] = breakdown["avg_txn"].round(2)
    return breakdown


# ── 3. Monthly Category Heatmap Data ──────────────────────────────────────

def monthly_category_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table: rows = month_label, cols = category, values = spend amount.
    """
    _, debits = split_credits_debits(df)
    pivot = debits.pivot_table(
        index="month_label", columns="category",
        values="amount", aggfunc="sum", fill_value=0,
    )
    # Chronological row order
    order = (
        df[["month_label", "date"]]
        .drop_duplicates("month_label")
        .sort_values("date")["month_label"]
        .tolist()
    )
    return pivot.reindex([m for m in order if m in pivot.index])


# ── 4. Rolling Average (3-month) ───────────────────────────────────────────

def rolling_expense_trend(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Monthly total expenses with a rolling mean overlay.
    """
    ms = monthly_summary(df)[["month_label", "expenses"]]
    ms[f"rolling_{window}m"] = (
        ms["expenses"]
        .rolling(window=window, min_periods=1)
        .mean()
        .round(2)
    )
    return ms


# ── 5. Anomaly Detection (Z-score) ────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> List[AnomalyRecord]:
    """
    Flags transactions whose amount is > `threshold` std-devs above
    the category mean. Returns a list of AnomalyRecord objects.
    """
    _, debits = split_credits_debits(df)
    anomalies: List[AnomalyRecord] = []

    for category, group in debits.groupby("category"):
        if len(group) < 3:           # too few data points for z-score
            continue
        mean   = group["amount"].mean()
        std    = group["amount"].std()
        if std == 0:
            continue
        z_scores = (group["amount"] - mean) / std
        flagged  = group[z_scores > threshold]
        for _, row in flagged.iterrows():
            anomalies.append(AnomalyRecord(
                date        = str(row["date"].date()),
                description = row["description"],
                amount      = row["amount"],
                category    = str(category),
                z_score     = round(float(z_scores.loc[row.name]), 2),
            ))

    return sorted(anomalies, key=lambda x: x.z_score, reverse=True)


# ── 6. Forecast: Next Month's Spend ───────────────────────────────────────

def forecast_next_month(df: pd.DataFrame, degree: int = 2) -> dict:
    """
    Polynomial regression (NumPy) on monthly expense totals.
    Returns:
        predicted   – forecasted spend (₹)
        lower_bound – predicted - 1σ residual
        upper_bound – predicted + 1σ residual
        coefficients – polynomial coefficients
    """
    ms      = monthly_summary(df)
    y       = ms["expenses"].values.astype(float)
    x       = np.arange(len(y), dtype=float)

    # Fit polynomial
    coeffs  = np.polyfit(x, y, deg=degree)
    poly    = np.poly1d(coeffs)

    # Residuals → confidence band
    fitted  = poly(x)
    residuals = y - fitted
    sigma   = float(np.std(residuals))

    next_x  = float(len(y))
    pred    = float(poly(next_x))

    return {
        "predicted":     round(max(pred, 0), 2),
        "lower_bound":   round(max(pred - sigma, 0), 2),
        "upper_bound":   round(pred + sigma, 2),
        "sigma":         round(sigma, 2),
        "coefficients":  coeffs.tolist(),
        "history_months": ms["month_label"].tolist(),
        "history_spend":  y.tolist(),
    }


# ── 7. Key Metrics (KPIs) ─────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Single-call summary of the most important financial KPIs.
    """
    ms       = monthly_summary(df)
    credits, debits = split_credits_debits(df)

    avg_income   = ms["income"].mean()
    avg_expense  = ms["expenses"].mean()
    avg_savings  = ms["savings"].mean()
    avg_rate     = ms["savings_rate"].mean()

    top_category = (
        debits.groupby("category")["amount"].sum().idxmax()
    )

    total_anomalies = len(detect_anomalies(df))

    savings_runway_months = (
        (ms["savings"].sum() / avg_expense)
        if avg_expense > 0 else float("inf")
    )

    return {
        "avg_monthly_income":   round(avg_income,  2),
        "avg_monthly_expense":  round(avg_expense, 2),
        "avg_monthly_savings":  round(avg_savings, 2),
        "avg_savings_rate_pct": round(avg_rate,    1),
        "top_spend_category":   top_category,
        "total_transactions":   len(df),
        "anomalies_detected":   total_anomalies,
        "savings_runway_months": round(savings_runway_months, 1),
    }
