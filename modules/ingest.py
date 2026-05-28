"""
modules/ingest.py
─────────────────
Loads, validates, and cleans raw transaction CSVs.
Handles messy real-world bank exports (mixed date formats,
encoding issues, extra whitespace, duplicate rows, etc.)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Union


# ── Constants ──────────────────────────────────────────────────────────────

DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
    "%d %b %Y", "%d-%b-%Y", "%m/%d/%Y",
]

REQUIRED_COLUMNS = {"date", "description", "amount", "type"}

CATEGORY_KEYWORDS = {
    "Food & Dining":    r"swiggy|zomato|domino|mcdonald|starbucks|bigbasket|blinkit|mess|cafe|restaurant|hotel",
    "Transport":        r"ola|uber|rapido|tsrtc|metro|airline|indigo|bus|petrol|fuel|redbus",
    "Shopping":         r"amazon|flipkart|myntra|reliance|dmart|nykaa|ajio|meesho|more super",
    "Utilities":        r"electricity|airtel|jio|broadband|water board|lpg|gas|tsspdcl|act ",
    "Entertainment":    r"netflix|hotstar|spotify|prime|bookmyshow|steam|youtube premium",
    "Health":           r"apollo|medplus|practo|cult.fit|gym|1mg|pharma|clinic|hospital",
    "Education":        r"udemy|coursera|geeksforgeeks|leetcode|pluralsight|skillshare",
    "Investments":      r"zerodha|groww|ppfas|mutual fund|sip|insurance|nps",
    "Income":           r"salary|credit|neft cr|imps cr|refund|cashback",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_date(series: pd.Series) -> pd.Series:
    """Try multiple date formats; fall back to pandas inference."""
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, dayfirst=True)
            if parsed.notna().sum() > len(series) * 0.9:
                return parsed
        except Exception:
            continue
    return pd.to_datetime(series, infer_datetime_format=True, dayfirst=True, errors="coerce")


def _infer_category(description: str) -> str:
    """Rule-based category tagger using regex keyword matching."""
    desc = str(description).lower()
    for category, pattern in CATEGORY_KEYWORDS.items():
        if re.search(pattern, desc):
            return category
    return "Miscellaneous"


# ── Main loader ────────────────────────────────────────────────────────────

def load_transactions(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load and clean a raw CSV transaction file.

    Parameters
    ----------
    filepath : path to the raw CSV

    Returns
    -------
    Cleaned DataFrame with columns:
        date, description, amount, type, category, month, year, month_label
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # ── Load ────────────────────────────────────────────────────────────────
    df = pd.read_csv(
        path,
        encoding="utf-8",
        on_bad_lines="warn",
        skipinitialspace=True,
    )

    # ── Normalise column names ──────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── Clean strings ───────────────────────────────────────────────────────
    df["description"] = df["description"].astype(str).str.strip()
    df["type"]        = df["type"].astype(str).str.strip().str.title()   # Credit / Debit

    # ── Parse dates ─────────────────────────────────────────────────────────
    df["date"] = _parse_date(df["date"])
    invalid_dates = df["date"].isna().sum()
    if invalid_dates:
        print(f"⚠️  Dropped {invalid_dates} rows with unparseable dates.")
    df.dropna(subset=["date"], inplace=True)

    # ── Parse amounts ────────────────────────────────────────────────────────
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[₹,\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df.dropna(subset=["amount"], inplace=True)
    df["amount"] = df["amount"].abs()                    # ensure positive

    # ── Drop duplicates ──────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["date", "description", "amount", "type"], inplace=True)
    dupes = before - len(df)
    if dupes:
        print(f"🗑️  Removed {dupes} duplicate rows.")

    # ── Category tagging ─────────────────────────────────────────────────────
    if "category" not in df.columns:
        df["category"] = df["description"].apply(_infer_category)
    else:
        # Fill missing categories via inference
        mask = df["category"].isna() | (df["category"].str.strip() == "")
        df.loc[mask, "category"] = df.loc[mask, "description"].apply(_infer_category)

    # ── Derived time columns ─────────────────────────────────────────────────
    df["month"]       = df["date"].dt.month
    df["year"]        = df["date"].dt.year
    df["month_label"] = df["date"].dt.strftime("%b %Y")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✅  Loaded {len(df)} clean transactions ({df['date'].min().date()} → {df['date'].max().date()})")
    return df
