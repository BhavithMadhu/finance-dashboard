"""
main.py
───────
Entry point for the Personal Finance Dashboard.
Run:  python main.py
"""

import sys
from pathlib import Path

# Allow `from modules.xxx import ...` from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_sample_data import generate_transactions
from modules.ingest   import load_transactions
from modules.analysis import compute_kpis, detect_anomalies, forecast_next_month
from modules.visualize import render_all


BANNER = """
╔══════════════════════════════════════════════════════╗
║    💰  Personal Finance Dashboard  💰                ║
║    Python • NumPy • Pandas • Matplotlib              ║
╚══════════════════════════════════════════════════════╝
"""

RAW_CSV = Path("data/raw/transactions.csv")


def print_kpis(kpis: dict):
    print("\n─── 📈  KEY METRICS ───────────────────────────────────")
    for k, v in kpis.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float) and "rate" in k:
            print(f"  {label:<35}  {v:.1f}%")
        elif isinstance(v, float) and "income" in k or "expense" in k or "savings" in k:
            print(f"  {label:<35}  ₹{v:,.0f}")
        else:
            print(f"  {label:<35}  {v}")


def print_forecast(df):
    fc = forecast_next_month(df)
    print("\n─── 🔮  NEXT MONTH FORECAST ───────────────────────────")
    print(f"  Predicted spend   :  ₹{fc['predicted']:,.0f}")
    print(f"  Lower bound (−1σ) :  ₹{fc['lower_bound']:,.0f}")
    print(f"  Upper bound (+1σ) :  ₹{fc['upper_bound']:,.0f}")


def print_anomalies(df):
    anomalies = detect_anomalies(df, threshold=2.5)
    print(f"\n─── ⚠️   TOP ANOMALIES (z > 2.5) — {len(anomalies)} found ──────")
    for a in anomalies[:5]:
        print(f"  [{a.date}]  {a.description[:40]:<40}  ₹{a.amount:>8,.0f}  (z={a.z_score})")


def main():
    print(BANNER)

    # ── Step 1: Ensure sample data exists ───────────────────────────────
    if not RAW_CSV.exists():
        print("⚙️   No transaction CSV found. Generating sample data…")
        generate_transactions(months=6)

    # ── Step 2: Load & clean ─────────────────────────────────────────────
    print("\n📂  Loading transactions…")
    df = load_transactions(RAW_CSV)

    # ── Step 3: Print KPIs ───────────────────────────────────────────────
    kpis = compute_kpis(df)
    print_kpis(kpis)
    print_forecast(df)
    print_anomalies(df)

    # ── Step 4: Render all charts ────────────────────────────────────────
    render_all(df)

    print("\n🎉  Done! Open the 'outputs/' folder to view your charts.\n")


if __name__ == "__main__":
    main()
