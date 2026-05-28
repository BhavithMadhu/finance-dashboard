# 💰 Personal Finance Dashboard

A data analysis project that ingests bank/UPI transaction CSVs, cleans and analyzes spending patterns, detects anomalies, and renders a full suite of visualizations — including a polynomial regression forecast for next month's spend.

Built with **Python · NumPy · Pandas · Matplotlib · Seaborn**

---

## 📊 Charts Generated

| Chart | What it shows |
|-------|--------------|
| Income vs Expenses | Monthly bar chart with savings trend overlay |
| Category Donut | Spend breakdown by category with % share |
| Heatmap | Month × Category spend matrix |
| Rolling Trend | 3-month rolling average over actual expenses |
| Forecast | Polynomial regression prediction for next month |
| Anomaly Scatter | Z-score flagged unusual transactions |

---

## 🗂 Project Structure

```
finance_dashboard/
│
├── main.py                        # Entry point — run this
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── generate_sample_data.py    # Generates realistic UPI/bank CSV
│   └── raw/                       # Drop your transactions.csv here
│
├── modules/
│   ├── ingest.py                  # CSV cleaning, date parsing, deduplication
│   ├── analysis.py                # KPIs, z-score anomalies, forecasting
│   └── visualize.py               # All 6 dark-themed charts
│
└── outputs/                       # Charts saved here as PNG
```

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/finance-dashboard.git
cd finance-dashboard
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run**
```bash
python main.py
```

On first run, sample data is auto-generated. Charts are saved to `outputs/`.

---

## 🏦 Using Your Own Bank Data

Export your transactions from any Indian bank or UPI app (PhonePe, GPay, HDFC, ICICI, etc.) as a CSV and place it at `data/raw/transactions.csv`.

The file needs these four columns — column names are flexible, the cleaner normalizes them:

| Column | Example |
|--------|---------|
| `date` | `29/05/2026` |
| `description` | `UPI/Swiggy/Pay` |
| `amount` | `349.00` |
| `type` | `Debit` or `Credit` |

The cleaner handles mixed date formats, encoding issues, duplicate rows, and missing categories automatically.

---

## 🧠 Key Concepts Practiced

- **Pandas** — `groupby`, `pivot_table`, rolling windows, multi-format date parsing
- **NumPy** — `np.polyfit` for regression, z-score anomaly detection, confidence intervals
- **Matplotlib / Seaborn** — dual-axis charts, donut charts, heatmaps, error bars
- **Software design** — clean separation of ingestion, analysis, and visualization layers

---

## 📈 Sample Output

> Charts use a dark theme and are saved as high-resolution PNGs in `outputs/`

```
─── 📈  KEY METRICS ──────────────────────────────────
  Avg Monthly Income               ₹85,000
  Avg Monthly Expense              ₹61,200
  Avg Monthly Savings              ₹23,800
  Avg Savings Rate                 28.0%
  Top Spend Category               Food & Dining
  Anomalies Detected               3

─── 🔮  NEXT MONTH FORECAST ──────────────────────────
  Predicted spend   :  ₹63,400
  Lower bound (−1σ) :  ₹57,100
  Upper bound (+1σ) :  ₹69,700
```

---

## 🔭 Stretch Goals

- [ ] Streamlit web UI for interactive filtering
- [ ] PDF monthly report auto-generation
- [ ] Budget limit alerts per category
- [ ] Multi-account support

---

## 🛠 Requirements

```
numpy>=1.26
pandas>=2.2
matplotlib>=3.8
seaborn>=0.13
```

Python 3.9+ supported.

---

## 📄 License

MIT — free to use, modify, and distribute.
