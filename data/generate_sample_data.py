"""
generate_sample_data.py
Generates realistic Indian UPI/bank transaction data for testing.
Run this once to populate data/raw/transactions.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

MERCHANTS = {
    "Food & Dining": [
        "Swiggy", "Zomato", "McDonald's", "Domino's", "Starbucks",
        "Cafe Coffee Day", "Saravana Bhavan", "Local Mess", "BigBasket", "Blinkit"
    ],
    "Transport": [
        "Ola", "Uber", "Rapido", "TSRTC Bus", "Metro Recharge",
        "IndiGo Airlines", "RedBus", "Petrol Bunk - HP", "Petrol Bunk - IOC"
    ],
    "Shopping": [
        "Amazon", "Flipkart", "Myntra", "Reliance Digital", "D-Mart",
        "More Supermarket", "Nykaa", "Ajio", "Meesho"
    ],
    "Utilities": [
        "TSSPDCL Electricity", "Airtel Postpaid", "Jio Recharge",
        "ACT Broadband", "Hyderabad Water Board", "LPG Cylinder - HP Gas"
    ],
    "Entertainment": [
        "Netflix", "Hotstar", "Spotify", "Amazon Prime", "BookMyShow",
        "Steam Games", "YouTube Premium"
    ],
    "Health": [
        "Apollo Pharmacy", "MedPlus", "Practo Consultation",
        "Cult.fit", "Gym Membership", "1mg"
    ],
    "Education": [
        "Udemy", "Coursera", "GeeksforGeeks", "LeetCode Premium", "YouTube"
    ],
    "Investments": [
        "Zerodha", "Groww SIP", "PPFAS Mutual Fund", "SBI Life Insurance"
    ],
}

AMOUNT_RANGES = {
    "Food & Dining":    (80,  1200),
    "Transport":        (50,  800),
    "Shopping":         (200, 5000),
    "Utilities":        (300, 3500),
    "Entertainment":    (99,  999),
    "Health":           (100, 2000),
    "Education":        (500, 5000),
    "Investments":      (1000, 10000),
}

MONTHLY_SALARY = 85000


def generate_transactions(months: int = 6) -> pd.DataFrame:
    records = []
    start_date = datetime.today() - timedelta(days=months * 30)

    for m in range(months):
        month_start = start_date + timedelta(days=m * 30)

        # Salary credit on 1st of each month
        salary_date = month_start + timedelta(days=random.randint(0, 2))
        records.append({
            "date": salary_date.strftime("%d/%m/%Y"),
            "description": "SALARY CREDIT - TechCorp India Pvt Ltd",
            "amount": MONTHLY_SALARY + random.randint(-2000, 5000),
            "type": "Credit",
            "category": "Income",
        })

        # Random debits per category
        for category, merchants in MERCHANTS.items():
            freq = random.randint(3, 18)
            for _ in range(freq):
                txn_date = month_start + timedelta(days=random.randint(0, 29))
                lo, hi = AMOUNT_RANGES[category]
                amount = round(np.random.uniform(lo, hi), 2)
                merchant = random.choice(merchants)
                records.append({
                    "date": txn_date.strftime("%d/%m/%Y"),
                    "description": f"UPI/{merchant}/Pay",
                    "amount": amount,
                    "type": "Debit",
                    "category": category,
                })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
    df.to_csv("data/raw/transactions.csv", index=False)
    print(f"✅  Generated {len(df)} transactions → data/raw/transactions.csv")
    return df


if __name__ == "__main__":
    generate_transactions(months=6)
