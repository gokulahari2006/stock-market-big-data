"""Generate sample historical datasets for offline demonstration using standard Python library."""

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

TICKERS_CONFIG = [
    ("RELIANCE.NS", "RELIANCE_NS_sample.csv", 2950.0, 0.017, 101),
    ("TCS.NS", "TCS_NS_sample.csv", 3850.0, 0.015, 102),
    ("INFY.NS", "INFY_NS_sample.csv", 1620.0, 0.019, 103),
    ("HDFCBANK.NS", "HDFCBANK_NS_sample.csv", 1660.0, 0.016, 104),
    ("ICICIBANK.NS", "ICICIBANK_NS_sample.csv", 1120.0, 0.018, 105),
    ("SBIN.NS", "SBIN_NS_sample.csv", 810.0, 0.021, 106),
    ("ITC.NS", "ITC_NS_sample.csv", 480.0, 0.012, 107),
    ("LT.NS", "LT_NS_sample.csv", 3650.0, 0.017, 108),
    ("TATAMOTORS.NS", "TATAMOTORS_NS_sample.csv", 980.0, 0.023, 109),
    ("BHARTIARTL.NS", "BHARTIARTL_NS_sample.csv", 1320.0, 0.016, 110)
]


def generate_sample_csv(symbol: str, filename: str, base_price: float, daily_vol: float, seed: int, days: int = 300):
    random.seed(seed)
    end_date = datetime(2026, 10, 9)  # Recent business day
    
    # Generate business dates (Mon-Fri)
    dates = []
    curr = end_date
    while len(dates) < days:
        if curr.weekday() < 5:  # Monday to Friday
            dates.append(curr)
        curr -= timedelta(days=1)
    dates.reverse()

    file_path = SAMPLE_DIR / filename
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"])

        current_price = base_price
        for d in dates:
            # Daily return shock using Box-Muller normal approximation
            u1 = max(random.random(), 1e-9)
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            
            daily_return = (0.0004) + daily_vol * z
            close_price = max(current_price * (1.0 + daily_return), 10.0)
            
            # Intraday variations
            spread = daily_vol * random.uniform(0.3, 0.8)
            open_price = current_price * (1.0 + daily_vol * random.uniform(-0.4, 0.4))
            high_price = max(open_price, close_price) * (1.0 + spread)
            low_price = min(open_price, close_price) * (1.0 - spread)
            
            # Volume in range 1M - 5M shares
            volume = int(random.lognormvariate(14.5, 0.4))

            writer.writerow([
                d.strftime("%Y-%m-%d"),
                round(open_price, 2),
                round(high_price, 2),
                round(low_price, 2),
                round(close_price, 2),
                round(close_price, 2),
                volume,
                symbol
            ])
            current_price = close_price


def main():
    for symbol, filename, base_p, vol, seed in TICKERS_CONFIG:
        generate_sample_csv(symbol, filename, base_p, vol, seed)
    print(f"Generated {len(TICKERS_CONFIG)} sample datasets in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
