"""Creates a synthetic household electricity dataset (no public dataset needed)."""
import os, numpy as np, pandas as pd
os.makedirs("data", exist_ok=True)
from bill_calculator import APPLIANCES, appliance_units, units_to_bill

rng = np.random.default_rng(42)
N = 5000
rows = []
for _ in range(N):
    members = rng.integers(1, 9)
    area = int(rng.integers(300, 3500))
    temp = round(rng.uniform(8, 44), 1)
    row = {"members": members, "area_sqft": area, "avg_temp": temp}
    total = 0.0
    for name, (w, h, always) in APPLIANCES.items():
        if name == "Ceiling Fan":   cnt = rng.integers(1, 2 + area // 400)
        elif name == "LED Light":   cnt = rng.integers(2, 4 + area // 150)
        elif name == "Air Conditioner": cnt = rng.choice([0, 0, 1, 1, 2, 3])
        elif name == "Air Cooler":  cnt = rng.choice([0, 0, 1, 2])
        elif name == "Refrigerator": cnt = rng.choice([1, 1, 1, 2])
        elif name == "Television":  cnt = rng.integers(0, 4)
        elif name == "Washing Machine": cnt = rng.choice([0, 1, 1])
        elif name == "Geyser":      cnt = rng.choice([0, 0, 1, 1, 2])
        elif name == "Water Pump":  cnt = rng.choice([0, 1, 1])
        else:                       cnt = rng.integers(0, 4)
        if always:
            hrs = 24
        else:
            hrs = float(np.clip(rng.normal(h, h * 0.4 + 0.5), 0, 24))
        # weather effect: AC/cooler used more when hot, geyser more when cold
        if name in ("Air Conditioner", "Air Cooler"):
            hrs = hrs * (0 if temp < 22 else min(1.6, (temp - 20) / 12))
        if name == "Geyser":
            hrs = hrs * (1.8 if temp < 15 else 0.3 if temp > 28 else 1.0)
        hrs = round(min(hrs, 24), 1)
        row[f"{name}_count"], row[f"{name}_hours"] = int(cnt), hrs
        total += appliance_units(name, cnt, hrs)
    units = max(5, total * rng.normal(1.0, 0.06) + rng.normal(0, 8))  # meter noise / unknown loads
    row["monthly_units"] = round(units, 2)
    row["monthly_bill"] = units_to_bill(units)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/electricity_data.csv", index=False)
print(f"Saved data/electricity_data.csv  shape={df.shape}")
