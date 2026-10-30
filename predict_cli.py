"""Quick command-line demo: bill from appliances (rule-based) vs ML prediction."""
import joblib, pandas as pd
from bill_calculator import APPLIANCES, FEATURES, bill_from_appliances, units_to_bill

# ---- edit this example household ----
home = {"members": 4, "area_sqft": 1200, "avg_temp": 35}
usage = {  # appliance: (count, hours/day)
    "Ceiling Fan": (4, 10), "LED Light": (8, 6), "Air Conditioner": (1, 6),
    "Air Cooler": (0, 0), "Refrigerator": (1, 24), "Television": (1, 5),
    "Washing Machine": (1, 1), "Geyser": (0, 0), "Water Pump": (1, 1),
    "Computer/Laptop": (1, 5),
}

units, bill, breakdown = bill_from_appliances(usage)
print("--- Appliance-wise units (30 days) ---")
for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
    print(f"{k:18s} {v:7.1f} units")
print(f"\nCalculated: {units} units -> Bill Rs {bill}")

row = dict(home)
for n, (c, h) in usage.items():
    row[f"{n}_count"], row[f"{n}_hours"] = c, h
model = joblib.load("model/best_model.joblib")
pu = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
print(f"ML predicted: {pu:.1f} units -> Bill Rs {units_to_bill(pu)}")
