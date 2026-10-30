"""Appliance-based electricity bill calculator (slab tariff).
Edit SLABS / FIXED_CHARGE / DUTY_PERCENT to match your state's current tariff."""

# name: (default watts, default hours/day, always_on)
APPLIANCES = {
    "Ceiling Fan":      (75, 10, False),
    "LED Light":        (9, 6, False),
    "Air Conditioner":  (1500, 6, False),
    "Air Cooler":       (200, 8, False),
    "Refrigerator":     (150, 24, True),   # compressor duty ~ 35% handled below
    "Television":       (100, 5, False),
    "Washing Machine":  (500, 1, False),
    "Geyser":           (2000, 1, False),
    "Water Pump":       (750, 1, False),
    "Computer/Laptop":  (80, 5, False),
}
FRIDGE_DUTY = 0.35

# (upper limit of slab in units, rate Rs/unit)  -- illustrative, editable
SLABS = [(100, 5.50), (150, 6.00), (300, 6.50), (float("inf"), 7.00)]
FIXED_CHARGE = 100.0      # Rs per month
DUTY_PERCENT = 5.0        # electricity duty %

FEATURES = (["members", "area_sqft", "avg_temp"] +
            [f"{a}_count" for a in APPLIANCES] + [f"{a}_hours" for a in APPLIANCES])

def appliance_units(name, count, hours, days=30):
    watts, _, always_on = APPLIANCES[name]
    duty = FRIDGE_DUTY if name == "Refrigerator" else 1.0
    return count * watts * hours * duty * days / 1000.0

def units_to_bill(units):
    """Slab-wise energy charge + fixed charge + duty."""
    energy, prev = 0.0, 0
    for limit, rate in SLABS:
        if units > prev:
            energy += (min(units, limit) - prev) * rate
        prev = limit
    subtotal = energy + FIXED_CHARGE
    return round(subtotal * (1 + DUTY_PERCENT / 100), 2)

def bill_from_appliances(items, days=30):
    """items: {name: (count, hours_per_day)} -> (total_units, bill, breakdown dict)"""
    breakdown = {n: appliance_units(n, c, h, days) for n, (c, h) in items.items()}
    units = sum(breakdown.values())
    return round(units, 2), units_to_bill(units), breakdown
