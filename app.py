import joblib, pandas as pd, streamlit as st
from bill_calculator import APPLIANCES, FEATURES, bill_from_appliances, units_to_bill, SLABS

st.set_page_config(page_title="Electricity Bill Predictor", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Bill Prediction")

@st.cache_resource
def load_model():
    return joblib.load("model/best_model.joblib")

st.sidebar.header("Household details")
members = st.sidebar.number_input("Family members", 1, 15, 4)
area = st.sidebar.number_input("House area (sq ft)", 200, 6000, 1200, step=50)
temp = st.sidebar.slider("Average temperature (°C)", 5, 48, 30)
days = st.sidebar.number_input("Days in billing cycle", 28, 31, 30)

st.subheader("Enter your appliances")
base = pd.DataFrame({
    "Appliance": list(APPLIANCES),
    "Count": [1] * len(APPLIANCES),
    "Watts": [v[0] for v in APPLIANCES.values()],
    "Hours/day": [float(v[1]) for v in APPLIANCES.values()],
})
edited = st.data_editor(base, use_container_width=True, hide_index=True,
                        disabled=["Appliance"], num_rows="fixed")

if st.button("Calculate Bill", type="primary"):
    # rule-based (uses the Watts the user typed)
    rows, usage = [], {}
    for _, r in edited.iterrows():
        duty = 0.35 if r["Appliance"] == "Refrigerator" else 1.0
        u = r["Count"] * r["Watts"] * r["Hours/day"] * duty * days / 1000
        rows.append({"Appliance": r["Appliance"], "Units (kWh)": round(u, 1)})
        usage[r["Appliance"]] = (r["Count"], r["Hours/day"])
    bd = pd.DataFrame(rows)
    units = bd["Units (kWh)"].sum()
    bill = units_to_bill(units)

    # ML prediction
    row = {"members": members, "area_sqft": area, "avg_temp": temp}
    for n, (c, h) in usage.items():
        row[f"{n}_count"], row[f"{n}_hours"] = c, h
    try:
        ml_units = float(load_model().predict(pd.DataFrame([row])[FEATURES])[0])
    except Exception:
        ml_units = None

    c1, c2, c3 = st.columns(3)
    c1.metric("Calculated units", f"{units:.0f} kWh")
    c2.metric("Calculated bill", f"₹ {bill:,.0f}")
    if ml_units is not None:
        c3.metric("ML predicted bill", f"₹ {units_to_bill(ml_units):,.0f}", f"{ml_units:.0f} kWh")
    else:
        c3.info("Run train_model.py to enable ML prediction")

    st.subheader("Where your units go")
    st.bar_chart(bd.set_index("Appliance"))
    top = bd.sort_values("Units (kWh)", ascending=False).iloc[0]
    st.info(f"💡 **{top['Appliance']}** uses the most power ({top['Units (kWh)']} kWh). "
            "Reducing its hours is the quickest way to cut your bill.")
