"""Trains several regression models to predict monthly units, converts to bill, saves the best."""
import os, joblib, numpy as np, pandas as pd, matplotlib
os.makedirs("model", exist_ok=True); os.makedirs("outputs", exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from bill_calculator import FEATURES, units_to_bill

df = pd.read_csv("data/electricity_data.csv")
print(df.describe().T[["mean", "min", "max"]].head(6), "\n")

X, y = df[FEATURES], df["monthly_units"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.08, random_state=42),
}
results, best_name, best_r2, best_model, best_pred = [], None, -9, None, None
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    r2 = r2_score(y_te, p)
    results.append({"Model": name, "R2": round(r2, 4),
                    "MAE (units)": round(mean_absolute_error(y_te, p), 2),
                    "RMSE (units)": round(np.sqrt(mean_squared_error(y_te, p)), 2)})
    if r2 > best_r2:
        best_name, best_r2, best_model, best_pred = name, r2, m, p

res = pd.DataFrame(results)
print(res.to_string(index=False))
res.to_csv("outputs/model_comparison.csv", index=False)

# bill-level error
true_bill = y_te.apply(units_to_bill)
pred_bill = pd.Series(best_pred, index=y_te.index).apply(units_to_bill)
print(f"\nBest model: {best_name}")
print(f"Bill MAE: Rs {mean_absolute_error(true_bill, pred_bill):.2f}")

joblib.dump(best_model, "model/best_model.joblib")

# plots
plt.figure(figsize=(6, 5))
plt.scatter(y_te, best_pred, s=8, alpha=0.5)
lim = [0, max(y_te.max(), best_pred.max())]
plt.plot(lim, lim, "r--"); plt.xlabel("Actual units"); plt.ylabel("Predicted units")
plt.title(f"{best_name}: Actual vs Predicted"); plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=120); plt.close()

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values().tail(12)
    plt.figure(figsize=(7, 5)); imp.plot.barh(); plt.title("Top feature importances")
    plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=120); plt.close()
print("Saved model/best_model.joblib and plots in outputs/")
