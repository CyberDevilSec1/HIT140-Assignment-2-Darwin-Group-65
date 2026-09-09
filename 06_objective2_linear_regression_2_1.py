# ============================================================
# HIT140 - FIFA World Cup 2026
# Objective 2 - Linear Regression 2.1: Goal Difference
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE = Path("outputs/objective2/objective2_1_dataset_104.csv")
OUT = Path("outputs/objective2")
FEATURES = ['team_a_prior_avg_goals_for', 'team_a_prior_avg_goals_against', 'team_a_prior_points_per_game', 'team_a_prior_clean_sheet_rate', 'team_b_prior_avg_goals_for', 'team_b_prior_avg_goals_against', 'team_b_prior_points_per_game', 'team_b_prior_clean_sheet_rate']
TARGET = "goal_difference_team_a"

df = pd.read_csv(FILE)
if len(df) != 104:
    raise ValueError("Expected exactly 104 rows.")
if len(FEATURES) != 8:
    raise ValueError("Exactly 8 explanatory variables are required.")
if df[FEATURES + [TARGET]].isna().any().any():
    raise ValueError("Missing model values detected.")

# Chronological 80/20 split: first 83 matches train, final 21 test.
# This respects the temporal nature of prediction.
split = int(np.floor(len(df) * 0.80))
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

model = LinearRegression()
model.fit(train[FEATURES], train[TARGET])
pred = model.predict(test[FEATURES])

mae = mean_absolute_error(test[TARGET], pred)
rmse = np.sqrt(mean_squared_error(test[TARGET], pred))
r2 = r2_score(test[TARGET], pred)

baseline = np.repeat(train[TARGET].mean(), len(test))
baseline_mae = mean_absolute_error(test[TARGET], baseline)
baseline_rmse = np.sqrt(mean_squared_error(test[TARGET], baseline))
baseline_r2 = r2_score(test[TARGET], baseline)

coef = pd.DataFrame({"variable": FEATURES, "coefficient": model.coef_})
coef["absolute_coefficient"] = coef["coefficient"].abs()
coef = coef.sort_values("absolute_coefficient", ascending=False)
coef.to_csv(OUT / "objective2_1_coefficients.csv", index=False)

result = test[["record_id", "date", "stage", "team_a", "team_b"] + [TARGET]].copy()
result["predicted"] = pred
result["residual"] = result[TARGET] - result["predicted"]
result.to_csv(OUT / "objective2_1_test_predictions.csv", index=False)

print("=" * 72)
print("LINEAR REGRESSION 2.1 - GOAL DIFFERENCE")
print("=" * 72)
print("Rows:", len(df))
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Explanatory variables:", len(FEATURES))
print(f"Intercept: {model.intercept_:.4f}")
print("\nCoefficients:")
print(coef[["variable", "coefficient"]].to_string(index=False))
print("\nTEST-SET PERFORMANCE")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 : {r2:.4f}")
print("\nTRAINING-MEAN BASELINE")
print(f"MAE : {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R^2 : {baseline_r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(test[TARGET], pred, alpha=0.75)
lo = min(test[TARGET].min(), pred.min())
hi = max(test[TARGET].max(), pred.max())
plt.plot([lo, hi], [lo, hi], linestyle="--")
plt.xlabel("Actual goal difference")
plt.ylabel("Predicted goal difference")
plt.title("Regression 2.1: Actual vs Predicted Goal Difference")
plt.tight_layout()
plt.savefig(OUT / "objective2_1_actual_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(pred, test[TARGET] - pred, alpha=0.75)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted goal difference")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Regression 2.1: Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "objective2_1_residuals.png", dpi=300)
plt.close()
