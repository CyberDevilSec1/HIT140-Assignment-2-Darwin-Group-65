# ============================================================
# HIT140 - FIFA World Cup 2026
# Objective 2 - Linear Regression 2.2: Goals Scored
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE = Path("outputs/objective2/objective2_2_dataset_208.csv")
OUT = Path("outputs/objective2")
FEATURES = ['team_prior_avg_goals_for', 'team_prior_avg_goals_against', 'team_prior_points_per_game', 'team_prior_clean_sheet_rate', 'opponent_prior_avg_goals_for', 'opponent_prior_avg_goals_against', 'opponent_prior_points_per_game', 'opponent_prior_clean_sheet_rate']
TARGET = "goals_scored"

df = pd.read_csv(FILE)
if len(df) != 208:
    raise ValueError("Expected exactly 208 rows.")
if len(FEATURES) != 8:
    raise ValueError("Exactly 8 explanatory variables are required.")
if df[FEATURES + [TARGET]].isna().any().any():
    raise ValueError("Missing model values detected.")

# Split BY MATCH so the two team rows from one match never enter different sets.
match_ids = list(dict.fromkeys(df["match_id"].tolist()))
cut = int(np.floor(len(match_ids) * 0.80))
train_ids = set(match_ids[:cut])
test_ids = set(match_ids[cut:])
train = df[df["match_id"].isin(train_ids)].copy()
test = df[df["match_id"].isin(test_ids)].copy()

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
coef.to_csv(OUT / "objective2_2_coefficients.csv", index=False)

result = test[["match_id", "date", "stage", "team", "opponent"] + [TARGET]].copy()
result["predicted"] = pred
result["residual"] = result[TARGET] - result["predicted"]
result.to_csv(OUT / "objective2_2_test_predictions.csv", index=False)

print("=" * 72)
print("LINEAR REGRESSION 2.2 - TEAM GOALS SCORED")
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
plt.xlabel("Actual goals scored")
plt.ylabel("Predicted goals scored")
plt.title("Regression 2.2: Actual vs Predicted Goals")
plt.tight_layout()
plt.savefig(OUT / "objective2_2_actual_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(pred, test[TARGET] - pred, alpha=0.75)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted goals scored")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Regression 2.2: Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "objective2_2_residuals.png", dpi=300)
plt.close()
