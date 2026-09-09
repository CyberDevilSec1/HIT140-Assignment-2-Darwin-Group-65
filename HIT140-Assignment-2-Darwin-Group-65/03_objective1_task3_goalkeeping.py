# ============================================================
# HIT140 - FIFA World Cup 2026
# Objective 1 - Group Analysis
# This script is part of our team's four-task statistical analysis.
# The code is written to keep the workflow reproducible and easy to audit.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# HIT140 - FIFA World Cup 2026
# Objective 1 - Task 3
# Focal point: Goalkeeping performance (Save Percentage)
# ============================================================

GOALKEEPING_FILE = "data/fbref_2026_goalkeeping.csv"
MATCH_FILE = "data/fifa_world_cup_2026_official_matches_104.csv"

def clean_team_names(series):
    return (
        series.astype(str)
        .str.replace(r"^[A-Za-z]{2,3}\s+", "", regex=True)
        .str.strip()
        .replace({
            "United States": "USA",
            "Iran": "IR Iran",
            "South Korea": "Korea Republic",
            "Ivory Coast": "Côte d'Ivoire",
            "Cape Verde": "Cabo Verde",
            "Turkey": "Türkiye",
            "Bosnia–Herz": "Bosnia and Herzegovina",
            "Bosnia-Herz": "Bosnia and Herzegovina",
            "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        })
    )

# ============================================================
# 1. Load the datasets
# ============================================================
# The FBref goalkeeping dataset uses a two-row header. The FIFA match file
# is required to identify the tournament progression group for each team.
goalkeeping = pd.read_csv(GOALKEEPING_FILE, header=[0, 1])
matches = pd.read_csv(MATCH_FILE)

matches["team_a"] = matches["team_a"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})
matches["team_b"] = matches["team_b"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})

# ============================================================
# 2. Clean and prepare the goalkeeping data
# ============================================================
# Simplify the FBref header structure and keep the required variables only.
# Save% is the focal variable because it represents goalkeeper shot-stopping
# performance at team level.
goalkeeping.columns = [
    col[1] if not str(col[1]).startswith("Unnamed") else col[0]
    for col in goalkeeping.columns
]
goalkeeping = goalkeeping.loc[:, ~goalkeeping.columns.duplicated()].copy()

required = ["Squad", "90s", "GA", "SoTA", "Saves", "Save%"]
missing = [c for c in required if c not in goalkeeping.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Available: {goalkeeping.columns.tolist()}")

goalkeeping = goalkeeping[required].copy()

for col in ["90s", "GA", "SoTA", "Saves", "Save%"]:
    goalkeeping[col] = pd.to_numeric(goalkeeping[col], errors="coerce")

goalkeeping["Squad"] = clean_team_names(goalkeeping["Squad"])
goalkeeping = goalkeeping.dropna(subset=["Squad", "Save%"]).copy()

# ============================================================
# 3. Define tournament progression groups
# ============================================================
# Teams that reached the Round of 32 form the knockout-stage group.
# The validation checks ensure that all 48 teams are matched and classified.
round32 = matches[matches["stage"].eq("Round of 32")]
knockout_teams = set(round32["team_a"].dropna()) | set(round32["team_b"].dropna())

if len(knockout_teams) != 32:
    raise ValueError(f"Expected 32 Round-of-32 teams, found {len(knockout_teams)}.")

unmatched = sorted(knockout_teams - set(goalkeeping["Squad"]))
if unmatched:
    raise ValueError("Unmatched knockout teams: " + ", ".join(unmatched))

goalkeeping["Tournament_Group"] = np.where(
    goalkeeping["Squad"].isin(knockout_teams),
    "Knockout Stage",
    "Group Stage Eliminated",
)

counts = goalkeeping["Tournament_Group"].value_counts()

if len(goalkeeping) != 48:
    raise ValueError(f"Expected 48 teams, found {len(goalkeeping)}.")
if counts.get("Knockout Stage", 0) != 32 or counts.get("Group Stage Eliminated", 0) != 16:
    raise ValueError(f"Unexpected group counts: {counts.to_dict()}")

print("=" * 70)
print("TASK 3 - DATA VALIDATION")
print("=" * 70)
print("Total teams:", len(goalkeeping))
print("Round-of-32 teams:", len(knockout_teams))
print(counts)
print("All team names matched successfully.")

# ============================================================
# 4. Define the population and take a stratified random sample
# ============================================================
# All 48 teams form the population. A 75% proportional stratified sample
# is selected from both groups using a fixed random seed for reproducibility.
population = goalkeeping.copy()
parts = []
for _, group in population.groupby("Tournament_Group"):
    parts.append(group.sample(n=round(len(group) * 0.75), random_state=42))
sample = pd.concat(parts, ignore_index=True)

print("\n" + "=" * 70)
print("POPULATION AND SAMPLE")
print("=" * 70)
print("Population size:", len(population))
print("Sample size:", len(sample))
print(sample["Tournament_Group"].value_counts())

# ============================================================
# 5. Calculate descriptive statistics
# ============================================================
# These statistics describe the typical goalkeeper save percentage and the
# amount of variation across the sampled teams.
x = sample["Save%"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS - SAVE %")
print("=" * 70)
print(f"n: {len(x)}")
print(f"Mean: {x.mean():.3f}%")
print(f"Median: {x.median():.3f}%")
print(f"Standard deviation: {x.std(ddof=1):.3f}")
print(f"Variance: {x.var(ddof=1):.3f}")
print(f"Minimum: {x.min():.3f}%")
print(f"Q1: {q1:.3f}%")
print(f"Q3: {q3:.3f}%")
print(f"Maximum: {x.max():.3f}%")
print(f"IQR: {(q3 - q1):.3f}")

# ============================================================
# 6. Calculate a 95% confidence interval
# ============================================================
# A t-based interval estimates the population mean save percentage from the
# sample while accounting for sampling uncertainty.
ci_low, ci_high = stats.t.interval(
    confidence=0.95,
    df=len(x) - 1,
    loc=x.mean(),
    scale=stats.sem(x),
)

print("\n" + "=" * 70)
print("95% CONFIDENCE INTERVAL")
print("=" * 70)
print(f"Sample mean: {x.mean():.3f}%")
print(f"95% CI: ({ci_low:.3f}%, {ci_high:.3f}%)")

# ============================================================
# 7. Perform Welch's independent two-sample t-test
# ============================================================
# The test examines whether mean save percentage differs between knockout
# teams and teams eliminated in the group stage.
knockout = sample.loc[sample["Tournament_Group"].eq("Knockout Stage"), "Save%"]
eliminated = sample.loc[sample["Tournament_Group"].eq("Group Stage Eliminated"), "Save%"]

print("\n" + "=" * 70)
print("GROUP DESCRIPTIVE STATISTICS")
print("=" * 70)
print(f"Knockout Stage: n={len(knockout)}, mean={knockout.mean():.3f}%, SD={knockout.std(ddof=1):.3f}")
print(f"Group Stage Eliminated: n={len(eliminated)}, mean={eliminated.mean():.3f}%, SD={eliminated.std(ddof=1):.3f}")

t_stat, p_value = stats.ttest_ind(knockout, eliminated, equal_var=False)

print("\n" + "=" * 70)
print("WELCH TWO-SAMPLE T-TEST")
print("=" * 70)
print("H0: mu_knockout = mu_group_stage")
print("H1: mu_knockout != mu_group_stage")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print("alpha = 0.05")

if p_value < 0.05:
    print("Decision: Reject H0.")
    print("Conclusion: Mean goalkeeper save percentage differs significantly between the groups.")
else:
    print("Decision: Do not reject H0.")
    print("Conclusion: Insufficient evidence that mean goalkeeper save percentage differs between the groups.")

# ============================================================
# 8. Save the sample and create visualisations
# ============================================================
# The exact sample is saved for reproducibility, followed by a histogram and
# boxplot to support the numerical analysis visually.
sample.to_csv("data/task3_goalkeeping_sample.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(x, bins=8, edgecolor="black")
plt.xlabel("Goalkeeper Save Percentage (%)")
plt.ylabel("Number of Teams")
plt.title("Distribution of Goalkeeper Save Percentage\nFIFA World Cup 2026 Sample")
plt.tight_layout()
plt.savefig("task3_save_percentage_histogram.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([knockout, eliminated], tick_labels=["Knockout Stage", "Group Stage Eliminated"])
plt.ylabel("Goalkeeper Save Percentage (%)")
plt.title("Goalkeeper Save Percentage by Tournament Progress")
plt.tight_layout()
plt.savefig("task3_save_percentage_boxplot.png", dpi=300)
plt.show()
