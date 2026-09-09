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
# Objective 1 - Task 2
# Focal point: Shooting accuracy (Shots on Target Percentage)
# ============================================================

SHOOTING_FILE = "data/fbref_2026_shooting.csv"
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
# The FBref shooting file uses a two-level header. The FIFA match file is
# used later to separate knockout-stage teams from group-stage eliminations.
shooting = pd.read_csv(SHOOTING_FILE, header=[0, 1])
matches = pd.read_csv(MATCH_FILE)

matches["team_a"] = matches["team_a"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})
matches["team_b"] = matches["team_b"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})

# ============================================================
# 2. Clean and prepare the shooting data
# ============================================================
# Simplify the two-level FBref header and retain one copy of duplicated
# column names. The analysis focuses on SoT%, which measures shooting accuracy.
shooting.columns = [
    col[1] if not str(col[1]).startswith("Unnamed") else col[0]
    for col in shooting.columns
]
shooting = shooting.loc[:, ~shooting.columns.duplicated()].copy()

required = ["Squad", "90s", "Sh", "SoT", "SoT%"]
missing = [c for c in required if c not in shooting.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Available: {shooting.columns.tolist()}")

shooting = shooting[required].copy()

for col in ["90s", "Sh", "SoT", "SoT%"]:
    shooting[col] = pd.to_numeric(shooting[col], errors="coerce")

shooting["Squad"] = clean_team_names(shooting["Squad"])
shooting = shooting.dropna(subset=["Squad", "SoT%"]).copy()

# ============================================================
# 3. Define tournament progression groups
# ============================================================
# Round-of-32 participants are treated as knockout-stage teams. Validation
# checks are included so incorrect team-name matching cannot silently affect
# the statistical results.
round32 = matches[matches["stage"].eq("Round of 32")]
knockout_teams = set(round32["team_a"].dropna()) | set(round32["team_b"].dropna())

if len(knockout_teams) != 32:
    raise ValueError(f"Expected 32 Round-of-32 teams, found {len(knockout_teams)}.")

unmatched = sorted(knockout_teams - set(shooting["Squad"]))
if unmatched:
    raise ValueError("Unmatched knockout teams: " + ", ".join(unmatched))

shooting["Tournament_Group"] = np.where(
    shooting["Squad"].isin(knockout_teams),
    "Knockout Stage",
    "Group Stage Eliminated",
)

counts = shooting["Tournament_Group"].value_counts()

if len(shooting) != 48:
    raise ValueError(f"Expected 48 teams, found {len(shooting)}.")
if counts.get("Knockout Stage", 0) != 32 or counts.get("Group Stage Eliminated", 0) != 16:
    raise ValueError(f"Unexpected group counts: {counts.to_dict()}")

print("=" * 70)
print("TASK 2 - DATA VALIDATION")
print("=" * 70)
print("Total teams:", len(shooting))
print("Round-of-32 teams:", len(knockout_teams))
print(counts)
print("All team names matched successfully.")

# ============================================================
# 4. Define the population and take a stratified random sample
# ============================================================
# The population contains all 48 teams. A reproducible 75% sample is taken
# separately from each progression group to preserve proportional representation.
population = shooting.copy()
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
# These values describe the centre, spread and range of shooting accuracy
# among the sampled teams.
x = sample["SoT%"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS - SHOTS ON TARGET %")
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
# The t-distribution is used because the population standard deviation is
# unknown and must be estimated from the sample.
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
# This test compares mean SoT% between the two independent progression groups
# without assuming equal population variances.
knockout = sample.loc[sample["Tournament_Group"].eq("Knockout Stage"), "SoT%"]
eliminated = sample.loc[sample["Tournament_Group"].eq("Group Stage Eliminated"), "SoT%"]

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
    print("Conclusion: Mean shots-on-target percentage differs significantly between the groups.")
else:
    print("Decision: Do not reject H0.")
    print("Conclusion: Insufficient evidence that mean shots-on-target percentage differs between the groups.")

# ============================================================
# 8. Save the sample and create visualisations
# ============================================================
# Saving the sample makes the analysis reproducible. The histogram summarises
# the overall distribution and the boxplot compares the two tournament groups.
sample.to_csv("data/task2_shooting_sample.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(x, bins=8, edgecolor="black")
plt.xlabel("Shots on Target (%)")
plt.ylabel("Number of Teams")
plt.title("Distribution of Shots on Target Percentage\nFIFA World Cup 2026 Sample")
plt.tight_layout()
plt.savefig("task2_sot_percentage_histogram.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([knockout, eliminated], tick_labels=["Knockout Stage", "Group Stage Eliminated"])
plt.ylabel("Shots on Target (%)")
plt.title("Shots on Target Percentage by Tournament Progress")
plt.tight_layout()
plt.savefig("task2_sot_percentage_boxplot.png", dpi=300)
plt.show()
