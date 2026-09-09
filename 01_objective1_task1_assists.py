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
# Objective 1 - Task 1
# Focal point: Chance creation (Assists per 90 minutes)
# ============================================================

STANDARD_FILE = "data/fbref_2026_standard.csv"
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
# FBref uses a two-row header, so header=[0, 1] is used to read
# both header levels correctly. The FIFA match file is also loaded
# because it is needed to identify which teams reached the knockout stage.
standard = pd.read_csv(STANDARD_FILE, header=[0, 1])
matches = pd.read_csv(MATCH_FILE)

matches["team_a"] = matches["team_a"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})
matches["team_b"] = matches["team_b"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})

# ============================================================
# 2. Clean and prepare the data
# ============================================================
# FBref exports contain two header rows. The useful statistic name is
# usually stored in the second row, so the columns are simplified first.
# Duplicate column names are removed to avoid selecting the wrong field.
standard.columns = [
    col[1] if not str(col[1]).startswith("Unnamed") else col[0]
    for col in standard.columns
]
standard = standard.loc[:, ~standard.columns.duplicated()].copy()

required = ["Squad", "MP", "Min", "90s", "Ast"]
missing = [c for c in required if c not in standard.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Available: {standard.columns.tolist()}")

standard = standard[required].copy()

for col in ["MP", "Min", "90s", "Ast"]:
    standard[col] = pd.to_numeric(standard[col], errors="coerce")

standard["Squad"] = clean_team_names(standard["Squad"])
standard["Ast_per_90"] = standard["Ast"] / standard["90s"]
standard = standard.replace([np.inf, -np.inf], np.nan)
standard = standard.dropna(subset=["Squad", "Ast_per_90"]).copy()

# ============================================================
# 3. Define tournament progression groups
# ============================================================
# Teams appearing in the Round of 32 are classified as knockout-stage
# teams. All other World Cup teams are treated as group-stage eliminations.
# The checks below make sure that all 48 teams are classified correctly.
round32 = matches[matches["stage"].eq("Round of 32")]
knockout_teams = set(round32["team_a"].dropna()) | set(round32["team_b"].dropna())

if len(knockout_teams) != 32:
    raise ValueError(f"Expected 32 Round-of-32 teams, found {len(knockout_teams)}.")

unmatched = sorted(knockout_teams - set(standard["Squad"]))
if unmatched:
    raise ValueError("Unmatched knockout teams: " + ", ".join(unmatched))

standard["Tournament_Group"] = np.where(
    standard["Squad"].isin(knockout_teams),
    "Knockout Stage",
    "Group Stage Eliminated",
)

counts = standard["Tournament_Group"].value_counts()

if len(standard) != 48:
    raise ValueError(f"Expected 48 teams, found {len(standard)}.")
if counts.get("Knockout Stage", 0) != 32:
    raise ValueError("Expected 32 knockout-stage teams.")
if counts.get("Group Stage Eliminated", 0) != 16:
    raise ValueError("Expected 16 group-stage-eliminated teams.")

print("=" * 70)
print("TASK 1 - DATA VALIDATION")
print("=" * 70)
print("Total teams:", len(standard))
print("Round-of-32 teams:", len(knockout_teams))
print(counts)
print("All team names matched successfully.")

# ============================================================
# 4. Define the population and take a stratified random sample
# ============================================================
# The population contains all 48 World Cup teams. A 75% proportional
# stratified random sample is taken from each progression group so both
# groups remain represented in the same proportions as the population.
# random_state=42 makes the sampling reproducible.
population = standard.copy()
sample_parts = []

for _, group in population.groupby("Tournament_Group"):
    n_sample = round(len(group) * 0.75)
    sample_parts.append(group.sample(n=n_sample, random_state=42))

sample = pd.concat(sample_parts, ignore_index=True)

print("\n" + "=" * 70)
print("POPULATION AND SAMPLE")
print("=" * 70)
print("Population size:", len(population))
print("Sample size:", len(sample))
print(sample["Tournament_Group"].value_counts())

# ============================================================
# 5. Calculate descriptive statistics
# ============================================================
# These statistics summarise the centre, spread and range of assists per 90
# for the sampled teams before carrying out inferential analysis.
x = sample["Ast_per_90"]
q1 = x.quantile(0.25)
q3 = x.quantile(0.75)

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS - ASSISTS PER 90")
print("=" * 70)
print(f"n: {len(x)}")
print(f"Mean: {x.mean():.3f}")
print(f"Median: {x.median():.3f}")
print(f"Standard deviation: {x.std(ddof=1):.3f}")
print(f"Variance: {x.var(ddof=1):.3f}")
print(f"Minimum: {x.min():.3f}")
print(f"Q1: {q1:.3f}")
print(f"Q3: {q3:.3f}")
print(f"Maximum: {x.max():.3f}")
print(f"IQR: {(q3 - q1):.3f}")

# ============================================================
# 6. Calculate a 95% confidence interval
# ============================================================
# A t-based confidence interval is used because the population standard
# deviation is unknown and is estimated from the sample.
ci_low, ci_high = stats.t.interval(
    confidence=0.95,
    df=len(x) - 1,
    loc=x.mean(),
    scale=stats.sem(x),
)

print("\n" + "=" * 70)
print("95% CONFIDENCE INTERVAL")
print("=" * 70)
print(f"Sample mean: {x.mean():.3f}")
print(f"95% CI: ({ci_low:.3f}, {ci_high:.3f})")

# ============================================================
# 7. Perform Welch's independent two-sample t-test
# ============================================================
# Welch's test compares the mean assists per 90 of the two independent
# tournament groups. It is suitable because it does not require equal
# variances or equal sample sizes.
knockout = sample.loc[sample["Tournament_Group"].eq("Knockout Stage"), "Ast_per_90"]
eliminated = sample.loc[sample["Tournament_Group"].eq("Group Stage Eliminated"), "Ast_per_90"]

print("\n" + "=" * 70)
print("GROUP DESCRIPTIVE STATISTICS")
print("=" * 70)
print(f"Knockout Stage: n={len(knockout)}, mean={knockout.mean():.3f}, SD={knockout.std(ddof=1):.3f}")
print(f"Group Stage Eliminated: n={len(eliminated)}, mean={eliminated.mean():.3f}, SD={eliminated.std(ddof=1):.3f}")

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
    print("Conclusion: Mean assists per 90 differs significantly between the groups.")
else:
    print("Decision: Do not reject H0.")
    print("Conclusion: Insufficient evidence that mean assists per 90 differs between the groups.")

# ============================================================
# 8. Save the sample and create visualisations
# ============================================================
# The sampled data are saved for reproducibility. A histogram shows the
# overall distribution, while a boxplot compares the two progression groups.
sample.to_csv("data/task1_assists_sample.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(x, bins=8, edgecolor="black")
plt.xlabel("Assists per 90 Minutes")
plt.ylabel("Number of Teams")
plt.title("Distribution of Assists per 90 Minutes\nFIFA World Cup 2026 Sample")
plt.tight_layout()
plt.savefig("task1_assists_histogram.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([knockout, eliminated], tick_labels=["Knockout Stage", "Group Stage Eliminated"])
plt.ylabel("Assists per 90 Minutes")
plt.title("Assists per 90 by Tournament Progress")
plt.tight_layout()
plt.savefig("task1_assists_boxplot.png", dpi=300)
plt.show()