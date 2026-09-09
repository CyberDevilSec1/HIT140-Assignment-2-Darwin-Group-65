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
# Objective 1 - Task 4
# Focal point: Discipline / physical play (Fouls per 90)
# ============================================================

MISC_FILE = "data/fbref_2026_misc.csv"
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
# The FBref miscellaneous table contains fouls and minutes played. The FIFA
# match file is used to classify each team by tournament progression.
misc = pd.read_csv(MISC_FILE, header=[0, 1])
matches = pd.read_csv(MATCH_FILE)

matches["team_a"] = matches["team_a"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})
matches["team_b"] = matches["team_b"].replace({"CÃ´te d'Ivoire": "Côte d'Ivoire"})

# ============================================================
# 2. Clean and prepare the discipline data
# ============================================================
# The FBref header is simplified and only the variables required for this
# task are retained. Fouls per 90 is calculated so teams that played different
# numbers of matches can be compared fairly.
misc.columns = [
    col[1] if not str(col[1]).startswith("Unnamed") else col[0]
    for col in misc.columns
]
misc = misc.loc[:, ~misc.columns.duplicated()].copy()

required = ["Squad", "90s", "Fls"]
missing = [c for c in required if c not in misc.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Available: {misc.columns.tolist()}")

misc = misc[required].copy()

for col in ["90s", "Fls"]:
    misc[col] = pd.to_numeric(misc[col], errors="coerce")

misc["Squad"] = clean_team_names(misc["Squad"])
misc["Fouls_per_90"] = misc["Fls"] / misc["90s"]
misc = misc.replace([np.inf, -np.inf], np.nan)
misc = misc.dropna(subset=["Squad", "Fouls_per_90"]).copy()

# ============================================================
# 3. Define tournament progression groups
# ============================================================
# Round-of-32 participants are classified as knockout-stage teams. The
# validation checks protect the analysis from team-name or classification errors.
round32 = matches[matches["stage"].eq("Round of 32")]
knockout_teams = set(round32["team_a"].dropna()) | set(round32["team_b"].dropna())

if len(knockout_teams) != 32:
    raise ValueError(f"Expected 32 Round-of-32 teams, found {len(knockout_teams)}.")

unmatched = sorted(knockout_teams - set(misc["Squad"]))
if unmatched:
    raise ValueError("Unmatched knockout teams: " + ", ".join(unmatched))

misc["Tournament_Group"] = np.where(
    misc["Squad"].isin(knockout_teams),
    "Knockout Stage",
    "Group Stage Eliminated",
)

counts = misc["Tournament_Group"].value_counts()

if len(misc) != 48:
    raise ValueError(f"Expected 48 teams, found {len(misc)}.")
if counts.get("Knockout Stage", 0) != 32 or counts.get("Group Stage Eliminated", 0) != 16:
    raise ValueError(f"Unexpected group counts: {counts.to_dict()}")

print("=" * 70)
print("TASK 4 - DATA VALIDATION")
print("=" * 70)
print("Total teams:", len(misc))
print("Round-of-32 teams:", len(knockout_teams))
print(counts)
print("All team names matched successfully.")

# ============================================================
# 4. Define the population and take a stratified random sample
# ============================================================
# The population is all 48 World Cup teams. A reproducible 75% proportional
# sample is drawn separately from the two progression groups.
population = misc.copy()
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
# These measures summarise the typical foul rate and the spread of fouls per
# 90 minutes across the sampled teams.
x = sample["Fouls_per_90"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS - FOULS PER 90")
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
# The t-distribution is used to estimate the population mean foul rate because
# the population standard deviation is not known.
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
# Welch's test compares mean fouls per 90 between the two independent groups
# and does not require the group variances to be equal.
knockout = sample.loc[sample["Tournament_Group"].eq("Knockout Stage"), "Fouls_per_90"]
eliminated = sample.loc[sample["Tournament_Group"].eq("Group Stage Eliminated"), "Fouls_per_90"]

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
    print("Conclusion: Mean fouls per 90 differs significantly between the groups.")
else:
    print("Decision: Do not reject H0.")
    print("Conclusion: Insufficient evidence that mean fouls per 90 differs between the groups.")

# ============================================================
# 8. Save the sample and create visualisations
# ============================================================
# The selected sample is saved so the analysis can be reproduced. The plots
# provide a visual summary of the overall distribution and group differences.
sample.to_csv("data/task4_discipline_sample.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(x, bins=8, edgecolor="black")
plt.xlabel("Fouls per 90 Minutes")
plt.ylabel("Number of Teams")
plt.title("Distribution of Fouls per 90 Minutes\nFIFA World Cup 2026 Sample")
plt.tight_layout()
plt.savefig("task4_fouls_per90_histogram.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([knockout, eliminated], tick_labels=["Knockout Stage", "Group Stage Eliminated"])
plt.ylabel("Fouls per 90 Minutes")
plt.title("Fouls per 90 Minutes by Tournament Progress")
plt.tight_layout()
plt.savefig("task4_fouls_per90_boxplot.png", dpi=300)
plt.show()
