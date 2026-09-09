# ============================================================
# HIT140 - FIFA World Cup 2026
# Objective 2 - Prepare regression datasets
# ============================================================
# Regression 2.1: 104 actual matches, exactly 8 pre-match predictors
# Regression 2.2: 208 team-match rows, exactly 8 pre-match predictors
#
# Predictor design:
# - prior average goals scored
# - prior average goals conceded
# - prior points per game
# - prior clean-sheet rate
# for BOTH opposing teams.
#
# All four measures are calculated only from World Cup matches completed
# before the current match. For a team's first tournament match, its prior
# tournament values are 0 because it has no earlier World Cup 2026 match.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
OUT_DIR = Path("outputs/objective2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MATCH_FILE = DATA_DIR / "fifa_world_cup_2026_official_matches_104.csv"
TEAM_MATCH_FILE = DATA_DIR / "fifa_world_cup_2026_team_match_208.csv"

matches = pd.read_csv(MATCH_FILE)
official_208 = pd.read_csv(TEAM_MATCH_FILE)

if len(matches) != 104:
    raise ValueError(f"Expected 104 match rows, found {len(matches)}.")
if len(official_208) != 208:
    raise ValueError(f"Expected 208 team-match rows, found {len(official_208)}.")

matches["date"] = pd.to_datetime(matches["date"], dayfirst=True, errors="raise")
matches["_order"] = matches["record_id"].astype(str).str.extract(r"(\d+)")[0].astype(int)
matches = matches.sort_values(["date", "_order"]).reset_index(drop=True)

calculated_gd = matches["team_a_goals"] - matches["team_b_goals"]
if not calculated_gd.equals(matches["goal_difference_team_a"]):
    raise ValueError("Goal-difference target does not match team_a_goals - team_b_goals.")

history = {}

def prior_stats(team):
    h = history.get(team, {"matches": 0, "gf": 0, "ga": 0, "points": 0, "clean_sheets": 0})
    n = h["matches"]
    if n == 0:
        return {
            "prior_avg_goals_for": 0.0,
            "prior_avg_goals_against": 0.0,
            "prior_points_per_game": 0.0,
            "prior_clean_sheet_rate": 0.0,
        }
    return {
        "prior_avg_goals_for": h["gf"] / n,
        "prior_avg_goals_against": h["ga"] / n,
        "prior_points_per_game": h["points"] / n,
        "prior_clean_sheet_rate": h["clean_sheets"] / n,
    }

def update_history(team, gf, ga):
    h = history.setdefault(team, {"matches": 0, "gf": 0, "ga": 0, "points": 0, "clean_sheets": 0})
    h["matches"] += 1
    h["gf"] += int(gf)
    h["ga"] += int(ga)
    if ga == 0:
        h["clean_sheets"] += 1
    if gf > ga:
        h["points"] += 3
    elif gf == ga:
        h["points"] += 1

rows_104 = []
rows_208 = []

for _, r in matches.iterrows():
    a = prior_stats(r["team_a"])
    b = prior_stats(r["team_b"])

    row_104 = {
        "record_id": r["record_id"],
        "date": r["date"].date().isoformat(),
        "stage": r["stage"],
        "team_a": r["team_a"],
        "team_b": r["team_b"],
    }
    row_104.update({f"team_a_{k}": v for k, v in a.items()})
    row_104.update({f"team_b_{k}": v for k, v in b.items()})
    row_104["goal_difference_team_a"] = r["goal_difference_team_a"]
    rows_104.append(row_104)

    rows_208.append({
        "match_id": r["record_id"], "date": r["date"].date().isoformat(),
        "stage": r["stage"], "team": r["team_a"], "opponent": r["team_b"],
        **{f"team_{k}": v for k, v in a.items()},
        **{f"opponent_{k}": v for k, v in b.items()},
        "goals_scored": r["team_a_goals"],
    })
    rows_208.append({
        "match_id": r["record_id"], "date": r["date"].date().isoformat(),
        "stage": r["stage"], "team": r["team_b"], "opponent": r["team_a"],
        **{f"team_{k}": v for k, v in b.items()},
        **{f"opponent_{k}": v for k, v in a.items()},
        "goals_scored": r["team_b_goals"],
    })

    # Update only AFTER the pre-match values for this match have been recorded.
    update_history(r["team_a"], r["team_a_goals"], r["team_b_goals"])
    update_history(r["team_b"], r["team_b_goals"], r["team_a_goals"])

d104 = pd.DataFrame(rows_104)
d208 = pd.DataFrame(rows_208)

F21 = [
    "team_a_prior_avg_goals_for", "team_a_prior_avg_goals_against",
    "team_a_prior_points_per_game", "team_a_prior_clean_sheet_rate",
    "team_b_prior_avg_goals_for", "team_b_prior_avg_goals_against",
    "team_b_prior_points_per_game", "team_b_prior_clean_sheet_rate",
]
F22 = [
    "team_prior_avg_goals_for", "team_prior_avg_goals_against",
    "team_prior_points_per_game", "team_prior_clean_sheet_rate",
    "opponent_prior_avg_goals_for", "opponent_prior_avg_goals_against",
    "opponent_prior_points_per_game", "opponent_prior_clean_sheet_rate",
]

assert len(d104) == 104 and len(F21) == 8
assert len(d208) == 208 and len(F22) == 8
assert not d104[F21].isna().any().any()
assert not d208[F22].isna().any().any()

# Validate 208 targets against the supplied official team-match file.
check = d208.merge(
    official_208[["match_id", "team", "opponent", "goals_scored"]],
    on=["match_id", "team", "opponent"], how="outer",
    suffixes=("_built", "_official"), indicator=True
)
if not check["_merge"].eq("both").all():
    raise ValueError("Constructed 208 rows do not align with the official team-match file.")
if not check["goals_scored_built"].eq(check["goals_scored_official"]).all():
    raise ValueError("Goals-scored target mismatch found.")

d104.to_csv(OUT_DIR / "objective2_1_dataset_104.csv", index=False)
d208.to_csv(OUT_DIR / "objective2_2_dataset_208.csv", index=False)

print("=" * 72)
print("OBJECTIVE 2 DATA VALIDATION SUCCESSFUL")
print("=" * 72)
print("Regression 2.1 rows:", len(d104))
print("Regression 2.1 explanatory variables:", len(F21))
print("Regression 2.2 rows:", len(d208))
print("Regression 2.2 explanatory variables:", len(F22))
print("208-row goals-scored targets: matched official team-match data")
print("Current-match predictor leakage: none")
