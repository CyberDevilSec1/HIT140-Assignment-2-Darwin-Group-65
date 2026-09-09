# HIT140-Assignment-2-Darwin-Group-65
HIT140 Foundations of Data Science - Assignment 2 - Darwin Group 65

## FIFA World Cup 2026 Data Analysis

**Unit:** HIT140 Foundations of Data Science  
**Assessment:** Assignment 2  
**Campus:** Darwin  
**Group:** Darwin Group 65  

---
### Group Members

| Student ID | Student Name |
|---|---|
| S397291 | Muhammad Kumail Abbas |
| S397541 | Roman Bisural |
| S397448 | Suriya Sankar |
| S398777 | Khushi | 

## 1. Project Overview

This repository contains the Python code and datasets used for our HIT140 Foundations of Data Science Assignment 2.

The project analyses FIFA World Cup 2026 data using descriptive statistics, confidence intervals, hypothesis testing, data visualisation, and linear regression.

The assessment contains two main objectives.

### Objective 1 – Statistical Analysis

Objective 1 investigates differences between teams that progressed to the knockout stage and teams that were eliminated during the group stage.

Four football performance measures are analysed:

1. Assists per 90 minutes
2. Shots on Target Percentage (SoT%)
3. Goalkeeper Save Percentage (Save%)
4. Fouls per 90 minutes

The population contains 48 FIFA World Cup 2026 teams.

A 75% stratified sample is used, giving:

- Total sample: 36 teams
- Knockout-stage teams: 24
- Group-stage eliminated teams: 12

The Objective 1 analysis includes:

- Descriptive statistics
- 95% confidence intervals
- Comparison of group means
- Welch's independent two-sample t-tests
- Data visualisations

A fixed random state is used where sampling is performed to make the analysis reproducible.

### Objective 2 – Linear Regression

Objective 2 contains two linear regression models.

#### Linear Regression 2.1 – Match Goal Difference

The first model predicts the goal difference between two opposing teams in a FIFA World Cup 2026 match.

The dataset contains:

- 104 rows
- One row per match
- Exactly 8 explanatory variables
- Goal difference as the response variable

The explanatory variables are:

1. Team A prior average goals scored
2. Team A prior average goals conceded
3. Team A prior points per game
4. Team A prior clean-sheet rate
5. Team B prior average goals scored
6. Team B prior average goals conceded
7. Team B prior points per game
8. Team B prior clean-sheet rate

#### Linear Regression 2.2 – Team Goals Scored

The second model predicts the number of goals scored by one team in a particular FIFA World Cup 2026 match.

The dataset contains:

- 208 rows
- One row for each team's perspective in each match
- 104 unique matches
- Exactly 8 explanatory variables
- Goals scored as the response variable

The explanatory variables are:

1. Team prior average goals scored
2. Team prior average goals conceded
3. Team prior points per game
4. Team prior clean-sheet rate
5. Opponent prior average goals scored
6. Opponent prior average goals conceded
7. Opponent prior points per game
8. Opponent prior clean-sheet rate

### Preventing Data Leakage

All explanatory variables used for Objective 2 are calculated using information available before the match being predicted.

The statistics for a particular match are calculated from matches that occurred earlier in the tournament. The current match result is added to a team's history only after the pre-match explanatory variables for both teams have been recorded.

This prevents information from the current match from being used to predict that same match.

For a team's first tournament match, no previous FIFA World Cup 2026 match information is available. Therefore, the prior tournament statistics are initialised to zero. This is a limitation of the tournament-only modelling approach.

---

## 2. Repository Structure

The repository is organised as follows:

```text
HIT140-Assignment-2-Darwin-Group-65/
│
├── 00_validate_project.py
├── 01_objective1_task1_assists.py
├── 02_objective1_task2_shooting_accuracy.py
├── 03_objective1_task3_goalkeeping.py
├── 04_objective1_task4_discipline.py
├── 05_objective2_prepare_datasets.py
├── 06_objective2_linear_regression_2_1.py
├── 07_objective2_linear_regression_2_2.py
├── README.md
│
├── data/
│   ├── fbref_2026_standard.csv
│   ├── fbref_2026_shooting.csv
│   ├── fbref_2026_goalkeeping.csv
│   ├── fbref_2026_playing_time.csv
│   ├── fbref_2026_misc.csv
│   ├── fifa_world_cup_2026_official_matches_104.csv
│   ├── fifa_world_cup_2026_team_match_208.csv
│   ├── task1_assists_sample.csv
│   ├── task2_shooting_sample.csv
│   ├── task3_goalkeeping_sample.csv
│   └── task4_discipline_sample.csv
│
└── outputs/
    └── objective2/
```

The `outputs/` directory is created/used by the analysis scripts for generated results where applicable.

---

## 3. Python Files

### `00_validate_project.py`

Validates the project environment and checks that the required datasets are available before the analysis is performed.

### `01_objective1_task1_assists.py`

Performs Objective 1 Task 1 analysis using assists per 90 minutes.

### `02_objective1_task2_shooting_accuracy.py`

Performs Objective 1 Task 2 analysis using Shots on Target Percentage (SoT%).

### `03_objective1_task3_goalkeeping.py`

Performs Objective 1 Task 3 analysis using goalkeeper Save Percentage (Save%).

### `04_objective1_task4_discipline.py`

Performs Objective 1 Task 4 analysis using fouls per 90 minutes.

### `05_objective2_prepare_datasets.py`

Prepares and validates the two datasets required for Objective 2.

It:

- Reads the 104 official match records
- Processes matches chronologically
- Calculates pre-match statistics
- Prevents current-match information leakage
- Creates the 104-row Regression 2.1 dataset
- Creates the 208-row Regression 2.2 dataset
- Checks the 208 goals-scored targets against the supplied team-match dataset
- Validates that both regression datasets contain exactly 8 explanatory variables

### `06_objective2_linear_regression_2_1.py`

Builds and evaluates Linear Regression 2.1 for predicting match goal difference.

The data is divided chronologically into:

- 83 training matches
- 21 testing matches

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

A training-mean baseline is also calculated for comparison.

### `07_objective2_linear_regression_2_2.py`

Builds and evaluates Linear Regression 2.2 for predicting goals scored by a team.

The split is performed by match rather than randomly splitting individual team rows. This ensures that the two team observations belonging to the same match remain in the same dataset partition.

The resulting split contains:

- 166 training rows (83 matches)
- 42 testing rows (21 matches)

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

A training-mean baseline is also included for comparison.

---

## 4. Data Files

### FBref Data

The following datasets contain team-level FIFA World Cup 2026 statistics used primarily for Objective 1:

- `fbref_2026_standard.csv`
- `fbref_2026_shooting.csv`
- `fbref_2026_goalkeeping.csv`
- `fbref_2026_playing_time.csv`
- `fbref_2026_misc.csv`

### FIFA World Cup Match Data

`fifa_world_cup_2026_official_matches_104.csv`

Contains the 104 FIFA World Cup 2026 matches used for the match-level regression analysis.

`fifa_world_cup_2026_team_match_208.csv`

Contains 208 team-match observations representing the two teams participating in each of the 104 matches.

### Objective 1 Sample Data

The following files contain the datasets used/generated for the four Objective 1 analyses:

- `task1_assists_sample.csv`
- `task2_shooting_sample.csv`
- `task3_goalkeeping_sample.csv`
- `task4_discipline_sample.csv`

---

## 5. Data Sources

Data for this project was collected from the sources specified for the assessment:

- FIFA official FIFA World Cup 2026 statistics and match information
- FBref
- The Stats Don't Lie

The CSV datasets used to reproduce the submitted analysis are included in the `data/` directory.

---

## 6. Software Requirements

The project was developed and tested using Python in a Conda environment.

### Environment Name

```text
fifa2026
```

### Main Python Libraries

The analysis uses:

- pandas
- NumPy
- SciPy
- Matplotlib
- scikit-learn

The project environment used the following versions during validation:

```text
pandas       3.0.5
numpy        2.5.1
scipy        1.18.0
matplotlib   3.11.0
```

`scikit-learn` is required for the Objective 2 linear regression models.

---

## 7. Creating the Python Environment

Anaconda or Miniconda can be used to create a separate environment for the project.

Open Anaconda Prompt and run:

```bash
conda create -n fifa2026 python
```

Activate the environment:

```bash
conda activate fifa2026
```

Install the required libraries:

```bash
pip install pandas numpy scipy matplotlib scikit-learn
```

Alternatively, the packages can be installed using Conda where available.

---

## 8. Running the Project

### Step 1 – Download the Repository

Download or clone the complete repository.

Using Git:

```bash
git clone <repository-url>
```

Then move into the project directory:

```bash
cd HIT140-Assignment-2-Darwin-Group-65
```

### Step 2 – Activate the Environment

```bash
conda activate fifa2026
```

### Step 3 – Validate the Project

Run:

```bash
python 00_validate_project.py
```

The validation script checks the required datasets and Python environment.

### Step 4 – Run Objective 1

Run the Objective 1 scripts in numerical order:

```bash
python 01_objective1_task1_assists.py
python 02_objective1_task2_shooting_accuracy.py
python 03_objective1_task3_goalkeeping.py
python 04_objective1_task4_discipline.py
```

### Step 5 – Prepare Objective 2 Datasets

Before running either regression model, run:

```bash
python 05_objective2_prepare_datasets.py
```

A successful run should confirm:

```text
OBJECTIVE 2 DATA VALIDATION SUCCESSFUL
Regression 2.1 rows: 104
Regression 2.1 explanatory variables: 8
Regression 2.2 rows: 208
Regression 2.2 explanatory variables: 8
208-row goals-scored targets: matched official team-match data
Current-match predictor leakage: none
```

### Step 6 – Run Linear Regression 2.1

```bash
python 06_objective2_linear_regression_2_1.py
```

This trains and evaluates the model for predicting match goal difference.

### Step 7 – Run Linear Regression 2.2

```bash
python 07_objective2_linear_regression_2_2.py
```

This trains and evaluates the model for predicting the number of goals scored by a team.

---

## 9. Objective 2 Dataset Validation

The preparation script performs checks before the regression models are executed.

The validated dataset structure is:

| Regression | Observations | Explanatory Variables | Response |
|---|---:|---:|---|
| 2.1 | 104 | 8 | Goal difference |
| 2.2 | 208 | 8 | Goals scored |

For Regression 2.2, the goals-scored targets generated during dataset preparation are cross-checked against the supplied 208-row team-match dataset.

---

## 10. Model Evaluation

Both regression models are evaluated on held-out chronological test data.

The following evaluation metrics are reported:

### Mean Absolute Error (MAE)

MAE represents the average absolute difference between the actual and predicted values.

### Root Mean Squared Error (RMSE)

RMSE measures prediction error while giving greater weight to larger errors.

### R-squared (R²)

R² measures how much of the variation in the response variable is explained by the regression model on the evaluated data.

A training-mean baseline is also calculated so that the regression models can be compared with a simple reference prediction.

---

## 11. Reproducibility

To reproduce the analysis:

1. Download the complete repository.
2. Keep the original directory structure.
3. Do not move the CSV files out of the `data/` directory.
4. Install the required Python libraries.
5. Run `00_validate_project.py`.
6. Run the Objective 1 scripts in numerical order.
7. Run `05_objective2_prepare_datasets.py`.
8. Run Regression 2.1 and Regression 2.2.

The numbered Python filenames indicate the intended execution order.

---

## 12. Important Notes

- The raw/source CSV files should not be manually modified before running the analysis.
- Data cleaning and processing are performed through the Python scripts.
- Objective 2 explanatory variables are constructed using information available before each match.
- Current-match information is not intentionally used as an explanatory variable for predicting that same match.
- The chronological split is used for Objective 2 rather than a standard random train/test split.
- For Regression 2.2, both team rows belonging to the same match are kept in the same train/test partition.
- The regression results should be interpreted as predictive associations and not as evidence of causation.

---

## 13. Academic Integrity and AI Usage

Any use of artificial intelligence tools during the assessment is declared separately in the AI Usage Declaration Form in accordance with the unit requirements and CDU academic integrity requirements.

---

## 15. Disclaimer

This repository was created for educational purposes as part of HIT140 Foundations of Data Science at Charles Darwin University.
