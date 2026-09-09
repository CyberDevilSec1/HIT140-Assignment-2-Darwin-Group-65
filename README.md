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

# HIT140 Foundations of Data Science – Assignment 2

## Objective 1: FIFA World Cup 2026 Statistical Analysis

**Unit:** HIT140 Foundations of Data Science  
**Assessment:** Assignment 2  
**Campus:** Darwin  
**Group:** Darwin Group 65  

---

## 1. Objective 1 Overview

This repository contains the Python code and datasets used for Objective 1 of the HIT140 Foundations of Data Science assignment.

Objective 1 analyses FIFA World Cup 2026 team statistics and compares teams that progressed to the knockout stage with teams that were eliminated during the group stage.

The analysis consists of four tasks:

1. Assists per 90 minutes
2. Shots on Target Percentage (SoT%)
3. Goalkeeper Save Percentage (Save%)
4. Fouls per 90 minutes

For each task, descriptive and inferential statistical methods are used to investigate differences between the two groups.

---

## 2. Population and Sample

The population consists of all 48 teams participating in the FIFA World Cup 2026.

The teams are divided into:

- 32 knockout-stage teams
- 16 group-stage eliminated teams

A 75% stratified sample is used.

The final sample contains:

- 36 teams in total
- 24 knockout-stage teams
- 12 group-stage eliminated teams

A fixed random state (`random_state=42`) is used during sampling to make the selection reproducible.

---

## 3. Objective 1 Tasks

### Task 1 – Assists per 90 Minutes

**Python file:**

`01_objective1_task1_assists.py`

**Main dataset:**

`fbref_2026_standard.csv`

Task 1 investigates assists per 90 minutes.

The analysis includes:

- Sample size
- Mean
- Median
- Standard deviation
- Variance
- Minimum
- First quartile (Q1)
- Third quartile (Q3)
- Maximum
- Interquartile range (IQR)
- 95% confidence interval for the mean
- Group comparison
- Welch's independent two-sample t-test

The knockout-stage teams and group-stage eliminated teams are compared to determine whether their mean assists per 90 minutes are significantly different.

---

### Task 2 – Shots on Target Percentage

**Python file:**

`02_objective1_task2_shooting_accuracy.py`

**Main dataset:**

`fbref_2026_shooting.csv`

Task 2 investigates Shots on Target Percentage (SoT%).

The analysis includes:

- Descriptive statistics
- 95% confidence interval
- Comparison between knockout-stage and group-stage eliminated teams
- Welch's independent two-sample t-test
- Data visualisation

The purpose is to determine whether shooting accuracy differs significantly between the two groups.

---

### Task 3 – Goalkeeper Save Percentage

**Python file:**

`03_objective1_task3_goalkeeping.py`

**Main dataset:**

`fbref_2026_goalkeeping.csv`

Task 3 investigates goalkeeper Save Percentage (Save%).

The analysis includes:

- Descriptive statistics
- 95% confidence interval
- Comparison between knockout-stage and group-stage eliminated teams
- Welch's independent two-sample t-test
- Data visualisation

The purpose is to investigate whether teams progressing to the knockout stage have significantly different goalkeeper save percentages compared with teams eliminated during the group stage.

---

### Task 4 – Fouls per 90 Minutes

**Python file:**

`04_objective1_task4_discipline.py`

**Main dataset:**

`fbref_2026_misc.csv`

Task 4 investigates fouls committed per 90 minutes.

The analysis includes:

- Descriptive statistics
- 95% confidence interval
- Comparison between knockout-stage and group-stage eliminated teams
- Welch's independent two-sample t-test
- Data visualisation

The purpose is to determine whether there is a statistically significant difference in fouls per 90 minutes between the two groups.

---

## 4. Statistical Methods

The following statistical methods are used across Objective 1:

### Descriptive Statistics

Descriptive statistics are calculated to summarise the selected football performance measures.

These include:

- Mean
- Median
- Standard deviation
- Variance
- Minimum and maximum
- Quartiles
- Interquartile range

### 95% Confidence Interval

A 95% confidence interval is calculated for the sample mean to estimate a plausible range for the population mean.

### Welch's Two-Sample t-Test

Welch's independent two-sample t-test is used to compare the mean values between:

- Knockout-stage teams
- Group-stage eliminated teams

Welch's t-test is used because it does not require the two groups to have equal variances.

The hypotheses are:

**Null hypothesis (H0):**  
There is no difference between the mean values of the knockout-stage and group-stage eliminated teams.

**Alternative hypothesis (H1):**  
There is a difference between the mean values of the knockout-stage and group-stage eliminated teams.

A significance level of:

`α = 0.05`

is used.

If `p < 0.05`, the null hypothesis is rejected.

If `p >= 0.05`, there is insufficient evidence to reject the null hypothesis.

---

## 5. Data Sources

The analysis uses FIFA World Cup 2026 data collected from the sources specified for the assessment:

- FIFA official FIFA World Cup 2026 statistics
- FBref
- The Stats Don't Lie

The datasets required to reproduce the analysis are included in the `data/` folder.

---

## 6. Repository Structure

```text
HIT140-Assignment-2-Darwin-Group-65/
│
├── README.md
├── 00_validate_project.py
├── 01_objective1_task1_assists.py
├── 02_objective1_task2_shooting_accuracy.py
├── 03_objective1_task3_goalkeeping.py
├── 04_objective1_task4_discipline.py
│
├── data/
│   ├── fbref_2026_standard.csv
│   ├── fbref_2026_shooting.csv
│   ├── fbref_2026_goalkeeping.csv
│   ├── fbref_2026_misc.csv
│   ├── fbref_2026_playing_time.csv
│   ├── fifa_world_cup_2026_official_matches_104.csv
│   ├── task1_assists_sample.csv
│   ├── task2_shooting_sample.csv
│   ├── task3_goalkeeping_sample.csv
│   └── task4_discipline_sample.csv
│
└── outputs/
    ├── task1_assists_boxplot.png
    ├── task1_assists_histogram.png
    ├── task2_sot_percentage_boxplot.png
    ├── task2_sot_percentage_histogram.png
    ├── task3_save_percentage_boxplot.png
    ├── task3_save_percentage_histogram.png
    └── task4...png
```

---

## 7. Python Environment

The project was developed using Python in an Anaconda/Conda environment.

**Environment name:**

`fifa2026`

The main Python libraries used for Objective 1 are:

- pandas
- NumPy
- SciPy
- Matplotlib

The validated development environment used:

```text
pandas      3.0.5
numpy       2.5.1
scipy       1.18.0
matplotlib  3.11.0
```

---

## 8. Environment Setup

If using Anaconda or Miniconda, create a Python environment:

```bash
conda create -n fifa2026 python
```

Activate the environment:

```bash
conda activate fifa2026
```

Install the required libraries:

```bash
pip install pandas numpy scipy matplotlib
```

---

## 9. How to Run Objective 1

Download or clone the repository and keep the existing folder structure.

Open a terminal in the repository directory.

Activate the environment:

```bash
conda activate fifa2026
```

### Step 1 – Validate the Project

Run:

```bash
python 00_validate_project.py
```

This checks that the required Python libraries and datasets are available.

### Step 2 – Run Task 1

```bash
python 01_objective1_task1_assists.py
```

This performs the Assists per 90 analysis.

### Step 3 – Run Task 2

```bash
python 02_objective1_task2_shooting_accuracy.py
```

This performs the Shots on Target Percentage analysis.

### Step 4 – Run Task 3

```bash
python 03_objective1_task3_goalkeeping.py
```

This performs the goalkeeper Save Percentage analysis.

### Step 5 – Run Task 4

```bash
python 04_objective1_task4_discipline.py
```

This performs the Fouls per 90 analysis.

---

## 10. Objective 1 Summary Results

### Task 1 – Assists per 90

- Sample size: 36
- Overall mean: 0.933
- 95% CI: 0.708 to 1.157
- Knockout-stage mean: 1.163
- Group-stage eliminated mean: 0.472
- Welch's t-test p-value: 0.0004

The difference was statistically significant at the 5% significance level.

### Task 2 – Shots on Target Percentage

- Sample size: 36
- Overall mean: 32.739%
- 95% CI: 29.510% to 35.968%
- Knockout-stage mean: 35.471%
- Group-stage eliminated mean: 27.275%
- Welch's t-test p-value: 0.0245

The difference was statistically significant at the 5% significance level.

### Task 3 – Goalkeeper Save Percentage

- Sample size: 36
- Overall mean: 62.981%
- 95% CI: 58.687% to 67.274%
- Knockout-stage mean: 66.608%
- Group-stage eliminated mean: 55.725%
- Welch's t-test p-value: 0.0071

The difference was statistically significant at the 5% significance level.

### Task 4 – Fouls per 90

- Sample size: 36
- Overall mean: 11.569
- 95% CI: 10.766 to 12.373
- Knockout-stage mean: 11.173
- Group-stage eliminated mean: 12.361
- Welch's t-test p-value: 0.2062

The difference was not statistically significant at the 5% significance level.

---

## 11. Overall Objective 1 Finding

Three of the four measures showed statistically significant differences between knockout-stage and group-stage eliminated teams:

- Assists per 90
- Shots on Target Percentage
- Goalkeeper Save Percentage

Fouls per 90 did not show a statistically significant difference.

These findings show statistical associations between the selected performance measures and tournament progression. They should not be interpreted as evidence that these variables caused teams to progress to the knockout stage.

---

## 12. Reproducibility Notes

To reproduce Objective 1:

1. Keep all datasets inside the `data/` directory.
2. Install the required Python libraries.
3. Run `00_validate_project.py`.
4. Run Tasks 1–4 in numerical order.
5. Do not manually modify the source datasets before running the analysis.

The Python scripts perform the required data preparation and statistical analysis.

## 13. Academic Integrity and AI Usage

Any use of artificial intelligence tools associated with this assessment is declared separately using the AI Usage Declaration Form according to the unit requirements and CDU academic integrity requirements.

---

This repository was prepared for educational purposes as part of HIT140 Foundations of Data Science at Charles Darwin University.
