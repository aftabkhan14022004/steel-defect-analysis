# Steel Surface Defect Analysis

A portfolio project that combines **real steel surface defect images** with **synthetic production data** to explore quality drivers using SQL, exploratory data analysis, and statistical testing.

---

## Key Features

- **Real + synthetic data** – 1 800 NEU defect images linked to 5 000 simulated production batches
- **Deliberate data messiness** – missing values, duplicate records, mixed timestamp formats (like real factory logs)
- **Advanced SQL** – window functions, CTEs, subqueries, date handling
- **Statistical testing** – chi‑square, t‑test, Pearson correlation, conditional probability heatmap
- **Reproducible pipeline** – every step scripted, from data generation to final visualizations

---

## Data Sources

- **NEU Steel Surface Defect Database (Kaggle)** – 6 defect types, 1 800 grayscale images
- **Synthetic production data** – furnace temperature (1 200 °C mean), rolling speed (5–15 m/min), 3‑shift pattern
  - 2% missing furnace temperatures
  - 5 duplicate batch IDs
  - Timestamps in two formats (with and without seconds)

All data is stored in a **MySQL** database with three tables: `production_batches`, `defect_inspections`, `machine_parameters`.

---

## Project Structure


```
 steel-defect-analysis/
  ├── data/raw/ # NEU-DET dataset (not pushed to GitHub)
  ├── scripts/
  │ ├── generate_data.py # Creates synthetic production data → MySQL
  │ ├── load_neu_data.py # Reads NEU images → defect_inspections table
  │ ├── eda.py # Data cleaning, merging, EDA plots
  │ └── statistical_tests.py # Hypothesis tests + probability heatmap
  ├── sql/
  │ ├── schema.sql # Database and table creation
  │ └── analysis_queries.sql # Advanced SQL queries (window functions, CTEs, subqueries)
  ├── fig/ # Saved plots
  ├── requirements.txt
  └── README.md

```
---

## Advanced SQL Queries

All queries are in `sql/analysis_queries.sql`:

1. **Rolling defect rate** – CTE cleans timestamps (`STR_TO_DATE`), then a window function computes a 5‑batch moving average of defect counts.
2. **Shift‑level aggregation** – CTE joins batches and inspections, then groups by shift to get total batches, total defects, and average defects per inspection.
3. **High‑defect batches vs. machine settings** – Subquery finds the overall average defects per batch, and the main query returns batches above that average with machine/operator details.

---

## Exploratory Data Analysis

- Merged all three tables and cleaned timestamps (two formats → single datetime column)
- Detected and removed duplicate batch IDs, noted missing furnace temperatures
- Created multiple visualizations:
  - Defect type distribution (balanced across six classes)
  - Defect counts by shift (countplot and boxplot)
  - Scatter plot of total defects vs. rolling speed (with regression line)
  - Furnace temperature histogram with median line

---

## Statistical Tests & Findings

All tests used only inspected batches. Results are printed by `statistical_tests.py`.

| Hypothesis | Test | Statistic | p‑value | Significant? |
|------------|------|-----------|---------|:---:|
| Defect type depends on shift | Chi‑square | χ² = 15.18 | 0.126 | No |
| Furnace temperature (high vs low median) affects defect rate | Welch’s t‑test | t = -1.09 | 0.276 | No |
| Rolling speed correlates with total defects | Pearson r | r = 0.049 | 0.058 | No |

**Interpretation:** None of the three factors showed a statistically significant effect in this dataset. This is a realistic outcome that highlights why rigorous hypothesis testing is essential before assuming process‑defect relationships in a manufacturing environment.

---

## Probability Heatmap

A heatmap of **P(defect_type | shift)** was generated to visualize conditional probabilities:

![Probability Heatmap](fig/probability_heatmap.png)

Even without statistical significance, the heatmap provides a qualitative view of defect distribution across shifts.

---

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/aftabkhan14022004/steel-defect-analysis.git
   cd steel-defect-analysis
   ```
   
2. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   ```

3. **Set up MySQL**
- Start your MySQL server (XAMPP, WAMP, etc.)

- Create a database named `steel_defects` 

- Run `sql/schema.sql` to create the three tables (you can do this in phpMyAdmin)

4. **Download the NEU dataset**
- Download from [Kaggle NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)  
- Place the `NEU-DET` folder inside `data/raw/` so that `data/raw/NEU-DET/train/images/` (and `valid/images/`) exists

5. **Populate the database**
    ```bash
   python scripts/generate_data.py
   python scripts/load_neu_data.py
   ```
6. **Run the analysis**
   ```bash
   python scripts/eda.py
   python scripts/statistical_tests.py
   ```


## Technologies Used
- **Python**: pandas, NumPy, matplotlib, seaborn, scipy, mysql-connector-python

- **MySQL**: CTEs, window functions, subqueries,  `STR_TO_DATE`, `CASE`

- **Data Engineering**: synthetic data generation, handling of missing/duplicate/inconsistent data

- **Statistics**: chi‑square test, Welch’s t‑test, Pearson correlation, conditional probability
  
---

## What I Learned

- How to **design a relational database schema** and enforce referential integrity with foreign keys
- Writing **advanced SQL queries** – CTEs, window functions, subqueries, and cleaning messy date strings with `STR_TO_DATE`
- **Handling real‑world data issues** – missing values, duplicate records, inconsistent formats – and documenting every cleaning decision
- **Exploratory data analysis** with pandas, matplotlib, and seaborn to uncover patterns before formal testing
- **Statistical hypothesis testing** – chi‑square, Welch’s t‑test, Pearson correlation – and interpreting p‑values honestly
- Building a **reproducible data pipeline** from raw data to final visualizations and documentation
- **Communicating results** clearly in a well‑structured README that tells the project’s story
  

---

## Author

**Aftab Khan**  
📧 aftabkhan14022004@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/aftab1402/)
🐙 [GitHub](https://github.com/aftabkhan14022004)

Feel free to reach out for collaboration, feedback, or just to connect!