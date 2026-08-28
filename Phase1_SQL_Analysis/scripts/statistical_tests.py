import pandas as pd
import mysql.connector
from scipy.stats import chi2_contingency,ttest_ind,pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

conn= mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="steel_defects",

)

df_batches = pd.read_sql("SELECT * FROM production_batches",conn)
df_inspections = pd.read_sql("SELECT * FROM defect_inspections",conn)
conn.close()


merged=df_batches.merge(df_inspections,on="batch_id",how="inner")

# chi-square test
contingency = pd.crosstab(merged["defect_type"], merged["SHIFT"])
chi2, p, dof, expected = chi2_contingency(contingency)

print("Chi-square test: defect_type vs SHIFT")
print(f"Chi2 = {chi2:.2f}, p = {p:.4f}")
if p < 0.05:
    print("Significant: defect type depends on shift.")
else:
    print("Not significant.")

# t-test
batch_stats = (
    merged.groupby("batch_id")
    .agg(total_defects=("defect_count", "sum"), avg_temp=("furnace_temp", "mean"))
    .dropna()
)

median_temp = batch_stats["avg_temp"].median()
high_group = batch_stats[batch_stats["avg_temp"] >= median_temp]["total_defects"]
low_group = batch_stats[batch_stats["avg_temp"] < median_temp]["total_defects"]

t_stat, p_ttest = ttest_ind(high_group, low_group, equal_var=False)
print("\nT-test: High vs Low furnace temperature on defect rate")
print(f"t = {t_stat:.2f}, p = {p_ttest:.4f}")
if p_ttest < 0.05:
    print("Significant: temperature affects defect rate.")
else:
    print("Not significant.")

#pearson correlation
#rolling speed vs total defects per batch
batch_speed = (
    merged.groupby("batch_id")
    .agg(total_defects=("defect_count", "sum"), avg_speed=("rolling_speed", "mean"))
)

r, p_corr = pearsonr(batch_speed["avg_speed"], batch_speed["total_defects"])

print("\nPearson correlation: rolling speed vs total defects")
print(f"r = {r:.3f}, p = {p_corr:.4f}")
if p_corr < 0.05:
    print("Significant linear relationship.")
else:
    print("Not significant.")

sns.regplot(data=batch_speed, x="avg_speed", y="total_defects", scatter_kws={"alpha": 0.5})
plt.title("Correlation: Rolling Speed vs Total Defects per Batch")
plt.xlabel("Average Rolling Speed (m/min)")
plt.ylabel("Total Defects per Batch")
plt.tight_layout()
plt.savefig(r"C:\Users\aftab\Desktop\Steel_project\fig\correlation_speed_defects.png")
plt.close()

#heatmap
prob_table = pd.crosstab(merged['defect_type'], merged['SHIFT'], normalize='columns')
plt.figure(figsize=(8,6))
sns.heatmap(prob_table, annot=True, cmap='Blues', fmt='.2f')
plt.title('P(defect_type | shift)')
plt.tight_layout()
plt.savefig(r"C:\Users\aftab\Desktop\Steel_project\fig\probability_heatmap.png")
plt.show()