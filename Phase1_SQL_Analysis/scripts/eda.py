import pandas as pd
import os
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend

sns.set_style("whitegrid")

conn=mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="steel_defects"
    )

df_batches = pd.read_sql("SELECT * FROM production_batches", conn)
df_inspections = pd.read_sql("SELECT * FROM defect_inspections", conn)
df_machines = pd.read_sql("SELECT * FROM machine_parameters", conn)
conn.close()

merged= df_batches.merge(df_inspections, on="batch_id",how="left")
merged= merged.merge(df_machines, on="batch_id",how="left")

print(merged.shape)
print(merged.head(5))
print(merged.columns.tolist())

merged['ts_clean']=pd.to_datetime(merged['timestamp'],format="%Y-%m-%d %H:%M:%S", errors='coerce')
na_mask =merged['ts_clean'].isna()
print(f"rows with failed conversion (NaT): {na_mask.sum()}")

merged.loc[na_mask,'ts_clean']=pd.to_datetime(
    merged.loc[na_mask,'timestamp'],
    format="%Y-%m-%d %H:%M",
    errors='coerce'
)
print("Remaining NaT after fix:", merged['ts_clean'].isna().sum())
missing_temps = merged['furnace_temp'].isna().sum()
print(f"Missing furnace_temp: {missing_temps} ({missing_temps/len(merged)*100:.1f}%)")
os.makedirs(r"C:\Users\aftab\Desktop\Steel_project\fig", exist_ok=True)
order = merged['defect_type'].value_counts().index
plt.figure(figsize=(10,6))
order = merged['defect_type'].value_counts().index
sns.countplot(data=merged, x='defect_type', hue='defect_type', order=order,
              palette='viridis', legend=False)
plt.title('Distribution of Steel Surface Defect Types')
plt.xlabel('Defect Type')
plt.ylabel('Number of Inspections')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"C:\Users\aftab\Desktop\Steel_project\fig\defect_type_distribution.png", dpi=150)
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(data=merged, x='defect_type', hue='SHIFT', palette='Set2')
plt.title('Defect Type Distribution by Shift')
plt.xlabel('Defect Type')
plt.ylabel('Number of Inspections')
plt.xticks(rotation=45)
plt.legend(title='Shift')
plt.tight_layout()
plt.savefig(r"C:\Users\aftab\Desktop\Steel_project\fig\defect_type_by_shift.png", dpi=150)
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(data=merged, x='SHIFT', y='defect_count',hue='SHIFT', palette='Set2',legend=False)
plt.title('Defect Count per Inspection by Shift')
plt.xlabel('Shift')
plt.ylabel('Defect Count')
plt.tight_layout()
plt.savefig(r"C:\Users\aftab\Desktop\Steel_project\fig\defect_count_by_shift.png", dpi=150)
plt.show()


inspected = merged[merged['defect_type'].notna()]
batch_stats = inspected.groupby('batch_id').agg(
    total_defects=('defect_count', 'sum'),
    avg_speed=('rolling_speed', 'mean'),
    shift=('SHIFT', 'first')
).reset_index()

plt.figure(figsize=(8,6))
sns.regplot(data=batch_stats, x='avg_speed', y='total_defects',
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Total Defects vs. Average Rolling Speed per Batch')
plt.xlabel('Average Rolling Speed (m/min)')
plt.ylabel('Total Defects per Batch')
plt.tight_layout()
plt.savefig(r"C:\Users\aftab\Desktop\Steel_project\fig\defects_vs_speed.png", dpi=150)
plt.show()


median_temp = merged['furnace_temp'].median()

plt.figure(figsize=(8,5))
sns.histplot(merged['furnace_temp'].dropna(), bins=40, kde=True, color='steelblue')
plt.axvline(median_temp, color='red', linestyle='--', linewidth=2, label=f'Median: {median_temp:.1f}°C')
plt.title('Distribution of Furnace Temperatures')
plt.xlabel('Furnace Temperature (°C)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\aftab\Desktop\Steel_project\fig\furnace_temp_distribution.png", dpi=150)
plt.show()