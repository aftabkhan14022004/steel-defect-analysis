# generating the fabricated data(random data)
import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from collections import Counter

np.random.seed(42)
random.seed(42)
#number of batches taking as 5000
N= 5000
START_DATE = datetime(2025,1,1,0,0,0)
END_DATE = datetime(2026,6,30,23,59,59)

batch_ids =[f"B{i:05d}" for i in range(N)]
print(len(batch_ids))
print(batch_ids[:5])
print(batch_ids[-5:])


timestamps =[]
for _ in range(N):
    random_days = random.randint(0,(END_DATE-START_DATE).days)
    random_hours = random.randint(0,23)
    random_minutes = random.randint(0,59)
    random_seconds = random.randint(0,59)

    ts = START_DATE + timedelta(
         days=random_days,
         hours=random_hours,
         minutes=random_minutes,
         seconds=random_seconds
    )
    timestamps.append(ts)

max_days = (END_DATE-START_DATE).days
print(max_days)
random_days = random.randint(0,max_days)
print(len(timestamps))
print(timestamps[:5])
print(timestamps[-5:])

"""assigning the shift timings """
def assign_shifts(ts):
    h=ts.hour
    if 6 <= h <= 14:
        return 'morning'
    elif 14 <= h <= 22:
        return 'evening'
    else:
        return 'night'
shifts =[assign_shifts(ts) for ts in timestamps]

print(" first 5 batches (shift check)")
for i in range(5):
    ts = timestamps[i]
    print(f" batch {batch_ids[i]}| {ts}  |{ts.hour}| shift={shifts[i]}")

shifts_counts=Counter(shifts)
for shift,count in shifts_counts.items():
    print(f"{shift}: {count}")

furnace_temp = np.random.normal(1200,30,N).clip(1100,1300)
print(f"first 5 furnace temperatures:", furnace_temp[:5])
print(furnace_temp.min())
print(furnace_temp.max())
print(furnace_temp.mean())

rolling_speed = np.round(np.random.uniform(5.0,15.0,N),2)
print(f"rolling speed: {rolling_speed[:5]}")
print(rolling_speed.min())
print(rolling_speed.max())
print(rolling_speed.mean())


df_batches = pd.DataFrame({
    'batch_id': batch_ids,
    'timestamp': timestamps,
    'shift': shifts,
    'furnace_temp': furnace_temp,
    'rolling_speed': rolling_speed
})

print(df_batches.head(5))
print(df_batches.shape)
print(df_batches.dtypes)

timestamp_str =[]
for i,ts in enumerate(timestamps):
    if i % 7== 0:
        ts_str =ts.strftime("%Y-%m-%d %H:%M")
    else:
        ts_str =ts.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_str.append(ts_str)

df_batches['timestamp'] =timestamp_str

print("sample timestamps (messy)")

for i in range(14):
    print(f" Row{i}:{df_batches['timestamp'].iloc[i]}")


print(" Data types after mess", df_batches.dtypes)


num_missing = int(0.02 * N)
missing_indices = np.random.choice(N, size=num_missing, replace=False)
furnace_temp[missing_indices] = np.nan

df_batches['furnace_temp'] = furnace_temp


print("MISSING FURNACE temperature after 2 percetage error ")
missing_count = df_batches['furnace_temp'].isna().sum()
print(f"total missing,{missing_count}")
print("Rows with missing temp:")
print(df_batches[df_batches['furnace_temp'].isna()].head(5))

dup_indices=np.random.choice(N, size=5, replace=False)
for i in dup_indices:
    new_index = i
    while new_index == i:
        new_index = np.random.randint(0,N)
    batch_ids[i]=batch_ids[new_index]

df_batches['batch_id'] = batch_ids

duplicate_count=df_batches['batch_id'].value_counts()
dupes = duplicate_count[duplicate_count>1]
print(dupes)

dup_ids_list =dupes.index.tolist()
print(df_batches[df_batches['batch_id'].isin(dup_ids_list)].sort_values(('batch_id')))


df_batches = df_batches.drop_duplicates(subset='batch_id', keep='first')
print("\nShape after removing duplicates:", df_batches.shape)

clean_ids = df_batches['batch_id'].tolist()
n_clean = len(clean_ids)
operators =[f"OP{random.randint(1,15):02d}" for _ in range(n_clean)]
machine_ids =[random.choice(['M1','M2','M3']) for _ in range(n_clean)]

df_machines= pd.DataFrame({
    'batch_id': clean_ids,
    'operator': operators,
    'machine': machine_ids
})
df_machines.rename(columns={'operator': 'operator_id', 'machine': 'machine_id'}, inplace=True)
print(df_machines.head(5))
print("shape:",df_machines.shape)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="steel_defects"
)
cursor = conn.cursor()


sql_batch="""
     INSERT IGNORE INTO production_batches
         (batch_id, timestamp,shift,furnace_temp,rolling_speed)
     VALUES (%s, %s, %s, %s, %s)
     """

for _, row in df_batches.iterrows():
    temp = row['furnace_temp']
    if pd.isna(temp):
        temp = None
    cursor.execute(sql_batch, (
        row['batch_id'], row['timestamp'], row['shift'], temp, row['rolling_speed']))
conn.commit()
print("production_batches inserted.")


sql_machine = """
    INSERT IGNORE INTO machine_parameters
        (batch_id, operator_id, machine_id)
    VALUES (%s, %s, %s)
"""
for _, row in df_machines.iterrows():
    cursor.execute(sql_machine, (
        row['batch_id'], row['operator_id'], row['machine_id']))
conn.commit()
print("machine_parameters inserted.")

cursor.close()
conn.close()
print("All data inserted successfully.")