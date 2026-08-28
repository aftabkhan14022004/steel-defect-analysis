import os
import random
import mysql.connector

random.seed(42)

conn=mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="steel_defects"
)
cursor = conn.cursor()

cursor.execute("SELECT batch_id FROM production_batches")
batch_ids = [row[0] for row in cursor.fetchall()]
base_path=r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET"
splits =["train","validation"]

sql_insert = """
    INSERT INTO defect_inspections (batch_id, defect_type, defect_count)
    VALUES (%s, %s, %s)
"""
total_inserted=0
for split in splits:
    img_dir = os.path.join(base_path,split,"images")
    class_folders = os.listdir(img_dir)

    for class_name in class_folders:
        class_path = os.path.join(img_dir, class_name)
        if not(os.path.isdir(class_path)):
            continue
        files=[f for f in os.listdir(class_path) if f.endswith(".jpg")]
        for fname in files:
            defect_type=class_name
            batch_id=random.choice(batch_ids)
            defect_count=random.randint(1,5)
            cursor.execute(sql_insert,(batch_id,defect_type,defect_count))
            total_inserted+=1

conn.commit()
cursor.close()
conn.close()

print(f"Successfully inserted {total_inserted} rows into defect_inspections.")
