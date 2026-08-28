import mysql.connector

try:
 conn=mysql.connector.connect(host="localhost", user="root", password="", database="steel_defects")
 if conn.is_connected():
     db_info = conn.server_info
     print(f"Connected{db_info}")

     cursor = conn.cursor()
     cursor.execute("SELECT DATABASE();")
     record = cursor.fetchone()
     print(f"You are connected to the database: {record[0]}")


     cursor.close()
     conn.close()
     print("Connection closed.")

except mysql.connector.Error as e:
    print(f"Error while connecting to MySQL: {e}")
    