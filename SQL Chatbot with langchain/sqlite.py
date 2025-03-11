"""import sqlite3
##connect to sqlite
connection = sqlite3.connect("student.db")

## create a cursor object to insert record, create table
cursor = connection.cursor()

#create the table
table_info = """
"""create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHART(25), MARKS INT)"""
"""
cursor.execute(table_info)

#insert some more records
cursor.execute(''' Insert into STUDENT values('Madhu','Artificial intelligence','A',95)''')
cursor.execute(''' Insert into STUDENT values('Krish','Data science','B',86)''')
cursor.execute(''' Insert into STUDENT values('Nishitaa','MA','A',90)''')
cursor.execute(''' Insert into STUDENT values('Gayatri','Development studies','B',75)''')
cursor.execute(''' Insert into STUDENT values('Utkarsha','Marketing ','A',92)''')

##display all the records
print('The inserted records are')
data = cursor.execute(''' Select * from STUDENT''')
for row in data:
    print(row)

#commmit changes in the database
connection.commit()
connection.close()"""
import sqlite3
import pandas as pd

# Connect to SQLite
conn = sqlite3.connect("Employee.db")
cursor = conn.cursor()

# Read CSV using pandas
df = pd.read_csv("data.csv")

# Insert data into the STUDENT table
df.to_sql("EMPLOYEE", conn, if_exists="append", index=False)

print("Data imported successfully!")

# Commit and close connection
conn.commit()
conn.close()