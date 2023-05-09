from sqlalchemy import create_engine, event
import pandas as pd
import time
from DataCleaning import DisplayInformation

server = 'AA\\SQLEXPRESS'
database = 'Santander'
driver = 'ODBC Driver 17 for SQL Server'

engine = create_engine('mssql+pyodbc://'+server+'/'+database+'?trusted_connection=yes&driver='+driver+'')

# Set up connection string to SQL Server using SSMS driver
def ExportCsvToSqlServer(file_name, df):
    try:
        with engine.connect():
            name_without_type = file_name.split('.')[0]
            df.to_sql(name= name_without_type, con=engine, if_exists='append', index=False, chunksize=10000)

        print("Data inserted successfully!")
    except Exception as e:
        print("Error:", e)
    finally:
        # Close SQL Server connection
        engine.dispose()

def LoadDfFromSqlServer(table_name):
    sql_query = f"SELECT * FROM {table_name}"
    chunks = []
    print("Loading data...")
    for chunk in pd.read_sql(sql_query, engine, chunksize=100000):
        chunks.append(chunk)
    print("Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    return df

time_start = time.time()
df = LoadDfFromSqlServer("smalltrain_ver2")
time_end = time.time()
print("Done in ", time_end - time_start, " seconds")
print("INFOS FROM SQLSERVER READ:\n")
DisplayInformation(df)