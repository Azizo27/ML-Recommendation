from sqlalchemy import create_engine
import pandas as pd
from DataCleaning import LoadCsv
import os, pyodbc


# Set up connection string to SQL Server using SSMS driver
def ExportCsvToSqlServer(file_name):
    server = 'AA\\SQLEXPRESS'
    database = 'Santander'
    driver = 'ODBC Driver 17 for SQL Server'

    engine = create_engine('mssql+pyodbc://'+server+'/'+database+'?trusted_connection=yes&driver='+driver+'')

    try:
        with engine.connect():
            print("Connection successful!")
            df = LoadCsv(file_name, file_name) # The second parameter is used IF you want to save the data to a new csv file

            name_without_type = file_name.split('.')[0]
            
            df.to_sql(name= name_without_type, con=engine, if_exists='replace', index=False)

        print("Data inserted successfully!")
    except Exception as e:
        print("Error:", e)
    finally:
        # Close SQL Server connection
        engine.dispose()
