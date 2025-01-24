# create_db_load_rates.py


import os
import sqlite3 as sql
import pandas as pd


def create_database(db_path):
    """
    Create the 'v0' database with a table named 'hourly_exchange_rates'.
    """    
    # Connect to the database
    # If it does not exist, it will be created
    conn = sql.connect(db_path)
    
    try:
        cursor = conn.cursor()
        
        # SQL query to create the table
        query = """
        CREATE TABLE IF NOT EXISTS hourly_exchange_rates (
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            date TEXT NOT NULL,
            pair TEXT NOT NULL,
            PRIMARY KEY (date, pair)
        );
        """
        
        # Execute the query
        cursor.execute(query)
        conn.commit()
        
        print("Database and table created successfully.")
        
    except sql.Error as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Ensure the connection is always closed
        conn.close()
    
    
def load_rates(db_path):
    """
    Loop through the exchange rate source files and load them to the database.
    """
    source_dir = (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\Automation"
                  r"\Predict Rate\Features - Revised Wide SL and TP Target")
    
    for file in os.listdir(source_dir):
        if 'target' in file:
            # Load data
            file_path = os.path.join(source_dir, file)
            df = pd.read_csv(file_path, usecols=[
                'Open', 'High', 'Low', 'Close', 'Timestamp (UTC)'])
            
            # Create pair column based on file name
            pair = file[:6]
            df['pair'] = pair
            
            # Rename columns
            rename = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Timestamp (UTC)': 'date'}
            df = df.rename(columns=rename)
            
            # Load the data to the database
            try:
                conn = sql.connect(db_path)
                df.to_sql("hourly_exchange_rates", conn, if_exists="append", 
                          index=False)
                print(f"Successfully loaded '{pair}' exchange rates")
            finally:
                conn.close()


def main():
    """
    Create the iter0 database if it does not exist. Loop through the files
    containing exchange rates and load them to the hourly_exchange_rates
    table.
    """
    # Path to the SQLite database
    db_path = (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
               r"\projects\data\base.db")
    
    create_database(db_path)
    
    load_rates(db_path)
    

if __name__ == "__main__":
    main()