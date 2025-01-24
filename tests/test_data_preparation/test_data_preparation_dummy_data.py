# proof_of_concept_data_preparation_prerequisites.py


import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta


def create_dummy_table():
    query = """
    CREATE TABLE IF NOT EXISTS test_base_data (
        date TEXT,
        asset TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        PRIMARY KEY (date, asset)
    );
    """
    
    db_path= r'C:\Users\BrandonBlase\Desktop\Database\test_balcones_training.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()  # Create a cursor object
        
        cursor.execute(query)  # Execute the CREATE TABLE query
        conn.commit()  # Commit the transaction to save changes
        print("Table created successfully or already exists.")
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        
    finally:
        conn.close()
        

def populate_dummy_table():
    # Generate dummy data
    data = {
        'date': [],
        'asset': [],
        'open': [],
        'high': [],
        'low': [],
        'close': []
    }
    
    # Start date for the dummy data
    start_date = datetime(2023, 1, 1)
    
    for i in range(1000):
        date = start_date + timedelta(days=i)
        open_price = round(random.uniform(1.0, 1.5), 5)
        high_price = round(open_price + random.uniform(0.01, 0.05), 5)
        low_price = round(open_price - random.uniform(0.01, 0.05), 5)
        close_price = round(random.uniform(low_price, high_price), 5)
        
        data['date'].append(date.strftime('%Y-%m-%d'))
        data['asset'].append('EURUSD')
        data['open'].append(open_price)
        data['high'].append(high_price)
        data['low'].append(low_price)
        data['close'].append(close_price)
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Database connection
    db_path = r'C:\Users\BrandonBlase\Desktop\Database\test_balcones_training.db'
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Insert data into the test table
        df.to_sql('test_base_data', conn, if_exists='append', index=False)
        print("Data inserted successfully into the test table.")
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        
    finally:
        conn.close()


def main():
    create_dummy_table()
    populate_dummy_table()


if __name__ == "__main__":
    main()
