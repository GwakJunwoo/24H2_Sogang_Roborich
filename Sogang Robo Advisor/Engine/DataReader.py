import FinanceDataReader as fdr
import pandas as pd
import sqlite3
from datetime import datetime

"""
This module provides functionality for fetching financial data from both 
external data sources and local SQLite databases. It supports fetching close 
prices for a specified set of tickers and retrieving historical data from a 
predefined database.

Functions:
    fetch_close_prices():
        Fetches historical close prices for a list of tickers from an external data source.
    
    fetch_data_from_db(tickers, db_path):
        Retrieves historical price data for specific tickers from a local SQLite database.
"""


def fetch_close_prices():
    start_date = '2000-01-01'
    end_date = datetime.today().strftime('%y-%m-%d')

    file_path = 'Sogang Robo Advisor/invest_universe.csv'

    db_path = 'Sogang Robo Advisor/financial_data.db'

    tickers_data = pd.read_csv(file_path, encoding='cp949')

    tickers_data['종목 코드'] = tickers_data['종목 코드'].apply(lambda x: f"{int(x):06}")

    tickers = tickers_data['종목 코드']

    all_close_prices = pd.DataFrame()

    for ticker in tickers:
        try:
            df = fdr.DataReader(ticker, start=start_date, end=end_date)
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            close_prices = df['Close']
            close_prices.name = ticker
            all_close_prices = pd.concat([all_close_prices, close_prices], axis=1)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return all_close_prices


def fetch_data_from_db(tickers, db_path='Sogang Robo Advisor/financial_data.db'):
    try:
        conn = sqlite3.connect(db_path)

        columns = ', '.join([f'"{ticker}"' for ticker in tickers])
        query = f"SELECT Date, {columns} FROM db"

        df = pd.read_sql_query(query, conn, index_col='Date')
        conn.close()

        df.index = pd.to_datetime(df.index)
        return df

    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return pd.DataFrame()
