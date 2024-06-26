import db_utils as dbu
import pandas as pd

df = dbu.csv_to_df('loan_payments.csv')

print(df.head())

print(df.info())