import yaml
import pandas as pd
from sqlalchemy import create_engine


class RDSDatabaseConnector:

    def __init__(self, cred_dict):
        self.cred_dict = cred_dict
    
    def db_connect(self):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.cred_dict['RDS_HOST']
        USER = self.cred_dict['RDS_USER']
        PASSWORD = self.cred_dict['RDS_PASSWORD']
        DATABASE = self.cred_dict['RDS_DATABASE']
        PORT = self.cred_dict['RDS_PORT']
        return create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
    
    def db_pull_table(self, engine, table):
        self.engine = engine
        self.table = table
        return pd.read_sql_table(self.table, self.engine)

def cred_loader():
    with open('credentials.yaml') as f:
        cred_dict = yaml.load(f, Loader= yaml.FullLoader)
    return cred_dict 

def save_csv(table):
    dbc1 = RDSDatabaseConnector(cred_loader())
    df = dbc1.db_pull_table(dbc1.db_connect(),table)
    df.to_csv(f'{table}.csv')

def csv_to_df(file_name):
    return pd.read_csv(file_name, index_col= 'Unnamed: 0')

