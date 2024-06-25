import yaml
import pandas as pd
from sqlalchemy import create_engine

def cred_loader():
    with open('credentials.yaml') as f:
        cred_dict = yaml.load(f, Loader= yaml.FullLoader)
    return cred_dict

cred_dict =  cred_loader()

#TODO: Create a class RDSDatabaseConnector
class RDSDatabaseConnector:

    def __init__(self, cred_dict):
        self.cred_dict = cred_dict
    
    def db_connect(self):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = cred_dict['RDS_HOST']
        USER = 
        PASSWORD = 
        DATABASE = 
        PORT = 
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
