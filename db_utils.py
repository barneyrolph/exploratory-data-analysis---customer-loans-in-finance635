import pandas as pd
import re
from sqlalchemy import create_engine
import yaml


class RDSDatabaseConnector:
    '''
    This class is used to connect to a postgresql database using psycopg2 and read tables from that database.

    Attributes:
    cred_dict (dict) : A dictionary file with the configuration information for the specific database
    '''

    def __init__(self, cred_dict):
        self.cred_dict = cred_dict
        '''
         See help(RDSDatabaseConnector)
        '''

    def db_connect(self):
        '''
        This function connects to the database.
         
        Returns: an Engine object from sqlalchemy
        '''
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.cred_dict['RDS_HOST']
        USER = self.cred_dict['RDS_USER']
        PASSWORD = self.cred_dict['RDS_PASSWORD']
        DATABASE = self.cred_dict['RDS_DATABASE']
        PORT = self.cred_dict['RDS_PORT']
        return create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
    
    def db_pull_table(self, engine, table):
        '''
        This function reads a table from the database that has been connected to.

        Args:
        engine (sqlalchemy.engine.base.Engine): The engine to connect with the database
        table (str): the name of the table to be read

        Returns: 
        the table as a pandas df
        '''
        self.engine = engine
        self.table = table
        return pd.read_sql_table(self.table, self.engine)


class DataTransform:
    """
    A class to perform various data transformations on a pandas DataFrame.
    
    Attributes:
    ----------
    data_frame : pd.DataFrame
        The pandas DataFrame to be transformed.
    """

    def __init__(self, data_frame):
        """
        Constructs all the necessary attributes for the DataTransform object.

        Parameters:
        ----------
        data_frame : pd.DataFrame
            The pandas DataFrame to be transformed.
        """
        self.data_frame = data_frame

    def format_to_date(self, column):
        """
        Converts the specified column to datetime format.

        Parameters:
        ----------
        column : str
            The name of the column to be converted to datetime format.
        """
        try:
            self.data_frame[column] = pd.to_datetime(self.data_frame[column])
            print(f"Column '{column}' converted to datetime format.")
        except Exception as e:
            print(f"Error converting column '{column}' to datetime: {e}")

    def format_to_categorical(self, column):
        """
        Converts the specified column to categorical data type.

        Parameters:
        ----------
        column : str
            The name of the column to be converted to categorical type.
        """
        try:
            self.data_frame[column] = self.data_frame[column].astype('category')
            print(f"Column '{column}' converted to categorical type.")
        except Exception as e:
            print(f"Error converting column '{column}' to categorical: {e}")

    def format_to_time_period(self, column, freq='M'):
        """
        Converts the specified column to a period data type with the given frequency.

        Parameters:
        ----------
        column : str
            The name of the column to be converted to period type.
        freq : str, default 'M'
            The frequency for the period conversion (e.g., 'M' for month, 'Q' for quarter, 'A' for year).
        """
        try:
            self.data_frame[column] = pd.to_datetime(self.data_frame[column]).dt.to_period(freq)
            print(f"Column '{column}' converted to period format with frequency '{freq}'.")
        except Exception as e:
            print(f"Error converting column '{column}' to period format: {e}")

    def extract_numerical(self, column):
        """
        Extracts numerical values from the specified column and creates a new column with numerical values only.

        Parameters:
        ----------
        column : str
            The name of the column to extract numerical values from.
        """
        try:
            new_column_name = f"{column} numerical"
            self.data_frame[new_column_name] = self.data_frame[column].apply(lambda x: self._extract_number(x))
            print(f"Column '{new_column_name}' created with numerical values extracted from '{column}'.")
        except Exception as e:
            print(f"Error extracting numerical values from column '{column}': {e}")

    def _extract_number(self, value):
        """
        Helper function to extract numerical value from a string.

        Parameters:
        ----------
        value : str
            The string value to extract numerical part from.

        Returns:
        -------
        int
            The extracted numerical value.
        """
        match = re.search(r'\d+', str(value))
        return int(match.group()) if match else None


def cred_loader():
    '''
    This function reads the credentials.yaml file.

    Returns:
    A dictionary of the credentials for use with the 
    db_connect method.
    '''
    with open('credentials.yaml') as f:
        cred_dict = yaml.load(f, Loader= yaml.FullLoader)
    return cred_dict 

def save_csv(table, name = 0):
    '''
    This function saves a table from a database as a csv 
    file.  It initialises an instance of the 
    RDSDatabaseConnector class with the credloader
    function supplying the credentials dictionary
    and uses the db_pull_table method to read the
    table.  It will save the csv to the same 
    directory as the code.

    Args:
    table (str): the name of the table in the database.
    name (str): the desired name of the csv file which 
    will default to the name of the table.
    '''
    dbc1 = RDSDatabaseConnector(cred_loader())
    df = dbc1.db_pull_table(dbc1.db_connect(),table)
    df.to_csv(f'{name}.csv')

def csv_to_df(file_name):
    '''
    This function basically does the same as pandas read_csv.
    '''
    return pd.read_csv(file_name, index_col= 'Unnamed: 0')


