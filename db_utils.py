import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy.stats import boxcox, yeojohnson, skew
from statsmodels.graphics.gofplots import qqplot
from sqlalchemy import create_engine
import yaml


class RDSDatabaseConnector:
    '''
    This class is used to connect to a postgresql 
    database using psycopg2 and read tables from 
    that database.

    Attributes:
    -----------
    cred_dict (dict) : A dictionary file with the 
    configuration information for the specific 
    database
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
        This function reads a table from the database
        that has been connected to.

        Arguments:
        -----
        engine (sqlalchemy.engine.base.Engine): The 
        engine to connect with the database 
        table (str): the name of the table to be read

        Returns:
        -------- 
        the table as a pandas df
        '''
        self.engine = engine
        self.table = table
        return pd.read_sql_table(self.table, self.engine)


class DataTransform:
    """
    A class to perform various data transformations on a
    pandas DataFrame.
    
    Attributes:
    ----------
    data_frame : pd.DataFrame
        The pandas DataFrame to be transformed.
    """

    def __init__(self, data_frame):
        """
        Constructs all the necessary attributes for the 
        DataTransform object.

        Parameters:
        ----------
        data_frame : pd.DataFrame
            The pandas DataFrame to be transformed.
        """
        self.data_frame = data_frame

    def format_to_date(self, column):
        """
        Converts the specified column to datetime 
        format.

        Parameters:
        ----------
        column : str
            The name of the column to be converted 
            to datetime format.
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

    def extract_numerical(self, column):
        """
        Extracts numerical values from the specified column 
        and creates a new column with numerical values only.

        Parameters:
        ----------
        column : str
            The name of the column to extract numerical 
            values from.
        """
        try:
            new_column_name = f"{column}_numerical"
            self.data_frame[new_column_name] = self.data_frame[column].apply(lambda x: self._extract_number(x)).astype(int)
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
        return int(match.group()) if match else 0
    
    def log_transform(self, column_name):
        """Applies a log transformation to the specified column and creates a new column."""
        if column_name in self.data_frame.columns:
            new_column_name = f"{column_name}_log"
            self.data_frame[new_column_name] = self.data_frame[column_name].apply(lambda x: np.log(x) if x > 0 else 0)
        else:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame.")
        return self.data_frame

    def box_cox_transform(self, column_name):
        """Applies a Box-Cox transformation to the specified column and creates a new column."""
        if column_name in self.data_frame.columns:
            new_column_name = f"{column_name}_box_cox"
            # Box-Cox requires strictly positive values
            positive_values = self.data_frame[column_name][self.data_frame[column_name] > 0]
            transformed_data, _ = boxcox(positive_values)
            self.data_frame[new_column_name] = np.nan  # Initialize with NaN
            self.data_frame.loc[self.data_frame[column_name] > 0, new_column_name] = transformed_data
        else:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame.")
        return self.data_frame

    def yeo_johnson_transform(self, column_name):
        """Applies a Yeo-Johnson transformation to the specified column and creates a new column."""
        if column_name in self.data_frame.columns:
            new_column_name = f"{column_name}_yeo_j"
            self.data_frame[new_column_name], _ = yeojohnson(self.data_frame[column_name])
        else:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame.")
        return self.data_frame

    def run_all_transformations_and_select_best(self, column_name):
        """Runs all transformations, calculates skewness, and drops the three transformed columns with skew values furthest from zero."""
        if column_name not in self.data_frame.columns:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame.")

        # Run all transformations
        self.log_transform(column_name)
        self.box_cox_transform(column_name)
        self.yeo_johnson_transform(column_name)

        # Calculate skewness
        skew_values = {
            column_name: skew(self.data_frame[column_name].dropna()),
            f"{column_name}_log": skew(self.data_frame[f"{column_name}_log"].dropna()),
            f"{column_name}_box_cox": skew(self.data_frame[f"{column_name}_box_cox"].dropna()),
            f"{column_name}_yeo_j": skew(self.data_frame[f"{column_name}_yeo_j"].dropna())
        }

        # Sort columns by skewness
        sorted_skew = sorted(skew_values.items(), key=lambda item: abs(item[1]))

        # Keep the original column and the transformed column with skew closest to zero
        columns_to_keep = [sorted_skew[0][0], sorted_skew[1][0]]
        columns_to_drop = [col for col in skew_values.keys() if col not in columns_to_keep]

        # Drop the columns with skew furthest from zero
        self.data_frame.drop(columns=columns_to_drop, inplace=True)

        return self.data_frame

class DataFrameInfo:
    """
    A class to extract and display useful information from a pandas DataFrame.
    
    Attributes:
    ----------
    data_frame : pd.DataFrame
        The pandas DataFrame to analyse.
    """

    def __init__(self, data_frame):
        """
        Constructs all the necessary attributes for the DataFrameInfo object.

        Parameters:
        ----------
        data_frame : pd.DataFrame
            The pandas DataFrame to be analysed.
        """
        self.data_frame = data_frame

    def describe_columns(self):
        """
        Describes all columns in the DataFrame to check their data types.
        """
        print("Column Descriptions:\n", self.data_frame.dtypes)

    def extract_statistics(self):
        """
        Extracts and prints statistical values: median, standard deviation, and mean for numerical columns,
        and mode for categorical/object columns.
        """
        print("Statistics:")

        numeric_cols = self.data_frame.select_dtypes(include=['number']).columns
        categorical_cols = self.data_frame.select_dtypes(include=['category', 'object']).columns

        if not numeric_cols.empty:
            print("\nNumerical Columns:")
            print("Median:\n", self.data_frame[numeric_cols].median())
            print("\nStandard Deviation:\n", self.data_frame[numeric_cols].std())
            print("\nMean:\n", self.data_frame[numeric_cols].mean())

        if not categorical_cols.empty:
            print("\nCategorical/Object Columns:")
            for col in categorical_cols:
                mode = self.data_frame[col].mode()
                if not mode.empty:
                    print(f"Mode of {col}: {mode.iloc[0]}")
                else:
                    print(f"Mode of {col}: No mode found")

    def count_distinct_values(self):
        """
        Counts distinct values in categorical columns and prints the counts.
        """
        categorical_cols = self.data_frame.select_dtypes(include=['category', 'object']).columns
        print("Distinct Values Count:")
        for col in categorical_cols:
            print(f"{col}: {self.data_frame[col].nunique()} distinct values")

    def print_shape(self):
        """
        Prints the shape of the DataFrame.
        """
        print("Shape of the DataFrame:", self.data_frame.shape)

    def null_values_count(self):
        """
        Generates a count and percentage count of NULL values in each column.
        """
        null_counts = self.data_frame.isnull().sum()
        null_percentages = (null_counts / len(self.data_frame)) * 100
        null_df = pd.DataFrame({'null_count': null_counts, 'null_percentage': null_percentages})
        print("NULL Values Count and Percentage:\n", null_df)


class Plotter:
    def __init__(self, dataframe):
        """
        Initialize the Plotter with a pandas dataframe.
        
        :param dataframe: pandas DataFrame
        """
        self.dataframe_number = dataframe.select_dtypes(include=['number'])
        self.dataframe = dataframe

    def plot_correlation_matrix(self, column=None):
        """
        Create and plot a correlation matrix for the dataframe.
        If a column is provided, plot the correlation of that column with all other columns.
        If no column is provided, plot the correlation matrix for all columns.
        
        :param column: str, column name of the dataframe
        """
        if column:
            if column in self.dataframe_number.columns:
                corr = self.dataframe_number.corr()[[column]].sort_values(by=column, ascending=False)
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Correlation Matrix of {column} with Other Columns')
                plt.show()
            else:
                print(f"Column '{column}' not found in dataframe.")
        else:
            corr = self.dataframe_number.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix')
            plt.show()

    def plot_distribution(self, variable=None):
        """
        Plot the distribution of a variable. If no variable is provided,
        plot the distribution of all variables.
        
        :param variable: str, column name of the dataframe
        """
        if variable:
            if variable in self.dataframe.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.dataframe[variable], kde=True)
                plt.title(f'Distribution of {variable}')
                plt.xlabel(variable)
                plt.ylabel('Frequency')
                plt.show()
            else:
                print(f"Variable '{variable}' not found in dataframe.")
        else:
            num_vars = self.dataframe.columns
            self.dataframe[num_vars].hist(bins=30, figsize=(20, 15), layout=(len(num_vars)//3+1, 3))
            plt.suptitle('Distribution of all variables')
            plt.show()
    def qqplot(self, column):
        '''
        Creates a Q-Q plots of a column.

        Parameters:
        -----------
         column : str
            The name of the column to plot. 
            values from.
        
        '''
        qq_plot = qqplot(self.dataframe[column] , scale=1 ,line='q', fit=True)
        plt.show()

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
