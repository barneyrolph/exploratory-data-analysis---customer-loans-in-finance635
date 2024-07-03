import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import boxcox, yeojohnson, skew
from sqlalchemy import create_engine
from statsmodels.graphics.gofplots import qqplot


class RDSDatabaseConnector:
    """
    This class is used to connect to a PostgreSQL 
    database using psycopg2 and read tables from 
    that database.

    Attributes
    ----------
    cred_dict : dict
        A dictionary file with the configuration 
        information for the specific database.
    """

    def __init__(self, cred_dict):
        """
        Initialize with the credentials dictionary.
        """
        self.cred_dict = cred_dict

    def db_connect(self):
        """
        Connect to the database.
        
        Returns
        -------
        sqlalchemy.engine.base.Engine
            An Engine object from sqlalchemy.
        """
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.cred_dict['RDS_HOST']
        USER = self.cred_dict['RDS_USER']
        PASSWORD = self.cred_dict['RDS_PASSWORD']
        DATABASE = self.cred_dict['RDS_DATABASE']
        PORT = self.cred_dict['RDS_PORT']
        return create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
    
    def db_pull_table(self, engine, table):
        """
        Read a table from the database.

        Parameters
        ----------
        engine : sqlalchemy.engine.base.Engine
            The engine to connect with the database.
        table : str
            The name of the table to be read.

        Returns
        -------
        pd.DataFrame
            The table as a pandas DataFrame.
        """
        self.engine = engine
        self.table = table
        return pd.read_sql_table(self.table, self.engine)


class DataTransform:
    """
    A class to perform various data transformations on a pandas DataFrame.
    
    Attributes
    ----------
    data_frame : pd.DataFrame
        The pandas DataFrame to be transformed.
    """

    def __init__(self, data_frame):
        """
        Initialize with the DataFrame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            The pandas DataFrame to be transformed.
        """
        self.data_frame = data_frame

    def format_to_date(self, column):
        """
        Convert the specified column to datetime format.

        Parameters
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
        Convert the specified column to categorical data type.

        Parameters
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
        Extract numerical values from the specified column 
        and create a new column with numerical values only.

        Parameters
        ----------
        column : str
            The name of the column to extract numerical values from.
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

        Parameters
        ----------
        value : str
            The string value to extract numerical part from.

        Returns
        -------
        int
            The extracted numerical value.
        """
        match = re.search(r'\d+', str(value))
        return int(match.group()) if match else 0
    
    def log_transform(self, column_name):
        """
        Apply a log transformation to the specified column and create a new column.

        Parameters
        ----------
        column_name : str
            The name of the column to be log transformed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the new log-transformed column.
        """
        if column_name in self.data_frame.columns:
            new_column_name = f"{column_name}_log"
            self.data_frame[new_column_name] = self.data_frame[column_name].apply(lambda x: np.log(x) if x > 0 else 0)
        else:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame.")
        return self.data_frame

    def box_cox_transform(self, column_name):
        """
        Apply a Box-Cox transformation to the specified column and create a new column.

        Parameters
        ----------
        column_name : str
            The name of the column to be Box-Cox transformed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the new Box-Cox transformed column.
        """
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
        """
        Apply a Yeo-Johnson transformation to the specified column and create a new column.

        Parameters
        ----------
        column_name : str
            The name of the column to be Yeo-Johnson transformed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the new Yeo-Johnson transformed column.
        """
        if column_name in self.data_frame.columns:
            new_column_name = f"{column_name}_yeo_j"
            self.data_frame[new_column_name], _ = yeojohnson(self.data_frame[column_name])
        else:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame.")
        return self.data_frame

    def run_all_transformations_and_select_best(self, column_name):
        """
        Run all transformations, calculate skewness, and drop the three transformed columns with skew values furthest from zero.

        Parameters
        ----------
        column_name : str
            The name of the column to be transformed and assessed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the best transformed column kept.
        """
        if column_name not in self.data_frame.columns:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame.")

        # Run all transformations
        self.log_transform(column_name)
        self.box_cox_transform(column_name)
        self.yeo_johnson_transform(column_name)

        # Calculate skewness
        skew_values = {
            f"{column_name}_log": skew(self.data_frame[f"{column_name}_log"].dropna()),
            f"{column_name}_box_cox": skew(self.data_frame[f"{column_name}_box_cox"].dropna()),
            f"{column_name}_yeo_j": skew(self.data_frame[f"{column_name}_yeo_j"].dropna())
        }

        # Sort columns by skewness
        sorted_skew = sorted(skew_values.items(), key=lambda item: abs(item[1]))

        # Keep the original column and the transformed column with skew closest to zero
        columns_to_keep = [sorted_skew[0][0]]
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

    def detect_outliers(self):
        """
        Creates a box plot for each column in the data frame to help detect outliers.
        """
        num_columns = len(self.data_frame.columns)
        num_rows = math.ceil(num_columns / 3)
        
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for i, column in enumerate(self.data_frame.columns):
            if pd.api.types.is_numeric_dtype(self.data_frame[column]):
                self.data_frame.boxplot(column=column, ax=axes[i])
                axes[i].set_title(f'Box Plot of {column}')
            else:
                axes[i].set_visible(False)
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def remove_outliers(self, column_names):
        """
        Filters the data frame and returns only rows that don't contain outliers in any of the specified columns.
        An outlier in each column is a value that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.
        
        Parameters:
        ----------
        column_names : list of str
            The names of the columns to check for outliers.
        """
        initial_row_count = self.data_frame.shape[0]
        df_filtered = self.data_frame.copy()

        for column_name in column_names:
            if column_name not in self.data_frame.columns:
                raise KeyError(f"Column {column_name} does not exist in the DataFrame.")
            
            if not pd.api.types.is_numeric_dtype(self.data_frame[column_name]):
                raise TypeError(f"Column {column_name} is not numeric.")
            
            Q1 = df_filtered[column_name].quantile(0.25)
            Q3 = df_filtered[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_filtered = df_filtered[(df_filtered[column_name] >= lower_bound) & (df_filtered[column_name] <= upper_bound)]
        
        final_row_count = df_filtered.shape[0]
        rows_removed = initial_row_count - final_row_count
        print(f"Total rows removed: {rows_removed}")
        
        return df_filtered
    
    
class Plotter:
    def __init__(self, data_frame):
        """
        Initialize the Plotter with a pandas dataframe.
        
        :param dataframe: pandas DataFrame
        """
        self.data_frame_number = data_frame.select_dtypes(include=['number'])
        self.data_frame = data_frame

    def plot_correlation_matrix(self, column=None):
        """
        Create and plot a correlation matrix for the dataframe.
        If a column is provided, plot the correlation of that column with all other columns.
        If no column is provided, plot the correlation matrix for all columns.
        
        :param column: str, column name of the dataframe
        """
        if column:
            if column in self.data_frame_number.columns:
                corr = self.data_frame_number.corr()[[column]].sort_values(by=column, ascending=False)
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Correlation Matrix of {column} with Other Columns')
                plt.show()
            else:
                print(f"Column '{column}' not found in dataframe.")
        else:
            corr = self.data_frame_number.corr()
            num_cols = len(self.data_frame_number.columns)
            plt.figure(figsize=(num_cols * 0.5, num_cols * 0.5))  # Adjusts figsize based on number of columns
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
            if variable in self.data_frame.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data_frame[variable], kde=True)
                plt.title(f'Distribution of {variable}')
                plt.xlabel(variable)
                plt.ylabel('Frequency')
                plt.show()
            else:
                print(f"Variable '{variable}' not found in dataframe.")
        else:
            num_vars = self.data_frame.columns
            self.data_frame[num_vars].hist(bins=30, figsize=(20, 15), layout=(len(num_vars)//3+1, 3))
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
        qq_plot = qqplot(self.data_frame[column] , scale=1 ,line='q', fit=True)
        plt.show()
    
    def plot_histogram(self, column, bins=10):
        """
        Plot a histogram of the specified column.

        Parameters
        ----------
        column : str
            The name of the column to plot.
        bins : int, optional
            Number of bins for the histogram (default is 10).
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data_frame[column], bins=bins, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_bar(self, column):
        """
        Plot a bar chart of the specified column.

        Parameters
        ----------
        column : str
            The name of the column to plot.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.data_frame[column])
        plt.title(f'Bar Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

    def plot_scatter(self, x_column, y_column):
        """
        Plot a scatter plot for two specified columns.

        Parameters
        ----------
        x_column : str
            The name of the x-axis column.
        y_column : str
            The name of the y-axis column.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data_frame[x_column], y=self.data_frame[y_column])
        plt.title(f'Scatter Plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def plot_box(self, column):
        """
        Plot a box plot of the specified column.

        Parameters
        ----------
        column : str
            The name of the column to plot.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data_frame[column])
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        plt.show()

    def plot_pair(self):
        """
        Plot pairwise relationships in the DataFrame.
        """
        plt.figure(figsize=(10, 6))
        sns.pairplot(self.data_frame)
        plt.title('Pair Plot of DataFrame')
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

    Arguments:
    ----------
    table: str
        The name of the table in the database.
    name: str
        The desired name of the csv file which 
        will default to the name of the table.
    '''
    dbc1 = RDSDatabaseConnector(cred_loader())
    df = dbc1.db_pull_table(dbc1.db_connect(),table)
    df.to_csv(f'{name}.csv')

