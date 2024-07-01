# Exploratory Data Analysis in Finance Project

---
by Barney Rolph

Welcome to my project designed to develop my skills as a Data Analysist.  The code I've created aims to help clean, plot and transform data ready for analysis.  I've also created ipynb files to document the process.


### Installation
- Clone the repository
- Create an environment using the json file for the correct dependencies.
- Import db_utils to use in you own projects or open the ipynb files to follow my journey.

### Usage and brief overview of classes
- RDSDatabaseConnector: This class is used to connect to a postgresql database using psycopg2 and read tables from that database.  The two methods establish the connection and pull a table into a pandas dataframe respectively.

- DataTransform: This class allows you to perform various transformations of a pandas dataframe.  The methods include functionality to: format variables as dates and categorical data; extract the numerical data from a variable; perform normalising tansformations to variables; select a normalising transformation and apply it to a variable based on the best reduction of skew.

- DataFrameInfo: This class contains methods to provide information about a pandas dataframe.  It also contains a method to plot boxplots to help identify outliers, along with a method that removed outliers.

- Plotter: This class provides methods for plotting correlation matrices, histograms and Q-Q plots.

- There are also functions in the script for loading a yaml file into a dictionary, and saving a table from the postgresql database to a csv.

### File structure
All the files are in the main directory, the only file not included is the credentials yaml that allows connection to the database.

### License Info
See license.txt.
