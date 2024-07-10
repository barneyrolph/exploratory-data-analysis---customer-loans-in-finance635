# Exploratory Data Analysis in Finance Project

---
by Barney Rolph

Welcome to my project designed to develop my skills as a Data Analysist.  The code I've created aims to help clean, plot and transform data ready for analysis.  I've also created ipynb files to document the process.

### Installation
- Clone the repository
- Create an environment using the yaml file (eda_env.yaml) for the correct dependencies.
- Import db_utils to use in you own projects or open the ipynb files to follow my journey.

### Usage and brief overview of classes in db_utils
- RDSDatabaseConnector: This class is used to connect to a postgresql database using psycopg2 and read tables from that database.  The two methods establish the connection and pull a table into a pandas dataframe respectively.

- DataTransform: This class allows you to perform various transformations of a pandas dataframe.  The methods include functionality to: format variables as dates and categorical data; extract the numerical data from a variable; perform normalising tansformations to variables; select a normalising transformation and apply it to a variable based on the best reduction of skew.

- DataFrameInfo: This class contains methods to provide information about a pandas dataframe.  It also contains a method to plot boxplots to help identify outliers, along with a method that removed outliers.

- Plotter: This class provides methods for plotting correlation matrices, histograms and Q-Q plots.

- There are also functions in the script for loading a yaml file into a dictionary, and saving a table from the postgresql database to a csv.

### File structure
All the files are in the main directory, the only file not included is the credentials yaml that allows connection to the database.

### License Info
See license.txt.

### What I've learn
I've improved on many skills in working on this project, I'll try to catergorise them here:
- git and github: I've become confident using these tools and have enjoyed the incremental steps of developing and tracking the project.  I've been using git through vscode and regularly pushing to github.

- VSCode: I've also developed confidence using this software, especially using ipynb files running on a virtual environment created with conda.  It's helped develop an appreciation for understanding the different verions of python and it's libraries.

- PostgreSQL: Although I didn't use SQL in this project, being able to connect to the database and pull a table to pandas gave me an appreciation of the flexibility of data analysis through pandas.

- Python coding: While I've been using ai tools to speed up my code writing, I've developed an understanding of the structures of the code and have been able to create classes, methods and functions that give meaningful results in this context.

- Exploratory data analysis:
    -  It's important to keep the analytics goal in mind withing the context of the industry data.  I found that in trying to clean the data, it was easy to over clean it and lose some important data that might be useful in future.  For example, I removed several columns due to overly high corelation with other columns, however I later found that I needed the information from a deleted column.  By exporting the csv at each step, I was able to revert to a previous data state.  In some cases, it was more useful to edit the ipynb and rerun the scripts to alter the data as necessary.
    - I found many patterns in the data that could be investigated further, which I chose not to do due to time constraints.  In a workplace setting, it's likely guidance from stakeholders would also guide the depth at which to carry out the analysis.
- File types: Carrying out this project has given me confidence using various file types ascociated with data analysis.
- Linux: I've also been able to install and set up Ubuntu on my mac which has given me an appreciation of installing and running the software needed for this project.




