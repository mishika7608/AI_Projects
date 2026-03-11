#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Read dataset
dataset = pd.read_csv("laptop_price.csv", encoding='latin-1')
dataset

# Show top 5 samples
dataset.head()

# Show last 5 samples
dataset.tail()

# Show random 5 Samples
dataset.sample(5)

# checking shape
print('DataFrame Shape : ', dataset.shape)


# Checking columns
print('DataFrame Column: ',dataset.columns)
print('\n')
print('No. of  features: ',len(dataset.columns))

# Basic information about dataset
# This method prints information about a DataFrame including the index dtype and columns, non-NA values and memory usage.
print('Laptop_Price_Prediction_Dataset: \n')
dataset.info()

# Check missing values
print('Column-wise missing values count: \n')
dataset.isnull().sum()

# Check any duplicated rows
print('No of duplicated records :', dataset.duplicated().sum())


#Categorical Data - Categorical data in pandas is a specialized data type for variables that have a limited, fixed, and discrete number of possible values (categories).
#Numerical Features - In pandas, numeric data refers to data types used to represent quantitative information that allows for mathematical operations.

# Getting categorical and numerical feature
cat_features = dataset.select_dtypes(include = 'object').columns
# alternate way : [feature for feature in df.columns if df[feature].dtypes == 'object']
num_features = dataset.select_dtypes(include = ['int32','int64','float32','float64']).columns
# alternate way : [feature for feature in df.columns if df[feature].dtypes != 'object']

# Print Categorical features
print(cat_features)

# Print Numerical features
print(num_features)

# Let's check the uniques values in the columns
print(dataset['Company'].unique())

# Function to get unique values present in cols
def uniqueValues(feature):

    print(f'Unique values in {feature} is : {dataset[feature].unique()}')

for feature in cat_features:
    uniqueValues(feature)
    print('\n')
    print('-'*100)