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