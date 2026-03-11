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

# Let's start with feature 'RAM' and 'Weight' : from the observations if we remove GB and kg the easialy can be converted into int.
dataset['Ram'] = dataset['Ram'].str.replace('GB','').astype('int32')

dataset['Weight'] = dataset['Weight'].str.replace('kg','').astype('float32')

# Check dtype
print(dataset.info())

# In the Ram column Ram is in GB and In weight column weight in KG 
# So, Rename the 'Ram' -- 'Ram_GB' and 'Weight' -- 'Weight_Kg'
dataset.rename(columns = {'Ram':'Ram_GB','Weight':'Weight_KG'}, inplace = True)

print(dataset.head())

# Statistical Analysis: 5-Point Summary

# Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
print(dataset.describe())
print(dataset.describe(include='all'))


# From above analysis, it is clear that 'Price' feature has some skewness in data. So let's analyse the Feature 'Price_euro'
plt.figure(figsize = (12,6))
sns.histplot(dataset['Price_euros'], color = 'red')
plt.xlabel('Price')
plt.title('Price Distribution')
plt.show()

dataset['Price_euros'].skew()
np.float64(1.5208655681688525)

for feature in ['Company', 'TypeName', 'OpSys']:
    plt.figure(figsize = (8,6))
    sns.countplot(data = dataset, x = dataset[feature], palette = 'plasma')
    plt.xticks(rotation = 'vertical')

# Inflation check with 'Company'

plt.figure(figsize = (12,6))
sns.barplot(x = dataset.Company, y = dataset.Price_euros, palette = 'plasma')
plt.title('Company vs Price')
plt.xticks(rotation = 'vertical')
plt.show()

# Inflation check with 'Company'

plt.figure(figsize = (12,6))
sns.barplot(x = dataset.Company, y = dataset.Price_euros, palette = 'plasma')
plt.title('Company vs Price')
plt.xticks(rotation = 'vertical')
plt.show()

# Visualizing the TypeName

plt.figure(figsize = (12,6))
ax = sns.countplot(data = dataset, x = dataset['TypeName'], palette = 'tab10')

for label in ax.containers:
    ax.bar_label(label)
plt.title('TypeName vs Price_euros')
plt.xticks(rotation = 'vertical')
plt.show()

plt.figure(figsize = (12,6))
ax = sns.barplot(data = dataset, x = dataset['TypeName'], y = dataset['Price_euros'], palette = 'Spectral')

for label in ax.containers:
    ax.bar_label(label, padding=35)
plt.title('TypeName vs Price_euros')
plt.xticks(rotation = 'vertical')
plt.show()

# Check ScreenResolutions

dataset['ScreenResolution'].value_counts()

# Visualization
plt.figure(figsize = (15,8))
dataset['ScreenResolution'].value_counts().plot(kind='bar')
plt.title('Barplot : ScreenResolution')
plt.show()

# Create a feature 'TouchScreen'

dataset['TouchScreen'] = dataset['ScreenResolution'].apply(lambda element:1 if 'Touchscreen' in element else 0)

dataset.head()
dataset['TouchScreen'].value_counts()
plt.figure(figsize = (8,6))
ax = sns.countplot(data = dataset, x = dataset['TouchScreen'], palette = 'plasma')
plt.title('Countplot: TouchScreen')

for label in ax.containers:
    ax.bar_label(label)

plt.show()

# Check inflation

plt.figure(figsize = (8,6))
sns.barplot(data = dataset, x = dataset['TouchScreen'], y = dataset['Price_euros'],palette = 'plasma')
plt.show()

# Create a feature 'IPS'

dataset['IPS'] = dataset['ScreenResolution'].apply(lambda element:1 if 'IPS' in element else 0)

dataset.head()

# Visualization

plt.figure(figsize = (8,6))
ax = sns.countplot(data = dataset, x = dataset['IPS'], palette = 'plasma')
plt.title('Countplot: IPS')

for label in ax.containers:
    ax.bar_label(label)
plt.show()

plt.figure(figsize = (8,6))
sns.barplot(data = dataset, x = dataset['IPS'], y = dataset['Price_euros'],palette = 'plasma')
plt.show()
# Extracting the X Resolution and Y Resolutions

dataset['ScreenResolution'].str.split('x',n=1,expand=True)

dataset['X_res'] = dataset['ScreenResolution'].str.split('x',n=1,expand=True)[0]
dataset['Y_res'] = dataset['ScreenResolution'].str.split('x',n=1,expand=True)[1]
dataset.head()

dataset['X_res'] = dataset['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x : x[0])
dataset.head()

dataset['X_res'] = dataset['X_res'].astype('int')
dataset['Y_res'] = dataset['Y_res'].astype('int')
dataset.info()

# Droping feature 'Product'

dataset.drop('Product', axis=1,inplace=True)

# Correlation

dataset_num = dataset[[feature for feature in dataset.columns if dataset[feature].dtypes != 'str']]

plt.figure(figsize = (12,8))
sns.heatmap(dataset_num .corr(), annot=True,cmap='plasma')
plt.title('Correlation Map for numerical feature')
plt.xticks(rotation=45)
plt.show()

# Let's analyze the price

dataset_num.corr()['Price_euros']

# Create a new column called PPI

dataset['PPI'] = (np.round((dataset['X_res']**2 + dataset['Y_res']**2)**(1/2))/dataset['Inches']).astype('float')
dataset.head()
# Check correlation between "Price and PPI"

dataset[['PPI','Price_euros']].corr()

# Now I have a column created using X_res, Y_res, Inches.

# So I am going to drop these columns

# Now we can delete 'Inches', 'X_res' and 'Y_res'

dataset.drop(columns = ['Inches','ScreenResolution', 'X_res','Y_res'],inplace=True)
dataset.head()

# Lets check the Cpu

dataset['Cpu'].value_counts()
# Create a column Cpu_name

dataset['Cpu_name'] = dataset['Cpu'].apply(lambda text:" ".join(text.split()[:3]))
dataset.head()

dataset['Cpu'][0].split()[:3]
dataset['Cpu_name'].value_counts()