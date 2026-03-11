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

# This is the function to organize the column Cpu_name

def processortype(text):

    if text == 'Intel Core i5' or text == 'Intel Core i7' or text == 'Intel Core i3':
        return text

    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


dataset['Cpu_name'] = dataset['Cpu_name'].apply(lambda text:processortype(text))
dataset.head()

# Visualization of the changes made

plt.figure(figsize = (8,6))
ax = sns.countplot(data = dataset, x = dataset['Cpu_name'], palette='plasma')
plt.xticks(rotation = 'vertical')

for label in ax.containers:
    ax.bar_label(label)
plt.show()

# Inflation with respect to cpu type

plt.figure(figsize = (8,6))
sns.barplot(data = dataset, x = dataset['Cpu_name'], y = dataset['Price_euros'], palette='plasma')
plt.xticks(rotation = 'vertical')
plt.show()# Now we will drop the Cpu column

dataset.drop(columns = ['Cpu'],inplace=True)
# check the dataset

dataset.head()

# Countplot of Ram

plt.figure(figsize = (8,6))
ax = sns.countplot(data = dataset, x = dataset['Ram_GB'], palette='plasma_r')
plt.xticks(rotation = 'vertical')

for label in ax.containers:
    ax.bar_label(label)
plt.show()
# Price variations w.r.t RAM

plt.figure(figsize = (8,6))
sns.barplot(data = dataset, x = dataset['Ram_GB'], y = dataset['Price_euros'], palette='plasma_r')
plt.xticks(rotation = 'vertical')
plt.show()
# 'Memory' Column

dataset['Memory'].value_counts()

# Preocess 'Memory' column

# The two units are available - GB and TB so needs to maintain the uniformity

# Remove the decimal for example 1.0 = 1
dataset['Memory'] = dataset['Memory'].astype(str).replace('\.0','',regex=True)

# Remove GB
dataset['Memory'] = dataset['Memory'].str.replace('GB','')

# Removve TB with 000
dataset['Memory'] = dataset['Memory'].str.replace('TB','000')
dataset['Memory'].value_counts()

# Split the Memory column with +

# split the text across '+'
newdf = dataset['Memory'].str.split('+',n=1,expand=True)
newdf

dataset['Memory_first'] = newdf[0]
dataset['Memory_first'] = dataset['Memory_first'].str.strip()
dataset.head()

# Creating a function that will take memory and and create different column of memory with suffix as the type

def applychanges(value):

    dataset['Memory_first-'+value] = dataset['Memory_first'].apply(lambda x:1 if value in x else 0)

valueList = ['SSD', 'HDD', 'Hybrid', 'Flash Storage']
for value in valueList:
    applychanges(value)

dataset.sample(10)

# Remove all the characters just keep the numbers

dataset['Memory_first'] = dataset['Memory_first'].str.replace(r'\D','',regex=True)
dataset['Memory_first'].value_counts()

# Applying same thing with memory second part as applied to memory first past 

dataset['Memory_second'] = newdf[1]
dataset.head()

# Check mising values

dataset['Memory_second'].isnull().sum()
# Fill nan value with 0
dataset['Memory_second'] = dataset['Memory_second'].fillna("0")
dataset['Memory_second'].value_counts()
# Creating seperate column with memeory varients

def applychanges2(value):

    dataset['Memory_second-'+value] = dataset['Memory_second'].apply(lambda x:1 if value in x else 0)

valueList = ['SSD', 'HDD', 'Hybrid', 'Flash Storage']

for value in valueList:
    applychanges2(value)

dataset.head()
# Remove all the characters just keep the numbers
dataset['Memory_second'] = dataset['Memory_second'].str.replace(r'\D','',regex=True)
dataset['Memory_second'].value_counts()
# Converting the data type of columns Memory_first and Memory_second

dataset['Memory_first'] = dataset['Memory_first'].astype('int')
dataset['Memory_second'] = dataset['Memory_second'].astype('int')

# Check dataframe information

dataset.info()

# Multiplying the column and storing the result in sunsequent column

dataset['HDD'] = (dataset['Memory_first']*dataset['Memory_first-HDD']+dataset['Memory_second']*dataset['Memory_second-HDD'])
dataset['SSD'] = (dataset['Memory_first']*dataset['Memory_first-SSD']+dataset['Memory_second']*dataset['Memory_second-SSD'])
dataset['Hybrid'] = (dataset['Memory_first']*dataset['Memory_first-Hybrid']+dataset['Memory_second']*dataset['Memory_second-Hybrid'])
dataset['Flash Storage'] = (dataset['Memory_first']*dataset['Memory_first-Flash Storage']+dataset['Memory_second']*dataset['Memory_second-Flash Storage'])
# Check top 5 samples
dataset.head()
dataset.iloc[0]

# As we created the seperate column for memory, lets delet few columns

dataset.drop(columns = ['Memory_first','Memory_second','Memory_first-SSD', 'Memory_first-HDD', 'Memory_first-Hybrid',
                   'Memory_first-Flash Storage','Memory_second-SSD','Memory_second-HDD', 'Memory_second-Hybrid',
                   'Memory_second-Flash Storage'],inplace = True)
dataset.head()

dataset.drop(columns = ['Memory'], inplace = True)

dataset.head()

# check dataset shape

dataset.shape

# check the correlation between memeories and price

dataset[['HDD','SSD','Hybrid','Flash Storage','Price_euros']].corr()

# so drop the Hybrid and Flash Storagre column

dataset.drop(columns = ['Hybrid','Flash Storage'], inplace = True)
# Analysing 'Gpu'

dataset['Gpu'].value_counts()

# Create a column for GPU Brand

dataset['Gpu_brand'] = dataset['Gpu'].str.split(' ').apply(lambda x:x[0])
dataset['Gpu_brand'].value_counts()

plt.figure(figsize = (8,6))
sns.countplot(data = dataset, x = dataset['Gpu_brand'], palette = 'plasma_r')
plt.show()
# Remove all laptop have gpu brand 'ARM'

dataset = dataset[dataset['Gpu_brand'] != 'ARM']

plt.figure(figsize = (8,6))
sns.countplot(data = dataset, x = dataset['Gpu_brand'], palette = 'plasma_r')
plt.show()
# Inflation with respect to GPU Brand

plt.figure(figsize = (8,6))
sns.barplot(data = dataset, x = dataset['Gpu_brand'], y = dataset['Price_euros'], palette = 'plasma_r')
plt.show()
dataset.drop(columns = ['Gpu'], inplace = True)
# Operating System Analysis

dataset['OpSys'].value_counts()
# Visualization with Countplot

plt.figure(figsize = (8,6))
sns.countplot(data = dataset, x = dataset['OpSys'], palette = 'plasma_r')
plt.xticks(rotation = 'vertical')
plt.show()

# Variation in price

plt.figure(figsize = (8,6))
sns.barplot(data = dataset, x = dataset['OpSys'], y = dataset['Price_euros'], palette = 'plasma_r')
plt.xticks(rotation = 'vertical')
plt.show()

# Clubing Windows all variations

def clubWindows(text):

    if text == 'Windows 10' or text == 'Windows 10 S' or text == 'Windows 7':
        return 'Windows'
    elif text == 'Mac OS X' or text == 'macOS':
        return 'MaC'
    else:
        return 'Others'

dataset['OpSys'] = dataset['OpSys'].apply(lambda x:clubWindows(x))

dataset['OpSys'].value_counts()

# Visualization

plt.figure(figsize = (8,6))
sns.countplot(data = dataset, x = dataset['OpSys'], palette = 'plasma_r')
plt.xticks(rotation = 'vertical')
plt.show()

# Variation in price

plt.figure(figsize = (8,6))
sns.barplot(data = dataset, x = dataset['OpSys'], y = dataset['Price_euros'], palette = 'plasma_r')
plt.xticks(rotation = 'vertical')
plt.show()

# Weight Analysis

dataset['Weight_KG']

# Distribution plot

sns.histplot(dataset['Weight_KG'])
plt.show()

# Price variations w.r.t Weight

sns.scatterplot(data = dataset, x = dataset['Weight_KG'], y = dataset['Price_euros'])

# Price analysis

sns.histplot(dataset['Price_euros'])

# apply log to normaize price

sns.histplot(np.log(dataset['Price_euros']))
df_clean = dataset.copy()
df_clean

# Creating two lists to hold the numeric and categorical dataset

numF = [feature for feature in dataset.columns if dataset[feature].dtypes != 'str']
catF = [feature for feature in dataset.columns if dataset[feature].dtypes == 'str']
numF

# create a dataframe with numeric feature

dfnum = df_clean[numF]
dfnum

# checking correlation

dfnum.corr()

plt.figure(figsize = (12,6))
sns.heatmap(dfnum.corr(), annot = True)
plt.title('Correlation Heatmap')
plt.show()

# Getting my X and y

X = df_clean.drop(['Price_euros'],axis=1)

y = np.log(df_clean['Price_euros'])

