import pandas as pd

data= pd.read_csv("SAHeart.csv")

##Exploratory Data Analysis
print(data.head())
data=data.drop('row.names', axis=1)
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data.describe())
print(data.duplicated().sum())

##Checking for outliers
import matplotlib.pyplot as plt
data.boxplot(figsize=(12,6))
plt.xticks(rotation=45)
plt.show()


##data cleaning and encoding
# Encode the famhist column since its an object
data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})

print(data[['famhist']].head())

##data preprocessing
x=data.drop('chd', axis=1)
y=data['chd']

data.to_csv("SAHeart_clean.csv", index=False)