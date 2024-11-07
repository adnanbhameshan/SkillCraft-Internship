import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Step 1: Initial Data Inspection
print("First 5 rows of the dataset:\n", data.head())
print("\nData summary:\n", data.info())
print("\nMissing values per column:\n", data.isnull().sum())

# Step 2: Data Cleaning
# Filling missing values in the 'Age' column with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Filling missing values in 'Embarked' with the most common value (mode)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Dropping the 'Cabin' column due to a large number of missing values
data.drop(columns=['Cabin'], inplace=True)

# Verify missing values after cleaning
print("\nMissing values after cleaning:\n", data.isnull().sum())

# Step 3: Exploratory Data Analysis (EDA)

# Distribution of survival rate
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data, palette='Set2')
plt.title('Survival Distribution (0 = No, 1 = Yes)')
plt.show()

# Distribution by gender and survival rate
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=data, palette='Set1')
plt.title('Survival Rate by Gender')
plt.show()

# Distribution of passengers by class and survival rate
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='Set3')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age distribution by survival
plt.figure(figsize=(10, 6))
sns.histplot(data[data['Survived'] == 1]['Age'], bins=30, color='green', label='Survived', kde=True)
sns.histplot(data[data['Survived'] == 0]['Age'], bins=30, color='red', label='Not Survived', kde=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Heatmap for correlation between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Numerical Features')
plt.show()
