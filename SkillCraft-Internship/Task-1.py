import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with error handling
data = pd.read_csv("C:/Placement Training/SkillCraft-Internship/Task-1/SkillCraft_Technology.csv", on_bad_lines='skip')

# Display sample data to verify structure
print(data.head())

# Select specific data columns for visualization
# We'll take the 'Data Source' as categories and one of the population columns, e.g., '_64' (population for a specific year)
categories = data['Data Source']
population = data['_64']

# Drop rows with missing values in the population column
filtered_data = data[['Data Source', '_64']].dropna()

# Create a bar chart for the population distribution
plt.figure(figsize=(10, 6))
plt.bar(filtered_data['Data Source'], filtered_data['_64'])
plt.xlabel('Countries/Regions')
plt.ylabel('Population')
plt.title('Population Distribution by Country/Region')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
