import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Placement Training/SkillCraft-Internship/Task-1/SkillCraft_Technology.csv", on_bad_lines='skip')

print(data.head())


categories = data['Data Source']
population = data['_64']

filtered_data = data[['Data Source', '_64']].dropna()

plt.figure(figsize=(10, 6))
plt.bar(filtered_data['Data Source'], filtered_data['_64'])
plt.xlabel('Countries/Regions')
plt.ylabel('Population')
plt.title('Population Distribution by Country/Region')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
