import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load the dataset
url = "C:/Placement Training/SkillCraft-Internship/Task-4/US_Accidents_March23/US_Accidents_March23.csv"
data = pd.read_csv(url, on_bad_lines='skip').head(1000)  # Or on_bad_lines='skip' if using pandas >= 1.3

# Step 1: Inspect the Data
print("First 5 rows of the dataset:\n", data.head())
print("\nData summary:\n", data.info())

# Step 2: Data Cleaning
# Drop any rows with missing values in important columns
data.dropna(subset=['Start_Time', 'Weather_Condition', 'Start_Lat', 'Start_Lng'], inplace=True)

# Convert time column to datetime format for time analysis
data['Start_Time'] = pd.to_datetime(data['Start_Time'])

# Step 3: Exploratory Data Analysis (EDA)
# 3.1 Analyzing Accidents by Time of Day
data['hour'] = data['Start_Time'].dt.hour
plt.figure(figsize=(10,6))
sns.histplot(data['hour'], bins=24, kde=True, color='blue')
plt.title("Accidents by Time of Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Accidents")
plt.show()

# 3.2 Accidents by Road Conditions
# Use relevant columns like Amenity, Bump, etc. for road conditions
plt.figure(figsize=(8,5))
sns.countplot(data=data, x='Amenity', palette='Set2')
plt.title("Accidents by Road Condition (Amenity)")
plt.xlabel("Road Condition (Amenity)")
plt.ylabel("Number of Accidents")
plt.show()

# 3.3 Accidents by Weather Condition
plt.figure(figsize=(8,5))
sns.countplot(data=data, x='Weather_Condition', palette='Set3')
plt.title("Accidents by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.show()

# Step 4: Hotspot Visualization
# Create a GeoDataFrame for accident locations
geometry = [Point(xy) for xy in zip(data['Start_Lng'], data['Start_Lat'])]  # Correct the column names
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Plot accident hotspots
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
plt.figure(figsize=(15, 10))
ax = world.plot(color='white', edgecolor='black')
gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5)
plt.title("Traffic Accident Hotspots")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Step 5: Analyzing Contributing Factors
# 5.1 Accident Severity by Road Condition and Weather
# Use Severity instead of accident_severity
plt.figure(figsize=(12,6))
sns.boxplot(data=data, x='Amenity', y='Severity', hue='Weather_Condition', palette="coolwarm")
plt.title("Accident Severity by Road Condition and Weather")
plt.xlabel("Road Condition (Amenity)")
plt.ylabel("Accident Severity")
plt.legend(title='Weather Condition')
plt.show()
