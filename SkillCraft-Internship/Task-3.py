import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
from pathlib import Path

# Load the Bank Marketing dataset
data = pd.read_csv("C:/Placement Training/SkillCraft-Internship/Task-3/bank/bank.csv", sep=";")

# Step 1: Inspect the Data
print("First 5 rows of the dataset:\n", data.head())
print("\nData summary:\n", data.info())
print("\nMissing values per column:\n", data.isnull().sum())

# Step 2: Data Cleaning
# Dropping any rows with missing values (if applicable)
data.dropna(inplace=True)

# Step 3: Encoding Categorical Variables
# Encode all categorical variables using Label Encoding
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Step 4: Splitting the Data into Training and Testing Sets
# Separate features and target variable
X = data.drop("y", axis=1)  # Features
y = data["y"]               # Target

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Building and Training the Decision Tree Classifier
# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Model Evaluation
# Predict on the test set
y_pred = clf.predict(X_test)

# Print accuracy, confusion matrix, and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.show()
