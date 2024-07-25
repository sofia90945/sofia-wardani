import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
data = pd.read_csv('path_to_your_dataset.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
data['MenuCategory'] = label_encoder.fit_transform(data['MenuCategory'])
data['MenuItem'] = label_encoder.fit_transform(data['MenuItem'])
data['Ingredients'] = label_encoder.fit_transform(data['Ingredients'])
data['Profitability'] = label_encoder.fit_transform(data['Profitability'])

# Define features and target
X = data[['MenuCategory', 'MenuItem', 'Ingredients', 'Price']]
y = data['Profitability']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)
