#!/usr/bin/env python
# coding: utf-8

# Importing libraries

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
#Read the .csv file
data = pd.read_csv(r"C:\Users\ADELINE CHRISTABEL\Desktop\Crop_recommendation.csv")
#data

data.head()
data.tail()
#data.size
data.sample(10)

#data.shape
data.info()
#Check for the null values
data.isna().sum()
#if null present, drop them
data = data.dropna()
#drop duplicates, if present
data = data.drop_duplicates()

#data.shape
data.describe()
#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is a pandas DataFrame
columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]

# Define a list of unique colors (length should match or exceed the number of columns)
colors = sns.color_palette("husl", len(columns))  # 'husl' generates visually distinct colors

# Adjust number of rows and columns for subplots based on the number of columns
n_cols = 3  # Number of columns in subplot grid
n_rows = len(columns) // n_cols + (1 if len(columns) % n_cols != 0 else 0)

# Create subplots grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6))
axes = axes.flatten()

# Plot each column in the grid
for i, column in enumerate(columns):
    sns.histplot(data[column], kde=True, ax=axes[i], color=colors[i])
    axes[i].set_title(column)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Remove any empty subplots if any
for j in range(len(columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()



# Define a list of unique colors (length should match or exceed the number of columns)
colors = sns.color_palette("husl", len(columns))  # 'husl' generates visually distinct colors

# Adjust number of rows and columns for subplots based on the number of columns
n_cols = 3  # Number of columns in subplot grid
n_rows = len(columns) // n_cols + (1 if len(columns) % n_cols != 0 else 0)

# Create subplots grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
axes = axes.flatten()

# Plot each column in the grid
for i, column in enumerate(columns):
    sns.boxplot(data=data, x=column, ax=axes[i], color=colors[i])  # Set `x=column` for horizontal box plots
    axes[i].set_title(column)
    axes[i].set_xlabel("Value")  # Generic x-axis label for all plots
    axes[i].set_ylabel(column)  # Show the column name on the y-axis

# Remove any empty subplots if any
for j in range(len(columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# Filter only numeric columns from the data
numeric_data = data[columns].select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_data = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Heatmap of Correlations for Selected Columns', fontsize=16)
plt.show()


#data Preprocessing
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Select numerical columns (all columns except 'label')
numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Apply StandardScaler to the numerical columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
# Show the scaled data
print(data.head())


from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
le_label = LabelEncoder()

# Apply Label Encoding to the 'label' column (since it's categorical)
data['label'] = le_label.fit_transform(data['label'])

# Show the encoded data
print(data.head())


#Splitting the data into train and test sets and training
from sklearn.model_selection import train_test_split

# Define features and target
X = data.drop(columns=['label'])  # Drop non-relevant columns
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# Save the dataset to a CSV file without the index
data.to_csv('CropRecommendationTesting.csv', index=None)

#training, using various  ML models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    
}


results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({"Model": model_name, "MSE": mse, "R2": r2})

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print(results_df)


#  Select the best model
best_model_name = results_df.iloc[0]["Model"]
print(f"Best model: {best_model_name}")

best_model = models[best_model_name]
best_model.fit(X_train, y_train)




from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate the models (example for one model)
y_pred = best_model.predict(X_test)  # Replace `best_model` with your model variable
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")



import joblib

model_filename = f"{best_model_name}_CropRecommend_model.pkl"
joblib.dump(best_model, model_filename)
joblib.dump(le_label, 'label_encoder.pkl')
print(f"Model saved successfully as {model_filename}")

#-----Streamlit framework-----
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
model_filename = 'Decision Tree_CropRecommend_model.pkl'  # Match the saved model filename
loaded_model = joblib.load(model_filename)

# Load the label encoder (make sure it's saved separately)
le_label = joblib.load('label_encoder.pkl')  # Assuming you saved the label encoder

# Title of the app
st.title("Crop Recommendation")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file for input data", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)

        # Ensure the uploaded file has the correct columns
        required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(col in data.columns for col in required_columns):
            st.error("Uploaded file does not contain the necessary columns!")
        else:
            # Dropdown to select a row
            selected_index = st.selectbox("Select a row for prediction:", data.index)

            # Pre-fill inputs based on the selected row or manual entry
            input_data = {}
            for col in required_columns:
                if selected_index is not None:
                    input_data[col] = st.number_input(f"{col}:", 
                                                      value=float(data.loc[selected_index, col]), 
                                                      step=0.00001,
                                                      format="%.5f", 
                                                      key=col)
                else:
                    input_data[col] = st.number_input(f"{col}:", value=0.0, step=0.00001, format="%.5f", key=col)

            # Convert input data to DataFrame for prediction
            input_df = pd.DataFrame([input_data])

            # Predict button
            if st.button("Predict"):
                prediction = loaded_model.predict(input_df)
                # Decode the numerical prediction back to the categorical label
                predicted_crop = le_label.inverse_transform([int(prediction[0])])[0]
                st.write(f"Recommended Crop: {predicted_crop}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.write("Please upload a CSV file or manually enter data for prediction.")






