# 🌾 Crop Recommendation System 🌱

A data-driven solution to guide farmers in selecting the best crop for cultivation based on soil, weather, and nutrient parameters. This application leverages machine learning and provides an intuitive web interface built with **Streamlit**.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📋 Table of Contents:
- [✨ Features]
- [🚀 Technologies Used]
- [📂 Dataset]
- [⚡ Getting Started]
- [🎯 Usage]
- [📊 Model Training and Evaluation]
- [📈 Results]
- [🚧 Future Enhancements]
- [🙏 Acknowledgments]
- [📜 License]

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## ✨ Features
- 🌟 **Accurate Recommendations**: Predicts the most suitable crop based on soil composition and weather conditions.
- 📁 **CSV Upload or Manual Input**: Accepts data via file upload or direct input for predictions.
- 📊 **Data Visualizations**: Insightful plots like histograms, box plots, and heatmaps to analyze data distribution and correlations.
- ⚙️ **Robust Model Selection**: Evaluates multiple machine learning models and selects the best-performing one.
- 💾 **Model Persistence**: Saves the trained model and label encoder for easy reuse.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Technologies Used
- **Programming Language**: Python 
- **Machine Learning**: Scikit-learn
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: Joblib

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📂 Dataset
The dataset `Crop_recommendation.csv` contains the following features:
- **N**: Nitrogen content in the soil.
- **P**: Phosphorus content in the soil.
- **K**: Potassium content in the soil.
- **temperature**: Temperature (°C).
- **humidity**: Relative humidity (%).
- **ph**: Soil pH level.
- **rainfall**: Rainfall (mm).
- **label**: The crop label (target variable).

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## ⚡ Getting Started

### Prerequisites
- **Python** 3.8 or higher.
- Install required libraries 

### Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/upskillcampus.git
    
2. **Navigate to the directory**:
    ```bash
    cd upskillcampus
    
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🎯 Usage

### Step 1: Train the Model
Run the following script to process the dataset, train the model, and save it for future use:
```bash
python CropRecommendation.py
```
This will generate:

Decision Tree_CropRecommend_model.pkl: The trained Decision Tree Regressor model.
label_encoder.pkl: The label encoder for crop labels.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Launch the Web Application
Start the Streamlit app with:

```bash
streamlit run CropRecommendation.py
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Predict Crops:
**Upload a CSV file containing the required columns: N, P, K, temperature, humidity, ph, and rainfall.
**Alternatively, manually input the values.
**The app will display the recommended crop based on the input.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------


📊 Model Training and Evaluation:
The following machine learning models were evaluated:
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Machine (SVR)
* K-Nearest Neighbors (KNN)
* The Decision Tree Regressor was selected as the best model based on the highest R² score and lowest Mean Squared Error (MSE).
-----------------------------------------------------------------------------------------------------------------------------------------------------------------


🚧 Future Enhancements
🌐 Multi-language Support: Enable support for predictions in multiple languages.
📊 Advanced Visualizations: Add interactive graphs for better data insights.
📡 Real-time Weather Data: Integrate live weather data to make predictions even more accurate.
🎛️ Customizable Inputs: Allow users to adjust weights for various parameters.

🙏 Acknowledgments
Special thanks to the authors of the dataset used in this project.
Kudos to the developers of the amazing libraries and tools utilized.
