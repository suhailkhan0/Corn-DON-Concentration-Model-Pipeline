import os
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from flask import Flask, request, render_template
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the model
model = joblib.load(open("model.pkl", "rb"))

# Define a directory to save the processed files
PROCESSED_DIR = 'processed_data'
os.makedirs(PROCESSED_DIR, exist_ok=True)

def normalize_data(df):
    """Normalize the data."""
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns), scaler

def create_spectral_indices(df):
    """Create spectral indices."""
    df["mean_reflectance"] = df.mean(axis=1)
    
    # Ensure there are enough columns before selecting indices
    if df.shape[1] > 200:
        df["NDSI_1"] = (df.iloc[:, 10] - df.iloc[:, 20]) / (df.iloc[:, 10] + df.iloc[:, 20])
        df["NDSI_2"] = (df.iloc[:, 50] - df.iloc[:, 100]) / (df.iloc[:, 50] + df.iloc[:, 100])
        df["NDSI_3"] = (df.iloc[:, 150] - df.iloc[:, 200]) / (df.iloc[:, 150] + df.iloc[:, 200])
    else:
        print("Warning: Not enough columns to create NDSI indices")
    
    return df

def apply_pca(df):
    """Apply PCA to the data."""
    pca = PCA(n_components=min(10, df.shape[1]))  # Ensure we don't select more components than available
    pca_result = pca.fit_transform(df)
    
    for i in range(pca_result.shape[1]):
        df[f"PCA_{i+1}"] = pca_result[:, i]
    
    return df

@app.route("/")
def home():
    try:
        logging.info("Serving Prediction Results Page")
        return render_template("index.html")
    except Exception as e:
        logging.error(f"Error loading page: {e}")
        return "Error loading page", 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        return "Unsupported file type", 400
    
    df_cleaned = df.select_dtypes(include=['int64', 'float64'])

    if "vomitoxin_ppb" not in df_cleaned.columns:
        return "Missing target column 'vomitoxin_ppb'", 400

    # Extract and store target variable
    original_target = df_cleaned.pop("vomitoxin_ppb")

    # Normalize the data
    df_normalized, scaler = normalize_data(df_cleaned)

    # Create spectral indices
    df_with_indices = create_spectral_indices(df_normalized)

    # Apply PCA
    pca_df = apply_pca(df_with_indices)

    # Make predictions
    if pca_df.shape[1] != model.n_features_in_:
        return f"Model expects {model.n_features_in_} features, but got {pca_df.shape[1]}", 400

    original_prediction = model.predict(pca_df)

    # Calculate residuals
    residuals = original_target.values - original_prediction

    # Plot residuals and predictions vs actual
    plt.figure(figsize=(18, 12))
    
    # Residual Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    
    # Predicted vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(original_target, original_prediction, alpha=0.5)
    plt.plot([original_target.min(), original_target.max()], 
             [original_target.min(), original_target.max()], 'r--')
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Save plot as bytes
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode image in base64
    img_data = base64.b64encode(img.getvalue()).decode("utf-8")

    return render_template('result.html', predictions=original_prediction, actual=original_target, residuals=residuals, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
