import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Dict, Any
from sklearn.linear_model import Ridge
import optuna
from typing import Dict, Tuple, Any

# Data Processing Module
class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.original_target = self.df.pop("vomitoxin_ppb")

    def clean_data(self) -> pd.DataFrame:
        """Cleans the dataset by removing outliers and normalizing the data."""
        self.remove_outliers("vomitoxin_ppb")
        self.normalize_data()
        self.remove_anomalies("vomitoxin_ppb")
        self.create_spectral_indices()
        self.df = self.apply_pca()
        return self.df.drop(columns=["hsi_id"]), self.original_target

    def remove_outliers(self, column_name: str) -> None:
        """Removes outliers from the specified column using the IQR method."""
        Q1 = self.df[column_name].quantile(0.25)
        Q3 = self.df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column_name] >= lower_bound) & (self.df[column_name] <= upper_bound)]

    def normalize_data(self) -> None:
        """Normalizes the numerical columns in the dataframe."""
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])

    def remove_anomalies(self, column_name: str, threshold: float = 3) -> None:
        """Removes anomalies based on z-scores."""
        z_scores = np.abs((self.df[column_name] - self.df[column_name].mean()) / self.df[column_name].std())
        self.df = self.df[z_scores <= threshold]

    def create_spectral_indices(self) -> None:
        """Creates spectral indices from the spectral data."""
        self.df["NDSI_1"] = (self.df.iloc[:, 10] - self.df.iloc[:, 20]) / (self.df.iloc[:, 10] + self.df.iloc[:, 20])
        self.df["NDSI_2"] = (self.df.iloc[:, 50] - self.df.iloc[:, 100]) / (self.df.iloc[:, 50] + self.df.iloc[:, 100])
        self.df["NDSI_3"] = (self.df.iloc[:, 150] - self.df.iloc[:, 200]) / (self.df.iloc[:, 150] + self.df.iloc[:, 200])

    def apply_pca(self, n_components: int = 10) -> None:
        """Applies PCA to reduce dimensionality of the dataset."""
        pca = PCA(n_components=n_components)
        self.final_df = self.df[['NDSI_1','NDSI_2','NDSI_3']]
        pca_features = pca.fit_transform(self.df.iloc[:, :-3])  # Excluding indices
        for i in range(n_components):
            self.final_df[f"PCA_{i+1}"] = pca_features[:, i]
        return self.final_df

# Modeling Module
class ModelTrainer:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluates the model performance using MAE, RMSE, and R²."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'R²': r2}

    def train_random_forest(self) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
        """Trains a Random Forest model with hyperparameter optimization using Optuna."""
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            model = RandomForestRegressor(**params, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='neg_mean_absolute_error')
            return -scores.mean()

        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(objective_rf, n_trials=20)
        best_params_rf = study_rf.best_params

        rf_model = RandomForestRegressor(**best_params_rf, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        return rf_model, best_params_rf

    def create_ann(self, hidden_layers=(64, 32), activation='relu', learning_rate=0.001):
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation=activation, input_shape=(self.X_train.shape[1],)))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
        
        model.add(Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])
        
        return model

    def train_ann(self) -> Tuple[Sequential, Dict[str, Any]]:
        param_grid = {
            'hidden_layers': [(64, 32), (128, 64), (256, 128)],
            'activation': ['relu', 'tanh'],
            'learning_rate': [0.0001, 0.001, 0.01],
            'epochs': [50, 100],
            'batch_size': [16, 32]
        }
        
        ann_regressor = KerasRegressor(build_fn=self.create_ann, verbose=0)
        grid_search = GridSearchCV(estimator=ann_regressor, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        
        best_ann_model = self.create_ann(hidden_layers=best_params['hidden_layers'], activation=best_params['activation'], learning_rate=best_params['learning_rate'])
        best_ann_model.fit(self.X_train, self.y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(self.X_test, self.y_test), verbose=2)
        
        return best_ann_model, best_params

    def train_stacking_model(self, rf_model: RandomForestRegressor, mlp_model: MLPRegressor) -> StackingRegressor:
        """Trains a stacking model using Random Forest and MLP as base models."""
        stacking_model = StackingRegressor(
            estimators=[('rf', rf_model), ('mlp', mlp_model)],
            final_estimator=Ridge(alpha=1.0)
        )
        stacking_model.fit(self.X_train, self.y_train)
        return stacking_model

# Evaluation Module
class ModelEvaluator:
    @staticmethod
    def plot_results(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
        """Plots actual vs predicted values and residuals."""
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Ideal line
        plt.xlabel("Actual DON Concentration")
        plt.ylabel("Predicted DON Concentration")
        plt.title(f"Actual vs. Predicted Values ({model_name})")
        plt.show()

        residuals = y_true - y_pred
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, bins=30, kde=True)
        plt.axvline(0, color='r', linestyle='--')  # Zero error line
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title(f"Residual Distribution ({model_name})")
        plt.show()

    @staticmethod
    def shap_summary_plot(model: Any, X: pd.DataFrame) -> None:
        """Generates SHAP summary plot for feature importance."""
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)

# Main Execution
if __name__ == "__main__":
    # Load and process data
    processor = DataProcessor("MLE-Assignment.csv")
    cleaned_data, original_target = processor.clean_data()

    # Split data into features and target
    X = cleaned_data
    y = original_target

    # Train models
    trainer = ModelTrainer(X, y)
    rf_model, rf_params = trainer.train_random_forest()
    ann_model, ann_params = trainer.train_ann()
    stacking_model = trainer.train_stacking_model(rf_model, ann_model)

    # Evaluate models
    rf_pred = rf_model.predict(trainer.X_test)
    ann_pred = ann_model.predict(trainer.X_test)
    stacking_pred = stacking_model.predict(trainer.X_test)

    rf_results = trainer.evaluate_model(trainer.y_test, rf_pred)
    ann_results = trainer.evaluate_model(trainer.y_test, ann_pred)
    stacking_results = trainer.evaluate_model(trainer.y_test, stacking_pred)

    print("Random Forest Results:", rf_results)
    print("ANN Results:", ann_results)
    print("Stacking Model Results:", stacking_results)

    # Plot results
    ModelEvaluator.plot_results(trainer.y_test, rf_pred, "Random Forest")
    ModelEvaluator.plot_results(trainer.y_test, ann_pred, "ANN")
    ModelEvaluator.plot_results(trainer.y_test, stacking_pred, "Stacking Model")

    # SHAP analysis
    ModelEvaluator.shap_summary_plot(rf_model, X)
    ModelEvaluator.shap_summary_plot(ann_model, X)
    ModelEvaluator.shap_summary_plot(stacking_model, X)
