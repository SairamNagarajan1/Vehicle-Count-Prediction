"""
Vehicle Count Prediction From Sensor Data
=========================================

This script implements a machine learning model to predict vehicle counts
from sensor data collected at road junctions using Random Forest Regressor.

Dataset Requirements:
- CSV file with columns: DateTime, Vehicles
- DateTime: Timestamp when data was collected
- Vehicles: Number of vehicles at that timestamp

Author: Generated based on GeeksforGeeks tutorial
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VehicleCountPredictor:
    """
    A class to predict vehicle counts from sensor data using Random Forest Regressor.
    
    This class handles data loading, feature engineering, model training, and predictions
    for vehicle count prediction based on temporal features.
    """
    
    def __init__(self):
        """Initialize the VehicleCountPredictor with default parameters."""
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def get_dom(self, dt):
        """
        Extract day of the month from datetime.
        
        Args:
            dt (datetime): Input datetime object
            
        Returns:
            int: Day of the month (1-31)
        """
        return dt.day
    
    def get_weekday(self, dt):
        """
        Extract weekday from datetime.
        
        Args:
            dt (datetime): Input datetime object
            
        Returns:
            int: Weekday number (Monday=0, Sunday=6)
        """
        return dt.weekday()
    
    def get_hour(self, dt):
        """
        Extract hour from datetime.
        
        Args:
            dt (datetime): Input datetime object
            
        Returns:
            int: Hour in 24-hour format (0-23)
        """
        return dt.hour
    
    def get_year(self, dt):
        """
        Extract year from datetime.
        
        Args:
            dt (datetime): Input datetime object
            
        Returns:
            int: Year (e.g., 2021, 2022)
        """
        return dt.year
    
    def get_month(self, dt):
        """
        Extract month from datetime.
        
        Args:
            dt (datetime): Input datetime object
            
        Returns:
            int: Month number (1-12)
        """
        return dt.month
    
    def get_dayofyear(self, dt):
        """
        Extract day of year from datetime.
        
        Args:
            dt (datetime): Input datetime object
            
        Returns:
            int: Day of year (1-366)
        """
        return dt.dayofyear
    
    def get_weekofyear(self, dt):
        """
        Extract week of year from datetime.
        
        Args:
            dt (datetime): Input datetime object
            
        Returns:
            int: Week number of the year
        """
        return dt.isocalendar()[1]  # Using isocalendar instead of deprecated weekofyear
    
    def load_and_preprocess_data(self, csv_path):
        """
        Load CSV data and perform feature engineering.
        
        Args:
            csv_path (str): Path to the CSV file containing vehicle data
            
        Returns:
            pd.DataFrame: Processed dataframe with extracted features
        """
        try:
            # Load the dataset
            print("Loading dataset...")
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Check if required columns exist
            required_cols = ['DateTime', 'Vehicles']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Display basic info about the dataset
            print("\nDataset Info:")
            print(df.info())
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Convert DateTime to datetime format
            print("\nConverting DateTime column to datetime format...")
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Feature extraction from DateTime
            print("Extracting temporal features...")
            df['date'] = df['DateTime'].map(self.get_dom)
            df['weekday'] = df['DateTime'].map(self.get_weekday)
            df['hour'] = df['DateTime'].map(self.get_hour)
            df['month'] = df['DateTime'].map(self.get_month)
            df['year'] = df['DateTime'].map(self.get_year)
            df['dayofyear'] = df['DateTime'].map(self.get_dayofyear)
            df['weekofyear'] = df['DateTime'].map(self.get_weekofyear)
            
            # Add additional useful features
            df['is_weekend'] = (df['weekday'] >= 5).astype(int)  # Saturday=5, Sunday=6
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                                 (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
            
            print("Feature extraction completed!")
            print(f"Final dataset shape: {df.shape}")
            
            return df
            
        except Exception as e:
            print(f"Error loading or preprocessing data: {str(e)}")
            raise
    
    def prepare_features_and_target(self, df):
        """
        Separate features and target variable.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            tuple: (features_df, target_series)
        """
        # Drop DateTime column and separate features from target
        df_clean = df.drop(['DateTime'], axis=1)
        features = df_clean.drop(['Vehicles'], axis=1)
        target = df_clean['Vehicles']
        
        # Store feature column names for later use
        self.feature_columns = features.columns.tolist()
        
        print(f"Features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Feature columns: {self.feature_columns}")
        
        return features, target
    
    def train_model(self, features, target, test_size=0.2, random_state=42):
        """
        Train the Random Forest Regressor model.
        
        Args:
            features (pd.DataFrame): Feature matrix
            target (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing training metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Initialize and train the model
        print("Training Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }
        
        self.is_trained = True
        
        # Print results
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED")
        print("="*50)
        print(f"Training Metrics:")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")
        print(f"  MAE:  {train_metrics['mae']:.4f}")
        print(f"  R²:   {train_metrics['r2']:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    
    def predict_single(self, date, weekday, hour, month, year, dayofyear, weekofyear):
        """
        Make a single prediction based on input features.
        
        Args:
            date (int): Day of the month
            weekday (int): Weekday (Monday=0)
            hour (int): Hour in 24-hour format
            month (int): Month number
            year (int): Year
            dayofyear (int): Day of year
            weekofyear (int): Week of year
            
        Returns:
            float: Predicted vehicle count
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create feature vector with additional features
        is_weekend = 1 if weekday >= 5 else 0
        is_rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        
        features = [date, weekday, hour, month, year, dayofyear, weekofyear, is_weekend, is_rush_hour]
        
        # Make prediction
        prediction = self.model.predict([features])[0]
        
        print(f"Input features: {dict(zip(self.feature_columns, features))}")
        print(f"Predicted vehicle count: {prediction:.2f}")
        
        return prediction
    
    def predict_from_datetime(self, datetime_str):
        """
        Make prediction from datetime string.
        
        Args:
            datetime_str (str): DateTime string (e.g., "2021-01-15 15:30:00")
            
        Returns:
            float: Predicted vehicle count
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Parse datetime
        dt = pd.to_datetime(datetime_str)
        
        # Extract features
        date = self.get_dom(dt)
        weekday = self.get_weekday(dt)
        hour = self.get_hour(dt)
        month = self.get_month(dt)
        year = self.get_year(dt)
        dayofyear = self.get_dayofyear(dt)
        weekofyear = self.get_weekofyear(dt)
        
        return self.predict_single(date, weekday, hour, month, year, dayofyear, weekofyear)
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self):
        """Plot feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")
        
        importance_df = self.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance in Vehicle Count Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to demonstrate the vehicle count prediction system.
    """
    # Initialize the predictor
    predictor = VehicleCountPredictor()
    
    # Example usage (uncomment and modify the path to your CSV file)
    """
    # Load and preprocess data
    csv_path = 'vehicles.csv'  # Replace with your actual file path
    df = predictor.load_and_preprocess_data(csv_path)
    
    # Prepare features and target
    features, target = predictor.prepare_features_and_target(df)
    
    # Train the model
    results = predictor.train_model(features, target)
    
    # Make predictions
    # Example 1: Predict for specific date and time
    prediction1 = predictor.predict_from_datetime("2021-01-15 15:30:00")
    
    # Example 2: Predict using individual features
    # predict_single(date, weekday, hour, month, year, dayofyear, weekofyear)
    prediction2 = predictor.predict_single(11, 6, 0, 1, 2015, 11, 2)
    
    # Show feature importance
    importance_df = predictor.get_feature_importance()
    print("\nFeature Importance:")
    print(importance_df)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    """
    
    print("Vehicle Count Prediction System initialized successfully!")
    print("To use this system:")
    print("1. Prepare your CSV file with 'DateTime' and 'Vehicles' columns")
    print("2. Uncomment and modify the main() function code")
    print("3. Run the script")


if __name__ == "__main__":
    main()
