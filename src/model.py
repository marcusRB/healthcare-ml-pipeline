import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.features = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'glucose', 
                         'smoking', 'family_history', 'exercise']
        self.target = 'diabetes_risk'
    
    def train(self, df: pd.DataFrame):
        """Train a simple logistic regression model"""
        X = df[self.features]
        y = df[self.target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        X = df[self.features]
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['prediction'] = predictions
        result_df['probability'] = probabilities
        
        return result_df
    
    def save_model(self, path: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, path)
    
    def load_model(self, path: str):
        """Load model from file"""
        self.model = joblib.load(path)