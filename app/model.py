import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.features = [
            'age', 'blood_pressure', 'cholesterol', 'bmi',
            'glucose', 'smoking', 'family_history', 'exercise'
        ]
        self.target = 'diabetes_risk'

    def train(self, df: pd.DataFrame):
        """Train a logistic regression model."""
        X = df[self.features]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

        return accuracy

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = df[self.features]
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        df = df.copy()
        df['prediction'] = predictions
        df['probability'] = probabilities
        return df

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)
