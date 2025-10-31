import pandas as pd
import numpy as np
import os

def generate_sample_data(num_records=1000):
    """Generate synthetic healthcare data."""
    np.random.seed(42)

    ages = np.clip(np.random.normal(45, 15, num_records).astype(int), 18, 90)
    blood_pressure = np.random.normal(120, 15, num_records).astype(int)
    cholesterol = np.random.normal(200, 40, num_records).astype(int)
    bmi = np.random.normal(25, 5, num_records)
    glucose = np.random.normal(100, 20, num_records)
    smoking = np.random.choice([0, 1], num_records, p=[0.7, 0.3])
    family_history = np.random.choice([0, 1], num_records, p=[0.6, 0.4])
    exercise = np.random.choice([0, 1], num_records, p=[0.4, 0.6])

    risk_score = (
        0.1 * (ages - 45)
        + 0.05 * (blood_pressure - 120)
        + 0.08 * (cholesterol - 200)
        + 0.12 * (bmi - 25)
        + 0.15 * (glucose - 100)
        + 0.3 * smoking
        + 0.25 * family_history
        - 0.2 * exercise
    )

    probability = 1 / (1 + np.exp(-risk_score))
    diabetes_risk = (probability > 0.5).astype(int)

    df = pd.DataFrame({
        'patient_id': range(1, num_records + 1),
        'age': ages,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'bmi': bmi,
        'glucose': glucose,
        'smoking': smoking,
        'family_history': family_history,
        'exercise': exercise,
        'diabetes_risk': diabetes_risk,
        'risk_probability': probability
    })

    os.makedirs('sample_data', exist_ok=True)
    df.to_csv('sample_data/healthcare_data.csv', index=False)
    print(f"Generated {num_records} sample records.")
    return df
