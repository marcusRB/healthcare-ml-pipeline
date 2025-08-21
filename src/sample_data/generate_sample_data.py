import pandas as pd
import numpy as np
import os

def generate_sample_data(num_records=1000):
    """Generate synthetic healthcare data for demonstration"""
    np.random.seed(42)
    
    # Create synthetic patient data
    ages = np.random.normal(45, 15, num_records).astype(int)
    ages = np.clip(ages, 18, 90)
    
    blood_pressure = np.random.normal(120, 15, num_records).astype(int)
    cholesterol = np.random.normal(200, 40, num_records).astype(int)
    bmi = np.random.normal(25, 5, num_records)
    glucose = np.random.normal(100, 20, num_records)
    
    # Create some risk factors
    smoking = np.random.choice([0, 1], num_records, p=[0.7, 0.3])
    family_history = np.random.choice([0, 1], num_records, p=[0.6, 0.4])
    exercise = np.random.choice([0, 1], num_records, p=[0.4, 0.6])
    
    # Create a synthetic target variable (diabetes risk)
    # This is a simplified model for demonstration only
    risk_score = (
        0.1 * (ages - 45) + 
        0.05 * (blood_pressure - 120) + 
        0.08 * (cholesterol - 200) + 
        0.12 * (bmi - 25) + 
        0.15 * (glucose - 100) + 
        0.3 * smoking + 
        0.25 * family_history -
        0.2 * exercise
    )
    
    # Convert to probability and create binary classification
    probability = 1 / (1 + np.exp(-risk_score))
    diabetes_risk = (probability > 0.5).astype(int)
    
    # Create DataFrame
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
    
    # Save to CSV
    os.makedirs('sample_data', exist_ok=True)
    df.to_csv('sample_data/healthcare_data.csv', index=False)
    print(f"Generated {num_records} sample records")
    
    return df

if __name__ == "__main__":
    generate_sample_data()