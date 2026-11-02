import pickle
import pandas as pd

with open("churn_pipeline.pkl", "rb") as archivo:
    model = pickle.load(archivo)

customer = {
    'tenure': 32,
    'monthlycharges': 93.95,
    'totalcharges': 2861.45,
    'gender': 'Male',
    'seniorcitizen': 1,
    'partner': 'Yes',
    'dependents': 'Yes',
    'phoneservice': 'Yes',
    'multiplelines': 'No',
    'internetservice': 'Fiber optic',
    'onlinesecurity': 'No',
    'onlinebackup': 'Yes',
    'deviceprotection': 'No',
    'techsupport': 'No',
    'streamingtv': 'Yes',
    'streamingmovies': 'Yes',
    'contract': 'Month-to-month',
    'paperlessbilling': 'Yes',
    'paymentmethod': 'Mailed check'
}

customer_df = pd.DataFrame([customer])

probabilidad = model.predict_proba(customer_df)[0, 1]
prediction = model.predict(customer_df)[0]

print(f"Churn probability: {probabilidad:.2%}")
print("Prediction:", "Churn" if prediction == 1 else "Stays")