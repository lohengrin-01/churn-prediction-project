from fastapi import FastAPI, Body
from typing import Dict, Any

import uvicorn
import pickle
import pandas as pd
from pydantic import BaseModel, Field, conint, confloat
from typing import Literal

app = FastAPI(title="churn-prediction")

class CustomerData(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["fiber_optic", "dsl", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "two_year", "one_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)"
    ]

    tenure: conint(ge=0, le=72)  # tenure is in months, range from data
    monthlycharges: confloat(ge=18.25, le=118.75)  # from min/max in data
    totalcharges: confloat(ge=0, le=8684.8)  # from min/max in data

with open("churn_pipeline.pkl", "rb") as archive:
    model = pickle.load(archive)



customer_json = {
    "tenure": 32,
    "monthlycharges": 93.95,
    "totalcharges": 2861.45,
    "gender": "male",
    "seniorcitizen": 1,
    "partner": "yes",
    "dependents": "yes",
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "mailed_check"
}



def predict_single(customer):
    customer_df = pd.DataFrame([customer])
    result = model.predict_proba(customer_df)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: CustomerData = Body(examples=[customer_json])):
    prob = predict_single(customer.model_dump())
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)