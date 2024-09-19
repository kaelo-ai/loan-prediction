from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from app.models.bank_loan import model

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the preprocessor file
preprocessor_path = os.path.join(current_dir, '..', 'models', 'artifacts', 'preprocessor.pkl')
# Load the preprocessor
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    age: int
    balance: int
    day: int
    duration: int
    campaign: int
    pdays: int
    previous: int
    deposit: str
    housing: str
    default: str
    job: str
    marital: str
    contact: str
    poutcome: str
    education: str
    month: str

    @validator('deposit', 'housing', 'default')
    def validate_binary(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError(f"Must be 'yes' or 'no'")
        return v.lower()

    @validator('month')
    def validate_month(cls, v):
        valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if v.lower() not in valid_months:
            raise ValueError(f"Must be a valid month abbreviation")
        return v.lower()

    @validator('education')
    def validate_education(cls, v):
        valid_education = ['unknown', 'primary', 'secondary', 'tertiary']
        if v.lower() not in valid_education:
            raise ValueError(f"Must be one of {valid_education}")
        return v.lower()

    @validator('age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous')
    def validate_numeric(cls, v):
        if not isinstance(v, int):
            raise ValueError("Must be an integer")
        return v

class PredictionOutput(BaseModel):
    prediction: str
    probability: float

def log_unexpected_value(field: str, value: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "field": field,
        "unexpected_value": value
    }
    log_path = os.path.join(current_dir, 'datadrift/drift.jsonl')
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: InputData):
    input_dict = input_data.dict()
    
    # Handle potentially unexpected values
    cat_fields = ['job', 'marital', 'contact', 'poutcome']
    cat_encoder = preprocessor.named_transformers_['cat']
    
    for field in cat_fields:
        if field in cat_encoder.feature_names_in_:
            field_index = list(cat_encoder.feature_names_in_).index(field)
            valid_categories = cat_encoder.categories_[field_index]
            if input_dict[field] not in valid_categories:
                log_unexpected_value(field, input_dict[field])
                input_dict[field] = 'unknown'

    # Handle 'education' separately as it's an ordinal feature
    ord_encoder = preprocessor.named_transformers_['ord']
    valid_education = ord_encoder.categories_[0]  # Assuming education is the first ordinal feature
    if input_dict['education'] not in valid_education:
        log_unexpected_value('education', input_dict['education'])
        input_dict['education'] = 'unknown'  # or you might want to use a different default value

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Preprocess the input
    input_preprocessed = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(input_preprocessed)
    probability = model.predict_proba(input_preprocessed)[0][1]

    return PredictionOutput(
        prediction="Yes" if prediction[0] == 1 else "No",
        probability=float(probability)
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}