from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, validator, Field
from typing import List, Dict
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from app.models.bank_loan import model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory: {current_dir}")

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
    loan: int = Field(0, description="Optional: Used for validation data, not used in prediction")

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

class BulkPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

def log_unexpected_value(field: str, value: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "field": field,
        "unexpected_value": value
    }
    datadrift_dir = os.path.join(current_dir, 'datadrift')
    os.makedirs(datadrift_dir, exist_ok=True)
    log_path = os.path.join(datadrift_dir, 'drift.jsonl')
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logger.info(f"Logged unexpected value: {field}={value}")

def process_input(input_data: InputData):
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

    return input_dict

def store_predictions(inputs, predictions):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.json"
    scores_dir = os.path.join(current_dir, 'scores')
    
    # Create the scores directory if it doesn't exist
    os.makedirs(scores_dir, exist_ok=True)
    
    filepath = os.path.join(scores_dir, filename)
    logger.info(f"Attempting to store predictions in: {filepath}")
    
    result = []
    for input_data, pred in zip(inputs, predictions):
        result.append({
            "input": input_data,
            "prediction": pred.prediction,
            "probability": pred.probability
        })
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Successfully stored predictions in: {filepath}")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: InputData = Body(
        ...,
        example={
            "age": 31,
            "job": "blue-collar",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 953,
            "housing": "yes",
            "contact": "cellular",
            "day": 14,
            "month": "may",
            "duration": 479,
            "campaign": 1,
            "pdays": 346,
            "previous": 2,
            "poutcome": "success",
            "deposit": "yes"
        }
    )
):
    input_dict = process_input(input_data)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Preprocess the input
    input_preprocessed = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(input_preprocessed)
    probability = model.predict_proba(input_preprocessed)[0][1]

    result = PredictionOutput(
        prediction="Yes" if prediction[0] == 1 else "No",
        probability=float(probability)
    )

    # Store prediction
    store_predictions([input_data.dict()], [result])

    return result

@app.post("/predict_bulk", response_model=BulkPredictionOutput)
async def predict_bulk(
    bulk_input: List[InputData] = Body(
        ...,
        example=[
            {
                "age": 31,
                "job": "blue-collar",
                "marital": "single",
                "education": "secondary",
                "default": "no",
                "balance": 953,
                "housing": "yes",
                "contact": "cellular",
                "day": 14,
                "month": "may",
                "duration": 479,
                "campaign": 1,
                "pdays": 346,
                "previous": 2,
                "poutcome": "success",
                "deposit": "yes"
            },
            {
                "age": 39,
                "job": "management",
                "marital": "married",
                "education": "primary",
                "default": "no",
                "balance": 738,
                "housing": "yes",
                "contact": "cellular",
                "day": 18,
                "month": "jul",
                "duration": 215,
                "campaign": 3,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
                "deposit": "no"
            },
            {
                "age": 34,
                "job": "technician",
                "marital": "divorced",
                "education": "tertiary",
                "default": "no",
                "balance": 66,
                "housing": "no",
                "contact": "cellular",
                "day": 5,
                "month": "feb",
                "duration": 102,
                "campaign": 1,
                "pdays": 170,
                "previous": 2,
                "poutcome": "failure",
                "deposit": "no"
            }
        ]
    )
):
    processed_inputs = [process_input(input_data) for input_data in bulk_input]
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame(processed_inputs)

    # Preprocess the input
    input_preprocessed = preprocessor.transform(input_df)

    # Make predictions
    predictions = model.predict(input_preprocessed)
    probabilities = model.predict_proba(input_preprocessed)[:, 1]

    # Prepare output
    results = [
        PredictionOutput(
            prediction="Yes" if pred == 1 else "No",
            probability=float(prob)
        )
        for pred, prob in zip(predictions, probabilities)
    ]

    # Store predictions
    store_predictions([input_data.dict() for input_data in bulk_input], results)

    return BulkPredictionOutput(predictions=results)

@app.get("/scores", response_model=List[str])
async def list_scores():
    """
    List all available score files.
    """
    scores_dir = os.path.join(current_dir, 'scores')
    os.makedirs(scores_dir, exist_ok=True)
    try:
        return sorted(os.listdir(scores_dir))
    except Exception as e:
        logger.error(f"Error listing scores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing scores: {str(e)}")

@app.get("/scores/{filename}", response_model=List[Dict])
async def get_score(filename: str):
    """
    Retrieve the content of a specific score file.
    """
    scores_dir = os.path.join(current_dir, 'scores')
    file_path = os.path.join(scores_dir, filename)
    
    if not os.path.exists(file_path):
        logger.warning(f"Score file not found: {filename}")
        raise HTTPException(status_code=404, detail=f"Score file {filename} not found")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filename}")
        raise HTTPException(status_code=500, detail=f"Error decoding JSON from {filename}")
    except Exception as e:
        logger.error(f"Error reading score file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading score file: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)