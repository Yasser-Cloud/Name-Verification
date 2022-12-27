from __future__ import annotations
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp.nlp import Trainer

app = FastAPI()
trainer = Trainer()



class TestingData(BaseModel):
    texts: List[str]


class StatusObject(BaseModel):
    status: str
    timestamp: str
    classes: List[str]

class PredictionObject(BaseModel):
    text: str
    predictions: Dict
    evaluation: str
    executiontime: str

class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]


@app.get("/status", summary="Get current status of the system")
def get_status():
    status = trainer.get_status()
    return StatusObject(**status)


@app.post("/predict-batch", summary="predict a batch of sentences")
def predict_batch(testing_data:TestingData):
    try:
        predictions = trainer.predict(testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/")
def home():
    return({"message": "System is up"})
