import pickle

from typing import Any
from fastapi import FastAPI


with open("pipeline_v1.bin", "rb") as file_in:
    pipeline = pickle.load(file_in)


# print(pipeline.predict_proba(record))

app = FastAPI()


def predict_single(customer):
    result = pipeline.predict_proba(customer)
    return result[0, 1]


client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
}


@app.get("/predict_fixed")
def predict_fixed():
    prob = predict_single(client)

    return {"prob": prob}


@app.post("/predict")
def predict(client: dict[str, Any]):
    prob = predict_single(client)

    return {"prob": prob}
