from fastapi import FastAPI

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.rag import RAG

# inits
app = FastAPI()
rag = RAG()

@app.get("/predict_demand")
async def predict_demand(input: str):
    energy_demand = rag.predict_energy_demand(input)
    weather = rag.get_weather_information_today(input)

    return {
        "status": "success",
        "energy": str(energy_demand),
        "weather": str(weather)
    }

@app.get("/query_agent")
async def query_agent(input: str):
    response = rag.query_agent_v2(input)
    return {
        "status": "success",
        "message": str(response)
    }

@app.get("/query")
async def query(input: str):
    response = rag.query(input)
    return {
        "status": "success",
        "message": str(response)
    }

# deploy.yaml test 2
