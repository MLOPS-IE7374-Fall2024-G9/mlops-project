from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.rag import RAG

# ------------------------------------------------
# inits
app = FastAPI()
rag = RAG()

# class
class QueryRequest(BaseModel):
    message: str

# end points
@app.get("/")
async def home():
    return {
        "status": "success",
    }

@app.get("/predict_demand")
async def predict_demand(location: str):
    energy_demand = rag.predict_energy_demand(location)
    weather = rag.get_weather_information_today(location)

    return {
        "status": "success",
        "energy_demand_prediction": str(energy_demand),
        "weather_information": str(weather)
    }

@app.post("/query_agent")
async def query_agent(request: QueryRequest):
    # Access the input message from the request body
    input_message = request.message
    
    # Process the input with your `rag.query_agent_v2` function
    response = rag.query_agent_v2(input_message)
    
    # Return the result
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
