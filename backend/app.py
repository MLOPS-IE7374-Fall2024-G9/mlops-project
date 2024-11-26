from fastapi import FastAPI
from rag import RAG

# inits
app = FastAPI()
rag = RAG()

@app.get("/predict_demand")
async def predict_demand(input: str):
    response = rag.predict_energy_demand(input)
    return {
        "status": "success",
        "message": str(response)
    }

@app.get("/query_agent")
async def query_agent(input: str):
    response = rag.query_agent(input)
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

