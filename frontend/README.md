This folder contains a Streamlit application that allows users to interact with two main features:

Electricity Demand Prediction: A page where users can input a location and get the predicted electricity demand for that location.
Chat with Energy LLM: A page where users can chat with a large language model (LLM) for energy demand-related queries.
The application communicates with backend APIs for both electricity demand predictions and the LLM query responses.

## Files
- app.py: The main Streamlit application file that provides the user interface and communicates with the backend.
- config.json: Configuration file containing the backend server IP address and port.
- Dockerfile: The docker container to run streamlit
