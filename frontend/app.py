import streamlit as st
import requests
import json
import os

# --------------------------------------------------
def query_chat(query):
    # Send message to LLM backend
    try:
        # Call your LLM API
        response = requests.post(LLM_CHAT_URL, json={"message": query})
        if response.status_code == 200:
            llm_response = response.json().get("message", "No response")
            print(llm_response)
            return llm_response
        else:
            return "Backend Failure"
    except Exception as e:
        return str(e)

# ---------------------------------------------------

# Load configuration from config.json
current_folder = os.path.dirname(os.path.abspath(__file__))
with open(current_folder + "/config.json", "r") as config_file:
    config = json.load(config_file)

SERVER_IP = config.get("SERVER_IP", "127.0.0.1")
SERVER_PORT = config.get("SERVER_PORT", 8000)

# Construct base URLs for the APIs
DEMAND_PREDICTION_URL = f"http://{SERVER_IP}:{SERVER_PORT}/predict_demand"
LLM_CHAT_URL = f"http://{SERVER_IP}:{SERVER_PORT}/query_agent"

# Set up the sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Demand Prediction", "Chat with Energy LLM"])

# Define the Demand Prediction page
if page == "Demand Prediction":
    st.title("Electricity Demand Prediction")
    st.write("Enter a location to get the electricity demand prediction.")

    # Input for location
    location = st.text_input("Location Name", placeholder="Enter location")
    
    # Button to fetch prediction
    if st.button("Get Prediction"):
        if location:
            try:
                # Call your API
                response = requests.get(f"{DEMAND_PREDICTION_URL}?location={location}")
                if response.status_code == 200:
                    demand = response.json()
                    if "status" in demand and demand["status"] == "success":
                        st.success(f"Data fetched successfully for {location}.")
                        # Pretty-print the JSON response
                        st.json(demand)
                    else:
                        st.error(f"No data available for {location}.")
                else:
                    st.error("Failed to fetch data. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a location.")

# Define the Chat with LLM page
elif page == "Chat with Energy LLM":
    st.title("Chat with Energy LLM")
    st.write("Type your queries to interact with the energy demand LLM.")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Hello!"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Show loading spinner while waiting for the response
        with st.spinner("Waiting for response..."):
            response = query_chat(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.split("assistant:")[-1])
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
