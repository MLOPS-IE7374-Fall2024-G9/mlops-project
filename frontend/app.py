import streamlit as st
import requests

# Set up the sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Demand Prediction", "Chat with LLM"])

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
                # Call your API (Replace `your_api_url` with the actual endpoint)
                response = requests.get(f"http://your_api_url/predict?location={location}")
                if response.status_code == 200:
                    demand = response.json().get("demand", "No data available")
                    st.success(f"Predicted Electricity Demand for {location}: {demand}")
                else:
                    st.error("Failed to fetch data. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a location.")

# Define the Chat with LLM page
elif page == "Chat with LLM":
    st.title("Chat with LLM")
    st.write("Type your queries to interact with the backend LLM.")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input for user message
    user_message = st.text_input("Your Message", placeholder="Ask something...")

    # Display chat history
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            st.write(f"**You:** {chat['user']}")
            st.write(f"**LLM:** {chat['llm']}")

    # Send message to LLM backend
    if st.button("Send"):
        if user_message:
            try:
                # Call your LLM API (Replace `your_llm_url` with the actual endpoint)
                response = requests.post("http://your_llm_url/chat", json={"message": user_message})
                if response.status_code == 200:
                    llm_response = response.json().get("response", "No response")
                    # Append user and LLM messages to chat history
                    st.session_state.chat_history.append({"user": user_message, "llm": llm_response})
                else:
                    st.error("Failed to communicate with LLM. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a message.")