# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY airflow-config/requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY frontend/ /app/

# Expose the port that Streamlit runs on
EXPOSE 8501

# Start streamlit
COPY frontend/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Use a script to start Streamlit in the background
ENTRYPOINT ["/app/start.sh"]
