# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install git inside container
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*  # Clean up the apt cache to reduce image size

# Install dependencies for Ollama and curl
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the requirements file into the container at /app
COPY airflow-config/requirements.txt /app/

# Install the Python dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# # Run setup.sh
# RUN chmod +x ./setup.sh
# RUN ./setup.sh

# Init git and dvc
RUN git init

# # Configure DVC to use credentials from the JSON file
# RUN dvc remote modify --local storage credentialpath /mlops-437516-b9a69694c897.json
# RUN dvc pull --force

# Expose the port the app will run on
EXPOSE 8000

# Command to run the app using Uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]