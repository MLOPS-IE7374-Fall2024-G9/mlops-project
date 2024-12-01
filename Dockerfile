FROM apache/airflow:2.10.2

# Switch to root user to install system packages
USER root

# Update and install git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*  # Clean up the apt cache to reduce image size

# Switch to airflow user
USER airflow

# Copy the requirements.txt file to the container
COPY requirements.txt /requirements.txt

# Copy the credentials JSON file to the container
COPY mlops-437516-b9a69694c897.json /mlops-437516-b9a69694c897.json

COPY mlops-7374-3e7424e80d76.json /mlops-7374-3e7424e80d76.json


# Install the Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Init git and dvc
RUN git init

# Configure DVC to use credentials from the JSON file
RUN dvc remote modify --local storage credentialpath /mlops-437516-b9a69694c897.json || echo "Failed to configure DVC remote"

# Run mlops config
RUN export GOOGLE_APPLICATION_CREDENTIALS=./mlops-7374-3e7424e80d76.json

# Switch to root user
USER airflow