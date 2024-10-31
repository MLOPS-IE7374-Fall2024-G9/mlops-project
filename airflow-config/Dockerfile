FROM apache/airflow:2.10.2

# Switch to the root user to install packages
USER airflow

# Copy the requirements.txt file to the container
COPY requirements.txt /requirements.txt

# Install the Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Switch back to the airflow user to run the server
USER root

# testing git workflow for dag folder update along with ariflow-compsoe file test 3