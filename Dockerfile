# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the Flask app code to the container
COPY . .

# Expose the port on which your Flask app runs
EXPOSE 5000

# Set the entrypoint command to run the Flask app
ENTRYPOINT ["python", "flask_app.py"]
