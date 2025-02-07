# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy the application files
COPY fmnist_flask/ fmnist_flask/ 
COPY fmnist.h5 .  
COPY requirements.txt . 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Flask
EXPOSE 8083

# Command to run the Flask app (inside the package)
CMD ["python", "-m", "fmnist_flask"]
