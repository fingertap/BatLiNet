# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# Set the working directory in the container to /app
WORKDIR /batlinet_reproduce
# Add the current directory contents into the container at /app
ADD . /batlinet_reproduce
# Upgrade pip  
RUN pip install --upgrade pip 
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter ipykernel

# Make port 80 and 22available to the world outside this container
EXPOSE 80 22
# # Run app.py when the container launches
# CMD ["python", "app.py"]