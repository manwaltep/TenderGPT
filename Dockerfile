# Use an official Python runtime as a base image
FROM python:3.9

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run streamlit run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
