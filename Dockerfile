# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Hugging Face Spaces always look for traffic on port 7860
EXPOSE 7860

# Run the app using Gunicorn bound to port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
