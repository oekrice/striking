# Use an official Python image
FROM python:3.11

# Set the working directory
WORKDIR /striking

#Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Copy the app code into the container
COPY . /striking

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#Try to remove headers
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.enableCORS=false"]