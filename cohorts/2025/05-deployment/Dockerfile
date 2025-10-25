# First install the python 3.8, the slim version uses less space
FROM agrigorev/zoomcamp-model:2025

# Install uv
RUN pip install uv

# create a directory in Docker named app and we're using it as work directory 
#WORKDIR /app    
# wordkir is code right now

# Copy the uv files and python file
COPY ["pyproject.toml", "uv.lock", "./"]

# copy python file
COPY ["question_6_server.py", "./"]

# install the pipenv dependencies for the project and deploy them.
# RUN uv sync
# (I don't think we need that)

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# "--bind=0.0.0.0:9696",
# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["uv", "run", "uvicorn", "question_6_server:app", "--host=0.0.0.0", "--port=9696"]