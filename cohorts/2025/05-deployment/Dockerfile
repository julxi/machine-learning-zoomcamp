# First install the python 3.8, the slim version uses less space
FROM agrigorev/zoomcamp-model:2025

# Install uv
RUN pip install uv

# Copy the uv files and python file
COPY ["pyproject.toml", "uv.lock", "./"]

# we don't need to install the dependencies,
# this is done automatically by "uv"
# when "run" is called in the entrypoint

# copy python file
COPY ["question_6_server.py", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# run the app from question_6_server.py
ENTRYPOINT ["uv", "run", "uvicorn", "question_6_server:app", "--host=0.0.0.0", "--port=9696"]