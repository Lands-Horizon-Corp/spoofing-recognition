FROM python:3.11

# 1. Set the working directory to the container root
WORKDIR /code

# 2. Copy dependency files first (for caching)
COPY ./requirements.txt /code/requirements.txt
COPY ./pyproject.toml /code/pyproject.toml


# 3. Install dependencies
# seaparete installation of torch to avoid issues with cache and large files
RUN pip install --no-cache-dir torch torchvision torchmetrics --index-url https://download.pytorch.org/whl/cpu
# NOTE make sure to that requirements.txt does not include torch or fastapi to avoid conflicts
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src
# We also run 'pip install -e .' to respect your pyproject.toml
RUN pip install --no-cache-dir -e .



# 4. Copy the Application Code
COPY ./spoofing_detection_api /code/spoofing_detection_api

# 5. Set PYTHONPATH
# This tells Python: "Look for imports in 'src' AND 'spoofing_detection_api'"
# This fixes "ModuleNotFoundError: No module named 'app'"
ENV PYTHONPATH="${PYTHONPATH}:/code/src:/code/spoofing_detection_api"

# 6. Run the App
# We point to the nested main.py file
CMD ["fastapi", "run", "spoofing_detection_api/app/main.py", "--port", "80"]
