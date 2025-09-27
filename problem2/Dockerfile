FROM python:3.11-slim

# Set working directory first
WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

# Copy your script
COPY train_embeddings.py /app/

# Set default command
ENTRYPOINT ["python", "train_embeddings.py"]
