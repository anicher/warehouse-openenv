FROM python:3.11-slim

WORKDIR /code

# Copy files first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Test import
RUN python -c "from environment import WarehouseEnv; print('OK')"

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
