FROM python:3.11-slim
WORKDIR /code
COPY . /code/
RUN pip install -r requirements.txt --no-cache-dir
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
