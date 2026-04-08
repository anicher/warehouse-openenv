FROM python:3.11-slim
WORKDIR /code
COPY . /code/
RUN pip install --no-cache-dir streamlit==1.28.0 numpy==1.24.3 gymnasium==0.29.1 openai==1.6.1
EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860 || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
