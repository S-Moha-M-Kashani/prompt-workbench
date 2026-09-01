FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv pip install --system .

EXPOSE 8501

# Run the Streamlit app (the product). Needs PROVIDER_API_KEY at runtime, e.g.
#   docker run --rm -p 8501:8501 -e PROVIDER_API_KEY=... prompt-workbench
CMD ["streamlit", "run", "src/prompt_workbench/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
