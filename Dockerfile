# ---- builder: install dependencies into an isolated prefix ----
FROM python:3.11-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- runtime: copy only the installed packages + app source ----
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY main.py case_study.py ./
COPY analysis/ ./analysis/
COPY docs/ ./docs/

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "60", "main:app"]
