FROM python:3.12-slim

WORKDIR /app

COPY . .

EXPOSE 5173

CMD ["python3", "-m", "http.server", "5173", "--bind", "0.0.0.0"]
