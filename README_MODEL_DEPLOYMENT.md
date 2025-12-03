# HBA1C Model Deployment Guide

## Current Status

### Model Information
- **Latest Run ID**: `6c03e9695c26400b8076856ef73d3d83`
- **Model URI**: `runs:/6c03e9695c26400b8076856ef73d3d83/model`
- **Metrics**: RMSE: 1.5222, MAE: 1.1693, R2: 0.6685

### DagsHub Links
- **MLflow UI**: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow
- **Models Registry**: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/models
- **Latest Run**: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/6c03e9695c26400b8076856ef73d3d83

## How to Run the App

### Option 1: Using app_simple.py (Recommended)
```bash
python3 app_simple.py
```

The app will:
1. Start Flask server
2. Load model on first prediction request (lazy loading)
3. Show warnings if model cannot be loaded

### Option 2: Using app.py
1. The model will attempt to load asynchronously at startup
2. May hang if DagsHub is slow to respond

## Updating the Model

When you run a new training:
```bash
python3 mlops_mlflow_metaflow.py run
```

You'll see output like:
```
Run ID: abc123def456...
Model URI for app.py:
  model_uri = 'runs:/abc123def456.../model'
```

Update the `LATEST_RUN_ID` in app.py with the new run ID:
```python
LATEST_RUN_ID = "abc123def456789..."  # Replace with new run ID
```

## Authentication

The app uses your `.env` file for DagsHub authentication:
```
DAGSHUB_TOKEN=your_token_here
```

## Troubleshooting

### Model Loading Issues
1. Check if the run ID is correct
2. Verify internet connection
3. Check if DagsHub is accessible
4. Try accessing the MLflow UI directly

### To Test Model Loading
```bash
python3 test_model_loading.py
```

This script tests multiple run IDs to find one that works.

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure .env file exists with DAGSHUB_TOKEN
echo "DAGSHUB_TOKEN=your_token" > .env
```

## Running with Docker (if needed)

```dockerfile
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app_simple.py"]
```

## Notes

- The DagsHub Model Registry UI shows the "hba1c" model but doesn't properly track versions
- Models are stored as artifacts in MLflow runs
- Use run-based URIs to access models: `runs:/<run-id>/model`
- The app includes error handling for when models cannot be loaded