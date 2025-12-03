# HBA1C Model Loading Options

## Summary
Your model is available at: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/fb8d1d9483ef436baee58405144978e4/artifacts

The model file `model.pkl` is stored in DagsHub MLflow artifacts.

## Available App Options

### 1. app_cache.py (Recommended)
```bash
python3 app_cache.py
```
**Features:**
- Downloads model ONCE and keeps it in memory
- Subsequent predictions use cached model (no network calls)
- Automatic cleanup on app exit
- Shows if model is loaded/cached
- Good balance between performance and simplicity

### 2. app.py (Current)
```bash
python3 app.py
```
**Features:**
- Downloads model to temp directory on first request
- Keeps model in memory after loading
- Has fallback model if DagsHub is inaccessible

### 3. app_mlflow.py (Alternative)
```bash
python3 app_mlflow.py
```
**Features:**
- Uses MLflow's internal loading mechanisms
- May have different download behavior
- Includes test endpoint for debugging

## How It Works

### The "No Download" Approach
Actually, MLflow ALWAYS downloads artifacts to a local location first:
- It downloads to a temporary directory (e.g., `/tmp/mlflow-xxxxx/`)
- The model is loaded from that temporary location
- The temporary files are cleaned up when done

### Our Optimization
- Keep the downloaded model in memory after first load
- Reuse the same model instance for all predictions
- Avoid repeated downloads for multiple requests

## To Update the Model

After running a new training:
```bash
python3 mlops_mlflow_metaflow.py run
```

You'll get a new Run ID. Update the `LATEST_RUN_ID` in the app file:
```python
LATEST_RUN_ID = "new-run-id-here"
```

## Troubleshooting

If the model doesn't load:
1. Check `.env` file has `DAGSHUB_TOKEN`
2. Verify Run ID is correct
3. Check internet connection
4. Visit the DagsHub URL to confirm model exists

## Dynamic Model Loading

To dynamically pick the latest model:
1. Use MLflow API to list runs
2. Find the latest run with model artifacts
3. Load that model

Note: This adds complexity and may slow down startup. For production, it's better to pin to a specific run ID.

## Performance Considerations

- **First prediction**: Slower (needs to download model, ~20-30 seconds)
- **Subsequent predictions**: Fast (model is cached in memory)
- **Memory usage**: Model stays in memory (~few MB)
- **Network usage**: Only on first load