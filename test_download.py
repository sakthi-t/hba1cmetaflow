#!/usr/bin/env python3
"""
Test downloading model.pkl directly from DagsHub
"""

import os
from dotenv import load_dotenv
import requests
import pickle

load_dotenv()

REPO_OWNER = "sakthi-t"
REPO_NAME = "hba1cmetaflow"
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
RUN_ID = "fb8d1d9483ef436baee58405144978e4"

print("=" * 60)
print("TESTING DIRECT MODEL DOWNLOAD FROM DAGSHUB")
print("=" * 60)

# Method 1: Try direct raw URL
print("\n1. Trying direct raw URL...")
url = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}/raw/mlflow/{RUN_ID}/artifacts/model/model.pkl"
print(f"URL: {url}")

headers = {
    "Authorization": f"Bearer {DAGSHUB_TOKEN}",
    "User-Agent": "Mozilla/5.0"
}

try:
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        # Save the model
        with open("downloaded_model.pkl", 'wb') as f:
            f.write(response.content)

        print("✓ Model downloaded successfully!")

        # Try to load it
        with open("downloaded_model.pkl", 'rb') as f:
            model = pickle.load(f)

        print(f"✓ Model loaded! Type: {type(model).__name__}")

        # Test prediction
        import numpy as np
        test_input = np.array([[1, 150, 2023, 12, 3]])
        prediction = model.predict(test_input)
        print(f"✓ Test prediction: {prediction[0]:.2f}")

    else:
        print(f"✗ Failed with status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")

except Exception as e:
    print(f"✗ Error: {e}")

# Method 2: Try API approach
print("\n2. Trying DagsHub API...")
api_url = f"https://dagshub.com/api/v1/repos/{REPO_OWNER}/{REPO_NAME}/mlflow/artifacts"
print(f"API URL: {api_url}")

params = {
    "run_id": RUN_ID,
    "path": "model/model.pkl"
}

try:
    response = requests.get(api_url, params=params, headers=headers, timeout=30)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✓ API response successful!")
        print(f"Response type: {response.headers.get('content-type')}")
    else:
        print(f"✗ API failed: {response.status_code}")

except Exception as e:
    print(f"✗ API error: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If Method 1 succeeded, you can use app_dynamic.py")
print("The app will download and cache the model automatically")
print("=" * 60)