#!/usr/bin/env python3
# api_seed.py
import requests
from datetime import datetime

API_URL = "http://localhost:8001"
SEED_NAME = "api_seed_program"

def seed_via_api():
    # Remove any prior seed entry with the same name
    del_url = f"{API_URL}/elements/by-name/{SEED_NAME}"
    try:
        del_resp = requests.delete(del_url, timeout=5)
        if del_resp.status_code == 200:
            print(f"🗑️  Removed existing entries with name '{SEED_NAME}'")
        else:
            print(f"⚠️  Delete returned {del_resp.status_code}: {del_resp.text}")
    except Exception as e:
        print(f"⚠️  Error deleting old entries: {e}")

    # Example data element for POST /elements (now with both train and test)
    element = {
    "time": datetime.utcnow().isoformat(),
    "name": "api_seed_program",
    "result": {"train": "complete", "test": "pass"},
    "program": "print('seeded via API')",
    "motivation": "ensure full result keys for sampling",
    "analysis": "Seeded element with full result keys",
    "cognition": "na",
    "log": "Test seeded via API",
    "parent": None,
    "index": None,
    "summary": "Seed program for ASI-Arch initialization"
    # "tags": ["api", "seed"]   # <-- only if your code expects this
    }
    url = API_URL + "/elements"
    try:
        resp = requests.post(url, json=element, timeout=5)
        if resp.status_code in (200, 201):
            print(f"✅ Seeded element: {resp.json()}")
        else:
            print(f"⚠️  /elements returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"⚠️  Error POST /elements: {e}")

if __name__ == "__main__":
    seed_via_api()
