from pymongo import MongoClient
from bson.objectid import ObjectId
from urllib.parse import quote_plus
from datetime import datetime

# --- CONFIG ---
username = "Maximellian"
password = quote_plus("Youbicurti$1126")
database = "myapp"
host = "localhost"
port = 27017
auth_db = "admin"
mongo_uri = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource={auth_db}"
client = MongoClient(mongo_uri)
db = client[database]

# Try to fetch an eligible program from data_elements
prog = db.data_elements.find_one()
if not prog:
    print("No data_elements found — cannot seed candidate!")
    exit()
print("Found data_element for candidate seed:")
print({k: v for k, v in prog.items() if k != 'motivation_embedding'})

# --- Main candidate (score in [1, 10]) ---
candidate_obj_main = {
    "time": datetime.utcnow().isoformat(),
    "name": prog.get("name", "seed_candidate_main"),
    "result": {"train": "pass", "test": "pass"},
    "program": prog.get("program", "print('seeded main')"),
    "analysis": prog.get("analysis", "Main experiment analysis"),
    "cognition": prog.get("cognition", "initial"),
    "log": prog.get("log", "Seeded for pipeline test"),
    "motivation": prog.get("motivation", "main experiment candidate"),
    "index": prog.get("index", 1),
    "parent": prog.get("parent", None),
    "summary": prog.get("summary", "Main seeded candidate"),
    "score": 5,  # within 1..10 (main experiment)
    "status": "active",
    "created_at": prog.get("created_at", datetime.utcnow()),
    "updated_at": prog.get("updated_at", datetime.utcnow()),
}

result_main = db.candidates.insert_one(candidate_obj_main)
print(f"Inserted candidate for main with _id {result_main.inserted_id}")

# --- Ref candidate (score in [11, 50]) ---
candidate_obj_ref = {
    "time": datetime.utcnow().isoformat(),
    "name": prog.get("name", "seed_candidate_ref"),
    "result": {"train": "pass", "test": "pass"},
    "program": prog.get("program", "print('seeded ref')"),
    "analysis": prog.get("analysis", "Ref experiment analysis"),
    "cognition": prog.get("cognition", "initial"),
    "log": prog.get("log", "Seeded for pipeline test"),
    "motivation": prog.get("motivation", "reference experiment candidate"),
    "index": prog.get("index", 2),
    "parent": prog.get("parent", None),
    "summary": prog.get("summary", "Reference seeded candidate"),
    "score": 20,  # within 11..50 (ref experiment)
    "status": "active",
    "created_at": prog.get("created_at", datetime.utcnow()),
    "updated_at": prog.get("updated_at", datetime.utcnow()),
}

result_ref = db.candidates.insert_one(candidate_obj_ref)
print(f"Inserted candidate for ref with _id {result_ref.inserted_id}")

# --- Verify collection ---
print("\nCurrent candidates:")
for doc in db.candidates.find():
    print({k: v for k, v in doc.items() if k != 'motivation_embedding'})
