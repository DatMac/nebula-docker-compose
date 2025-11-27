import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# 1. Setup credentials
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "serviceAccountKey.json")
cred = credentials.Certificate(file_path)

# 2. Initialize App (No databaseURL needed for Firestore usually)
firebase_admin.initialize_app(cred)

# 3. Connect to Firestore
db = firestore.client()

def list_collections():
    print("--- Checking Firestore ---")
    # Get all root collections
    collections = db.collections()
    
    found = False
    for col in collections:
        found = True
        print(f"\n📂 Collection (Table): {col.id}")
        
        # Get first 3 documents
        docs = col.limit(3).stream()
        for doc in docs:
            print(f"   - Doc ID: {doc.id} => {doc.to_dict()}")
            
    if not found:
        print("Firestore is also empty.")

if __name__ == "__main__":
    list_collections()
