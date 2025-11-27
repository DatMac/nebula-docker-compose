import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# 1. Load your service account credentials
# Ensure 'serviceAccountKey.json' is in the same folder
cred = credentials.Certificate("./serviceAccountKey.json")

# 2. Initialize the app
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://quiz-app-85db6-default-rtdb.firebaseio.com'
})

def list_tables_and_samples():
    print("--- Connecting to Firebase ---")
    
    try:
        # 3. Get a reference to the root of the database
        root_ref = db.reference('/')

        # 4. FETCH TABLE NAMES (Use shallow=True to avoid downloading all data)
        # This returns a dictionary where keys are the 'tables' and values are True
        print("Fetching database structure...")
        root_snapshot = root_ref.get(shallow=True)

        if not root_snapshot:
            print("Database is empty.")
            return

        print(f"\n✅ Found {len(root_snapshot)} 'tables' (root nodes):")
        
        # 5. LOOP THROUGH EACH TABLE TO GET SAMPLES
        for table_name in root_snapshot:
            print(f"\n------------------------------------------------")
            print(f"📂 Table: {table_name}")
            
            # Create a reference to this specific table
            table_ref = db.reference(table_name)

            # Query the first 3 entries only
            # We typically use order_by_key() before limiting
            query = table_ref.order_by_key().limit_to_first(3)
            results = query.get()

            if results:
                print(f"   Found {len(results)} sample entries:")
                for key, value in results.items():
                    print(f"   - ID: {key} | Data: {value}")
            else:
                print("   (Table is empty)")

        print("\n------------------------------------------------")
        print("Done listing.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_tables_and_samples()
