from app import app, db
from sqlalchemy import text

def init_database():
    with app.app_context():
        print(f"🔌 Connecting to database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        try:
            # This is the magic command that creates the tables
            # based on the classes defined in app.py
            db.create_all()
            print("✅ Tables created successfully!")
            
            # Verify they exist
            result = db.session.execute(text("SELECT to_regclass('public.podcasts')")).scalar()
            if result:
                print("   Verified: 'podcasts' table exists.")
            else:
                print("❌ Error: Table was not found after creation attempt.")
                
        except Exception as e:
            print(f"❌ Error creating tables: {e}")

if __name__ == "__main__":
    init_database()
