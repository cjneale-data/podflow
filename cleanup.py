from app import app, db, Chapter
from datetime import datetime, timedelta
import os

def clean_old_data():
    with app.app_context():
        # 1. Calculate the cutoff date (30 days ago)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        print(f"🧹 Running Cleanup. Deleting chapters created before: {cutoff_date}")

        # 2. Find and delete old chapters
        # We look at the 'created_at' timestamp we defined in app.py
        try:
            num_deleted = Chapter.query.filter(Chapter.created_at < cutoff_date).delete()
            db.session.commit()
            print(f"✅ Successfully deleted {num_deleted} old chapters.")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    clean_old_data()
