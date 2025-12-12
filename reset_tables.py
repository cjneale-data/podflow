from app import app, db

with app.app_context():
    print("🗑️  Dropping all tables...")
    db.drop_all()
    print("✨ Recreating all tables with correct schema...")
    db.create_all()
    print("✅ Done! Database is ready.")
