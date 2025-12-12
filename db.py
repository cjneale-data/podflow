import psycopg2
from psycopg2.extras import RealDictCursor
import os


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "podcasts"),
        user=os.getenv("DB_USER", "myuser"),
        password=os.getenv("DB_PASSWORD", "password"),
        port=os.getenv("DB_PORT", 5432)
    )


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # podcasts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS podcasts (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            feed_url TEXT UNIQUE NOT NULL
        );
    """)

    # episodes table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id SERIAL PRIMARY KEY,
            podcast_id INTEGER REFERENCES podcasts(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            description TEXT,
            pub_date TIMESTAMP,
            audio_url TEXT
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized with podcasts and episodes tables.")


if __name__ == "__main__":
    init_db()
