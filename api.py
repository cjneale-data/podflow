import os
import json
from flask import Flask, jsonify, abort, request
import psycopg2
from datetime import datetime, timedelta
from flask_cors import CORS

# --- Configuration ---
DB_CONFIG = {
    "dbname": "podcasts",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}

LOOKBACK_DAYS = 7 

app = Flask(__name__)
CORS(app)

# --- Database Connection Helper ---

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_filter_options():
    """Fetches unique genres and individual topics for the frontend filter UI."""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cur = conn.cursor()
        
        # 1. Get Genres
        cur.execute("SELECT DISTINCT genre FROM podcasts WHERE genre IS NOT NULL ORDER BY genre")
        genres = [row[0] for row in cur.fetchall()]
        
        # 2. Get Topics (Split comma-separated strings)
        cur.execute("SELECT topics FROM podcasts WHERE topics IS NOT NULL")
        topic_rows = cur.fetchall()
        
        unique_topics = set()
        for row in topic_rows:
            # Row is a tuple like ('Football, News',)
            if row[0]:
                # Split "Football, News" -> ["Football", "News"]
                parts = [t.strip() for t in row[0].split(',')]
                unique_topics.update(parts)
        
        return {
            "genres": genres,
            "topics": sorted(list(unique_topics))
        }
    except Exception as e:
        print(f"Error fetching filters: {e}")
        return None
    finally:
        conn.close()

def fetch_random_episode_chapters(genre_filter=None, topic_filter=None):
    """
    Fetches one random episode with optional filtering.
    """
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cur = conn.cursor()
        seven_days_ago = datetime.now() - timedelta(days=LOOKBACK_DAYS)

        # Prepare filter parameters
        # If filter is None, we pass None to SQL. 
        # The SQL logic `($2::text IS NULL OR p.genre = $2)` handles the optionality.
        
        # For Topic, we wrap in wildcards for ILIKE if it exists
        topic_param = f"%{topic_filter}%" if topic_filter else None

        # 1. Select a random episode matching criteria
        query = """
            WITH RecentEpisodesWithChapters AS (
                SELECT DISTINCT e.id
                FROM episodes e
                JOIN chapters c ON e.id = c.episode_id
                JOIN podcasts p ON e.podcast_id = p.id
                WHERE e.pub_date >= %s
                AND (%s::text IS NULL OR p.genre = %s)
                AND (%s::text IS NULL OR p.topics ILIKE %s)
            )
            SELECT e.id, p.title, e.title, e.audio_url, p.artwork_url
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            JOIN RecentEpisodesWithChapters rec ON e.id = rec.id
            ORDER BY RANDOM()
            LIMIT 1;
        """
        
        cur.execute(query, (
            seven_days_ago, 
            genre_filter, genre_filter,
            topic_filter, topic_param
        ))
        episode_data = cur.fetchone()

        if not episode_data:
            return [] 

        episode_id, podcast_title, episode_title, audio_url, artwork_url = episode_data

        # 2. Fetch chapters
        chapters_query = """
            SELECT start_time, end_time, title, confidence
            FROM chapters
            WHERE episode_id = %s
            ORDER BY start_time;
        """
        cur.execute(chapters_query, (episode_id,))
        chapters_raw = cur.fetchall()

        # 3. Structure data
        frontend_chapters = []
        for start, end, chap_title, confidence in chapters_raw:
            frontend_chapters.append({
                "id": episode_id,
                "podcast_title": podcast_title,
                "episode_title": episode_title,
                "chapter_title": chap_title,
                "artwork_url": artwork_url,
                "audio_url": audio_url,
                "start_time": start,
                "end_time": end,
                "confidence": confidence
            })

        return frontend_chapters

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        conn.close()

# --- API Routes ---

@app.route('/api/filter-options', methods=['GET'])
def filter_options_route():
    options = get_filter_options()
    if options is None:
        abort(500)
    return jsonify(options)

@app.route('/api/random-chapters', methods=['GET'])
def random_chapters_route():
    # Get filters from query parameters ?genre=Sports&topic=Football
    genre = request.args.get('genre')
    topic = request.args.get('topic')
    
    # Treat empty strings as None
    if genre == "": genre = None
    if topic == "": topic = None

    chapters = fetch_random_episode_chapters(genre, topic)
    
    if chapters is None:
        abort(500, description="Database Error")
        
    if not chapters:
        # Return empty list instead of 404 to let frontend handle "No Results" cleanly
        return jsonify([])

    return jsonify(chapters)

if __name__ == '__main__':
    print("Starting Flask API server...")
    app.run(debug=True)