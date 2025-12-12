import psycopg2
from datetime import datetime, timedelta
from db import get_conn
from rss_parser import parse_feed

# ─────────────────────────────────────────────
# Insert or retrieve a podcast entry
# ─────────────────────────────────────────────
def insert_podcast(cur, title, author, feed_url, artwork_url, genre, topics):
    # Updated SQL to include genre and topics
    cur.execute("""
        INSERT INTO podcasts (title, author, feed_url, artwork_url, genre, topics)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (feed_url) DO UPDATE
        SET title = EXCLUDED.title,
            author = EXCLUDED.author,
            artwork_url = EXCLUDED.artwork_url,
            genre = EXCLUDED.genre,
            topics = EXCLUDED.topics
        RETURNING id;
    """, (title, author, feed_url, artwork_url, genre, topics))
    return cur.fetchone()[0]

# ─────────────────────────────────────────────
# Find an existing episode
# ─────────────────────────────────────────────
def find_existing_episode(cur, ep):
    guid = ep.get("guid")
    audio = ep.get("audio_url")
    pub_date = ep.get("pub_date")
    title = ep.get("title")

    if guid:
        cur.execute("SELECT id FROM episodes WHERE guid = %s", (guid,))
        row = cur.fetchone()
        if row: return row[0]

    if audio:
        cur.execute("SELECT id FROM episodes WHERE audio_url = %s", (audio,))
        row = cur.fetchone()
        if row: return row[0]

    if title and pub_date:
        cur.execute("SELECT id FROM episodes WHERE title = %s AND pub_date = %s", (title, pub_date))
        row = cur.fetchone()
        if row: return row[0]

    return None

# ─────────────────────────────────────────────
# Insert or update an episode
# ─────────────────────────────────────────────
def insert_or_update_episode(cur, podcast_id, ep):
    # Normalize pub_date
    pub_date = ep.get("pub_date")
    if isinstance(pub_date, str):
        try:
            pub_date = datetime.fromisoformat(pub_date)
        except Exception:
            pub_date = None

    existing_id = find_existing_episode(cur, ep)
    
    if existing_id:
        cur.execute("""
            UPDATE episodes
            SET title = %s,
                description = %s,
                pub_date = %s,
                audio_url = %s,
                duration = %s,
                guid = COALESCE(guid, %s)
            WHERE id = %s
            RETURNING id;
        """, (
            ep.get("title"),
            ep.get("description"),
            pub_date,
            ep.get("audio_url"),
            ep.get("duration"),
            ep.get("guid"),
            existing_id
        ))
        row = cur.fetchone()
        return (row[0], False) if row else (existing_id, False)

    cur.execute("""
        INSERT INTO episodes (
            podcast_id, title, description, pub_date, audio_url, guid, duration
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (
        podcast_id,
        ep.get("title"),
        ep.get("description"),
        pub_date,
        ep.get("audio_url"),
        ep.get("guid"),
        ep.get("duration")
    ))
    row = cur.fetchone()
    return (row[0], True) if row else (None, False)

# ─────────────────────────────────────────────
# Podcast Feed List (Top 50 Sports)
# ─────────────────────────────────────────────
def get_top_sports_podcasts(limit=50):
    return [
        "https://feeds.megaphone.fm/BVLLC5943241697",
        "https://feeds.megaphone.fm/the-bill-simmons-podcast", 
        "https://feeds.megaphone.fm/newheights",
        "https://feeds.megaphone.fm/ESP1539938155",
        "https://feeds.megaphone.fm/ESP4820632502",
        "https://feeds.megaphone.fm/32fans",
        "https://feeds.megaphone.fm/cffallaccess",
        "https://feeds.megaphone.fm/wfandaily",
        "https://feeds.megaphone.fm/basic",
        "https://feeds.megaphone.fm/ESP6037677737",
        "https://feeds.megaphone.fm/ESP2298543312",
        "https://feeds.megaphone.fm/papn",
        "https://feeds.megaphone.fm/gomyfavoritesportsteam",
        "https://feeds.megaphone.fm/ESP6168120254",
        "https://feeds.simplecast.com/sw7PGWfw",
        "https://mcsorleys.barstoolsports.com/feed/pardon-my-take",
        "https://feeds.simplecast.com/RkSs4uS_",
        "https://feeds.megaphone.fm/ESP3625084333",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/f7d20af0-ea4d-48fc-bc4c-b2920123b22a/95b710b5-a45b-4abe-a98a-b2920123b23f/podcast.rss",
        "https://mcsorleys.barstoolsports.com/feed/spittin-chiclets",
        "https://omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/2C906E2B-2518-466C-A457-AE320005BAFB/4818243E-950B-4FC4-8A22-AE320005BB09/podcast.rss",
        "https://feeds.megaphone.fm/ESP7297553965",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/a425ba31-4316-4647-847e-b0030136e912/0afc299b-d04e-4bd7-91a8-b0030136e938/podcast.rss",
        "https://mcsorleys.barstoolsports.com/podcasts/prPvJHG2VIlkl2YQGY4SAwgx/feed.xml",
        "https://feeds.megaphone.fm/ESP8957020927",
        "https://feeds.megaphone.fm/the-ringer-nba-show",
        "https://feeds.megaphone.fm/ESP5765452710",
        "https://feeds.megaphone.fm/CBS6458271717",
        "https://feeds.simplecast.com/06DZNq60",
        "https://feeds.megaphone.fm/ringer-fantasy-football-show",
        "https://feeds.megaphone.fm/ESP7699980268",
        "https://feeds.megaphone.fm/the-ryen-russillo-podcast",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/3bb6ed46-a795-4116-b7ef-ae3900375e77/fd0a584d-3ada-4c93-bbe1-ae3900375e85/podcast.rss",
        "https://feeds.megaphone.fm/CBS8358037229",
        "https://feeds.megaphone.fm/ESP4033296397",
        "https://feeds.megaphone.fm/the-ringer-nfl-show",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/daca7931-e14b-4914-b693-b33100f0b4dc/b0c69b90-3aca-495e-b4ef-b33100f0b4e5/podcast.rss",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/77a6df17-4a8a-4929-ac1b-b088010683dd/e3575515-668a-4eb8-8c52-b0880106843a/podcast.rss",
        "https://feeds.simplecast.com/xHwJAwNo",
        "https://feeds.simplecast.com/8PBMLPBg",
        "https://mcsorleys.barstoolsports.com/feed/fore-play",
        "https://feeds.megaphone.fm/the-jj-redick-podcast",
        "https://feeds.megaphone.fm/ESP8348692127",
        "https://feeds.megaphone.fm/ESP1539938155",
        "https://feeds.acast.com/public/shows/68b1d0ea993d10acb9c3fe4f",
        "https://mcsorleys.barstoolsports.com/feed/comeback-szn",
        "https://omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8BC7753F-B108-49B7-915F-AE280065A716/DD773BCC-2806-4AB9-9E57-AE280065A75D/podcast.rss",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/6ced0da2-f257-4234-bf7c-b07601488685/99f1543b-ae3c-4f45-90f2-b076014995d9/podcast.rss",
        "https://mcsorleys.barstoolsports.com/feed/barstool-pickem",
        "https://feeds.feedburner.com/BreakingTheHuddle",
        "https://feeds.megaphone.fm/the-tony-kornheiser-show",
        "https://feeds.megaphone.fm/nolayingup",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/0a233618-d694-4c81-a139-b34e00f98fb1/5747cc82-33a9-497c-8353-b34e00f98fbe/podcast.rss"
    ][:limit]

# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def main():
    feeds = get_top_sports_podcasts()
    conn = get_conn()
    cur = conn.cursor()
    CUT_OFF_DATE = datetime.utcnow() - timedelta(days=60)

    for feed_url in feeds:
        print(f"Processing feed: {feed_url}")

        try:
            podcast, episodes = parse_feed(feed_url)
        except Exception as e:
            print(f"❌ Error parsing feed {feed_url}: {e}")
            continue

        if not podcast.get("title"):
            print(f"❌ Skipping feed — missing title/metadata: {feed_url}")
            continue

        # ─── UPDATED CALL with genre and topics ───
        try:
            podcast_id = insert_podcast(
                cur,
                podcast["title"],
                podcast["author"],
                feed_url,
                podcast.get("artwork_url"),
                podcast.get("genre"),  # New Argument
                podcast.get("topics")  # New Argument
            )        
            conn.commit()
        except Exception as e:
            print(f"❌ Database error inserting podcast {podcast.get('title')}: {e}")
            conn.rollback()
            continue

        inserted = 0
        updated = 0
        skipped = 0
        
        for ep in episodes:
            pub_date = ep.get("pub_date")

            if not pub_date:
                continue

            if pub_date < CUT_OFF_DATE:
                continue
            
            try:
                ep_id, did_insert = insert_or_update_episode(cur, podcast_id, ep)
                if did_insert:
                    inserted += 1
                else:
                    updated += 1
                 
            except Exception as e:
                print(f"⚠️ Skipping episode due to error: {e}")
                conn.rollback()
                skipped += 1
                continue

        conn.commit()
        print(f"✅ Finished {podcast.get('title')} — inserted={inserted}, updated={updated}, genre={podcast.get('genre')}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()