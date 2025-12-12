import feedparser
from datetime import datetime

def parse_feed(feed_url):
    feed = feedparser.parse(feed_url)
    
    # ─── NEW: Extract Genre and Topics ────────────────────────
    # feedparser aggregates categories and keywords into 'tags'
    tags_list = feed.feed.get("tags", [])
    
    # Extract just the label/term from the tag objects
    categories = []
    for tag in tags_list:
        term = tag.get("term") or tag.get("label")
        if term:
            categories.append(term)

    # Heuristic: The first category is usually the primary Genre (e.g., "Sports")
    # The rest are treated as topics/keywords
    primary_genre = categories[0] if categories else None
    topics = ", ".join(categories[1:]) if len(categories) > 1 else None
    # ──────────────────────────────────────────────────────────

    podcast = {
        "title": feed.feed.get("title"),
        "author": feed.feed.get("author"),
        "artwork_url": feed.feed.get("image", {}).get("href"),
        "genre": primary_genre, # New field
        "topics": topics        # New field
    }

    episodes = []
    for entry in feed.entries:
        pub_date_raw = entry.get("published") or entry.get("pubDate")
        pub_date = None
        if pub_date_raw:
            try:
                # feedparser.published_parsed is a struct_time, take first 6 args for datetime
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pub_date = pub_date_raw 

        episodes.append({
            "title": entry.get("title"),
            "description": entry.get("summary"),
            "pub_date": pub_date,
            "audio_url": entry.enclosures[0].href if entry.get("enclosures") else None,
            "guid": entry.get("guid"),
            # Optional: Episodes often have specific durations in 'itunes_duration'
            "duration": entry.get("itunes_duration") 
        })

    return podcast, episodes