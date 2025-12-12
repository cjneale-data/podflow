import time
import os
import gc
import torch
from datetime import datetime, timedelta
from tqdm import tqdm
from app import app, db, Episode, Chapter, load_model_artifacts, download_audio, extract_and_process_features, predict_chapters

# Configuration
BATCH_SIZE = 1 
RESET_ALL = False
LOOKBACK_DAYS = 1  # <--- New Configuration: Only process episodes from the last 7 days

def process_batch():
    with app.app_context():
        print("--- Starting Batch Processor ---")
        
        # 1. Load Model
        load_model_artifacts()
        
        # 2. Identify Episodes to Process
        # Calculate the date 7 days ago
        cutoff_date = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
        print(f"📅 Filtering for episodes published after: {cutoff_date.strftime('%Y-%m-%d')}")

        if RESET_ALL:
            print("⚠️ RESET MODE: Deleting ALL existing chapters...")
            # Note: This delete logic remains the same, but the select below will now respect the date
            try:
                # Only delete chapters for episodes in the date range if you want to be specific,
                # but RESET_ALL usually implies a hard reset. Keeping it simple here.
                num_deleted = db.session.query(Chapter).delete()
                db.session.commit()
                print(f"   Deleted {num_deleted} existing chapters.")
            except Exception as e:
                db.session.rollback()
                print(f"   Error clearing chapters: {e}")
            
            # Fetch episodes in date range
            episodes_to_process = Episode.query.filter(
                Episode.pub_date >= cutoff_date
            ).all()
        else:
            # Fetch episodes that:
            # 1. Do NOT have chapters yet
            # 2. Were published in the last 7 days
            processed_ids = db.session.query(Chapter.episode_id).distinct()
            
            episodes_to_process = Episode.query.filter(
                Episode.id.notin_(processed_ids),
                Episode.pub_date >= cutoff_date
            ).all()

        total = len(episodes_to_process)
        print(f"📋 Found {total} recent episodes pending processing.\n")
        
        if total == 0:
            print("Nothing to do. Exiting.")
            return

        # 3. Processing Loop
        success_count = 0
        error_count = 0
        
        for episode in tqdm(episodes_to_process, desc="Processing Queue", unit="ep"):
            audio_path = None
            try:
                # A. Download
                # print(f"\nDownloading: {episode.title}...") 
                if not episode.audio_url:
                    print(f"Skipping Episode {episode.id}: No Audio URL")
                    continue

                audio_path = download_audio(episode.audio_url)
                
                # B. Feature Extraction
                X_scaled, duration = extract_and_process_features(audio_path)
                
                # C. Inference
                chapters_data = predict_chapters(X_scaled, duration)
                
                # D. Save to DB
                Chapter.query.filter_by(episode_id=episode.id).delete()
                
                for c_data in chapters_data:
                    chap = Chapter(
                        episode_id=episode.id,
                        start_time=c_data['start_time'],
                        end_time=c_data['end_time'],
                        title=c_data['title'],
                        confidence=c_data['confidence']
                    )
                    db.session.add(chap)
                
                db.session.commit()
                success_count += 1
                
                # E. Cleanup Memory
                del X_scaled
                del chapters_data
                gc.collect() 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                print("\n🛑 Stopping batch process...")
                break
            except Exception as e:
                error_count += 1
                db.session.rollback()
                with open("processing_errors.log", "a") as log:
                    log.write(f"Episode ID {episode.id} Failed: {str(e)}\n")
            finally:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)

        print("\n" + "="*50)
        print("BATCH COMPLETE")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed:  {error_count}")
        print("="*50)

if __name__ == "__main__":
    process_batch()
